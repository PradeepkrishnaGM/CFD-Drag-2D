"""Dataset generation: run sampled cases in parallel and collect masks + targets.

Back-ends:
    cpu   one solver process per case, `workers` processes in parallel
    cuda  batches of cases per solver_cuda process, one process per GPU

Layout of a dataset directory:
    config.json          seed, back-end and all parameters (a resumed run must match it)
    samples/NNNNNN.npz   one file per case: cropped obstacle mask + metadata (resume unit)
    images/NNNNNN.png    optional velocity-magnitude image (visualisation only)
    data.csv             consolidated table, one row per case, ordered by index
    masks.npy            (N, crop, crop) uint8, row i belongs to data.csv row i

Masks are 1 on obstacle nodes, 0 elsewhere. Row index = lattice y (upwards), column = x
(downstream). Failed cases keep their row (ok = False) and an all-zero mask.
"""
import csv
import json
import math
import multiprocessing
import os
import queue
import subprocess
import tempfile
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

from .params import DomainParams, GpuParams, PhysicsParams, SamplingParams
from .paths import SOLVER_CUDA_EXE, SOLVER_EXE
from .sampling import Case, sample_case

RESULT_FIELDS = ["tau", "Cd", "Cl", "Cl_rms", "cd_drift", "pressure_loss", "status"]
COLUMNS = (["index", "shape", "size1", "size2", "angle", "re", "D", "blockage"]
           + RESULT_FIELDS + ["ok", "error"])


def global_args(domain: DomainParams, physics: PhysicsParams) -> list[str]:
    """Solver options shared by every case."""
    return ["--width", str(domain.nx), "--height", str(domain.ny), "--u", repr(physics.u),
            "--min-tau", repr(physics.min_tau), "--max-blockage", repr(physics.max_blockage),
            "--steady-cl-rms", repr(physics.steady_cl_rms),
            "--warmup", str(physics.warmup_steps(domain.nx)),
            "--avg", str(physics.avg_steps(domain.nx)), "--dump"]


def solver_command(case: Case, out_dir: Path, domain: DomainParams, physics: PhysicsParams,
                   exe: Path = SOLVER_EXE) -> list[str]:
    return [str(exe), "--out-dir", str(out_dir), *global_args(domain, physics),
            "--shape", case.shape, "--size1", repr(case.size1), "--size2", repr(case.size2),
            "--angle", repr(case.angle), "--re", repr(case.re)]


def crop_mask(full_mask: np.ndarray, domain: DomainParams, crop: int) -> np.ndarray:
    cx, cy = domain.centre
    h = crop // 2
    return (full_mask[cy - h:cy - h + crop, cx - h:cx - h + crop] == 1).astype(np.uint8)


def collect_case(case: Case, case_dir: Path, error: str, domain: DomainParams,
                 sampling: SamplingParams, images_dir: Path | None) -> tuple[dict, np.ndarray]:
    """Turn one case's solver output directory into (row, cropped mask).

    `error` is a back-end level failure (timeout, crash); it applies only if the case
    produced no results of its own.
    """
    row = {**asdict(case), "blockage": case.D / domain.H, "ok": False, "error": ""}
    row.update({k: math.nan for k in RESULT_FIELDS})
    row["status"] = ""
    mask = np.zeros((sampling.crop, sampling.crop), np.uint8)
    if (case_dir / "rejected.txt").exists():
        row["error"] = "rejected: " + (case_dir / "rejected.txt").read_text().strip()
        return row, mask
    if not (case_dir / "results.csv").exists():
        row["error"] = error or "no results"
        return row, mask
    with open(case_dir / "results.csv") as f:
        res = next(csv.DictReader(f))
    for k in RESULT_FIELDS:
        row[k] = res[k] if k == "status" else float(res[k])
    finite = all(math.isfinite(row[k]) for k in ("Cd", "Cl", "Cl_rms", "pressure_loss"))
    if row["status"] == "diverged" or not finite:
        row["error"] = "diverged"
    elif abs(row["Cd"]) > sampling.max_abs_cd:
        row["error"] = "|Cd| above max_abs_cd"
    else:
        row["ok"] = True
        mask = crop_mask(np.load(case_dir / "mask.npy"), domain, sampling.crop)
        if images_dir is not None:
            with Image.open(case_dir / "flow_field.ppm") as img:
                img.save(images_dir / f"{case.index:06d}.png")
    return row, mask


def run_case(case: Case, domain: DomainParams, physics: PhysicsParams,
             sampling: SamplingParams, images_dir: Path | None = None) -> tuple[dict, np.ndarray]:
    """Run one case with the CPU solver in a private temp dir."""
    with tempfile.TemporaryDirectory(prefix="cfd_run_") as tmp:
        tmp = Path(tmp)
        error = ""
        try:
            proc = subprocess.run(solver_command(case, tmp, domain, physics),
                                  capture_output=True, text=True, timeout=domain.timeout_s)
            if proc.returncode == 2:  # rejected before running
                (tmp / "rejected.txt").write_text(proc.stderr.strip())
            elif proc.returncode not in (0, 3):
                error = f"exit {proc.returncode}"
        except subprocess.TimeoutExpired:
            error = "timeout"
        return collect_case(case, tmp, error, domain, sampling, images_dir)


def run_cuda_batch(cases: list[Case], domain: DomainParams, physics: PhysicsParams,
                   sampling: SamplingParams, gpu: GpuParams, device: int | None = None,
                   images_dir: Path | None = None) -> list[tuple[dict, np.ndarray]]:
    """Run a batch of cases in one solver_cuda process (optionally pinned to `device`)."""
    with tempfile.TemporaryDirectory(prefix="cfd_batch_") as tmp:
        tmp = Path(tmp)
        dirs = [tmp / f"{c.index:06d}" for c in cases]
        with open(tmp / "cases.csv", "w") as f:
            f.write("out_dir,shape,size1,size2,angle,re\n")
            for c, d in zip(cases, dirs):
                d.mkdir()
                f.write(f"{d},{c.shape},{c.size1!r},{c.size2!r},{c.angle!r},{c.re!r}\n")
        env = dict(os.environ)
        if device is not None:
            env["CUDA_VISIBLE_DEVICES"] = str(device)
        error = ""
        try:
            proc = subprocess.run([str(SOLVER_CUDA_EXE), "--batch", str(tmp / "cases.csv"),
                                   *global_args(domain, physics)],
                                  capture_output=True, text=True, timeout=gpu.timeout_s, env=env)
            if proc.returncode != 0:
                error = f"cuda exit {proc.returncode}: {proc.stderr.strip()[-300:]}"
        except subprocess.TimeoutExpired:
            error = "timeout"
        return [collect_case(c, d, error, domain, sampling, images_dir) for c, d in zip(cases, dirs)]


def _sample_path(out: Path, index: int) -> Path:
    return out / "samples" / f"{index:06d}.npz"


def _save(out: Path, index: int, row: dict, mask: np.ndarray) -> None:
    path = _sample_path(out, index)
    tmp = path.with_name(path.stem + ".tmp.npz")
    np.savez_compressed(tmp, mask=mask, meta=np.array(json.dumps(row)))
    os.replace(tmp, path)  # atomic: a half-written file never counts as done


def _cpu_worker(args) -> int:
    index, seed, out, domain, physics, sampling, images = args
    case = sample_case(seed, index, domain, physics, sampling)
    row, mask = run_case(case, domain, physics, sampling, out / "images" if images else None)
    _save(out, index, row, mask)
    return index


def count_gpus() -> int:
    try:
        out = subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True, check=True).stdout
        return max(1, sum(line.startswith("GPU ") for line in out.splitlines()))
    except (OSError, subprocess.CalledProcessError):
        return 1


def _config(seed: int, backend: str, domain, physics, sampling) -> dict:
    return {"seed": seed, "backend": backend, "domain": asdict(domain), "physics": asdict(physics),
            "sampling": {k: list(v) if isinstance(v, tuple) else v
                         for k, v in asdict(sampling).items()}}


def generate(out: Path, n: int, seed: int, workers: int, domain: DomainParams,
             physics: PhysicsParams, sampling: SamplingParams, images: bool = False,
             progress: bool = True, backend: str = "cpu", gpu: GpuParams = GpuParams(),
             gpus: int | None = None) -> pd.DataFrame:
    """Generate cases 0..n-1 (skipping ones already on disk) and consolidate.

    `workers` applies to the cpu back-end; `gpus` (default: all visible) to cuda.
    """
    exe = {"cpu": SOLVER_EXE, "cuda": SOLVER_CUDA_EXE}[backend]
    if not exe.exists():
        raise FileNotFoundError(f"{exe} not found; run `make {'solver' if backend == 'cpu' else 'cuda'}` first")
    out = Path(out)
    (out / "samples").mkdir(parents=True, exist_ok=True)
    images_dir = out / "images" if images else None
    if images_dir:
        images_dir.mkdir(exist_ok=True)
    cfg = _config(seed, backend, domain, physics, sampling)
    cfg_path = out / "config.json"
    if cfg_path.exists():
        if json.loads(cfg_path.read_text()) != cfg:
            raise ValueError(f"{cfg_path} differs from the requested settings; use a new --out")
    else:
        cfg_path.write_text(json.dumps(cfg, indent=2))

    todo = [i for i in range(n) if not _sample_path(out, i).exists()]
    bar = tqdm(total=len(todo), desc=f"Simulating ({backend})", disable=not progress)
    if backend == "cpu" and todo:
        jobs = [(i, seed, out, domain, physics, sampling, images) for i in todo]
        ctx = multiprocessing.get_context("spawn")  # fork is unsafe in threaded parents
        with ProcessPoolExecutor(max_workers=workers, mp_context=ctx) as pool:
            for _ in pool.map(_cpu_worker, jobs, chunksize=1):
                bar.update(1)
    elif backend == "cuda" and todo:
        n_gpus = gpus or count_gpus()
        free = queue.Queue()
        for d in range(n_gpus):
            free.put(d)

        def run_chunk(indices):
            device = free.get()  # one batch per GPU at a time
            try:
                cases = [sample_case(seed, i, domain, physics, sampling) for i in indices]
                results = run_cuda_batch(cases, domain, physics, sampling, gpu,
                                         device if n_gpus > 1 else None, images_dir)
            finally:
                free.put(device)
            for c, (row, mask) in zip(cases, results):
                _save(out, c.index, row, mask)
            bar.update(len(indices))

        chunks = [todo[i:i + gpu.batch_size] for i in range(0, len(todo), gpu.batch_size)]
        with ThreadPoolExecutor(max_workers=n_gpus) as pool:
            list(pool.map(run_chunk, chunks))
    bar.close()
    return consolidate(out, n)


def consolidate(out: Path, n: int) -> pd.DataFrame:
    rows, masks = [], []
    for i in range(n):
        with np.load(_sample_path(out, i)) as z:
            rows.append(json.loads(str(z["meta"])))
            masks.append(z["mask"])
    df = pd.DataFrame(rows, columns=COLUMNS)
    df.to_csv(out / "data.csv", index=False, float_format="%.10g")
    np.save(out / "masks.npy", np.stack(masks))
    return df


def load(out: Path, only_ok: bool = True) -> tuple[pd.DataFrame, np.ndarray]:
    """Load a consolidated dataset; by default only successful cases."""
    df = pd.read_csv(Path(out) / "data.csv", keep_default_na=False, na_values=["nan", "NaN"])
    masks = np.load(Path(out) / "masks.npy")
    if only_ok:
        keep = df["ok"].to_numpy(bool)
        df, masks = df[keep].reset_index(drop=True), masks[keep]
    return df, masks
