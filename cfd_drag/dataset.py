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
    fields.npy           only with fields: (N, 3, height, width) float16 time-averaged
                         (ux / U, uy / U, Cp) in the FieldParams window; zeros for failed cases

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
import threading
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

from .params import DomainParams, FieldParams, GpuParams, PhysicsParams, SamplingParams
from .paths import SOLVER_CUDA_EXE, SOLVER_EXE
from .sampling import Case, sample_case

RESULT_FIELDS = ["tau", "Cd", "Cl", "Cl_rms", "cd_drift", "pressure_loss", "status"]
COLUMNS = (["index", "shape", "size1", "size2", "angle", "re", "D", "blockage"]
           + RESULT_FIELDS + ["ok", "error"])


def global_args(domain: DomainParams, physics: PhysicsParams, mean_fields: bool = False) -> list[str]:
    """Solver options shared by every case."""
    return [*(["--mean-fields"] if mean_fields else []), "--width", str(domain.nx), "--height", str(domain.ny), "--u", repr(physics.u),
            "--min-tau", repr(physics.min_tau), "--max-blockage", repr(physics.max_blockage),
            "--steady-cl-rms", repr(physics.steady_cl_rms),
            "--warmup", str(physics.warmup_steps(domain.nx)),
            "--avg", str(physics.avg_steps(domain.nx)), "--dump",
            *(["--walls", physics.walls] if physics.walls != "noslip" else []),
            *(["--precision", physics.precision] if physics.precision not in ("double", "auto") else [])]


def resolve_precision(physics: PhysicsParams, backend: str) -> PhysicsParams:
    """precision "auto" -> "float" on the cuda back-end (3.98x faster on a T4), "double" on cpu."""
    if physics.precision != "auto":
        return physics
    return replace(physics, precision="float" if backend == "cuda" else "double")


def solver_command(case: Case, out_dir: Path, domain: DomainParams, physics: PhysicsParams,
                   exe: Path = SOLVER_EXE, mean_fields: bool = False) -> list[str]:
    return [str(exe), "--out-dir", str(out_dir), *global_args(domain, physics, mean_fields),
            "--shape", case.shape, "--size1", repr(case.size1), "--size2", repr(case.size2),
            "--angle", repr(case.angle), "--re", repr(case.re)]


def crop_mask(full_mask: np.ndarray, domain: DomainParams, crop: int) -> np.ndarray:
    cx, cy = domain.centre
    h = crop // 2
    return (full_mask[cy - h:cy - h + crop, cx - h:cx - h + crop] == 1).astype(np.uint8)


def field_window(domain: DomainParams, fields: FieldParams) -> tuple[slice, slice]:
    """(rows, cols) slices of the full lattice covered by the field window."""
    cx, cy = domain.centre
    x0, y0 = cx - fields.x_before, cy - fields.height // 2
    if x0 < 0 or x0 + fields.width > domain.nx or y0 < 1 or y0 + fields.height > domain.ny - 1:
        raise ValueError("field window does not fit inside the channel")
    return slice(y0, y0 + fields.height), slice(x0, x0 + fields.width)


def crop_field(mean: np.ndarray, domain: DomainParams, physics: PhysicsParams,
               fields: FieldParams) -> np.ndarray:
    """Solver fields_mean (ux, uy, rho) -> float16 (ux / U, uy / U, Cp) in the window."""
    rows, cols = field_window(domain, fields)
    ux, uy, rho = (m[rows, cols] for m in mean)
    u = physics.u
    cp = (rho - 1.0) / 3.0 / (0.5 * u * u)
    return np.stack([ux / u, uy / u, cp]).astype(np.float16)


def window_mask(mask: np.ndarray, domain: DomainParams, sampling: SamplingParams,
                fields: FieldParams) -> np.ndarray:
    """Place a cropped obstacle mask (crop x crop) into the field window."""
    rows, cols = field_window(domain, fields)
    cx, cy = domain.centre
    h = sampling.crop // 2
    out = np.zeros((fields.height, fields.width), mask.dtype)
    y0, x0 = cy - h - rows.start, cx - h - cols.start
    if y0 < 0 or x0 < 0 or y0 + sampling.crop > fields.height or x0 + sampling.crop > fields.width:
        raise ValueError("mask crop does not fit inside the field window")
    out[y0:y0 + sampling.crop, x0:x0 + sampling.crop] = mask
    return out


def collect_case(case: Case, case_dir: Path, error: str, domain: DomainParams,
                 sampling: SamplingParams, images_dir: Path | None,
                 physics: PhysicsParams | None = None,
                 fields: FieldParams | None = None) -> tuple[dict, np.ndarray, np.ndarray | None]:
    """Turn one case's solver output directory into (row, cropped mask, field window).

    The field is None unless `fields` (and `physics`) are given; failed cases get zeros.
    `error` is a back-end level failure (timeout, crash); it applies only if the case
    produced no results of its own.
    """
    row = {**asdict(case), "blockage": case.D / domain.H, "ok": False, "error": ""}
    row.update({k: math.nan for k in RESULT_FIELDS})
    row["status"] = ""
    mask = np.zeros((sampling.crop, sampling.crop), np.uint8)
    field = None if fields is None else np.zeros((3, fields.height, fields.width), np.float16)
    if (case_dir / "rejected.txt").exists():
        row["error"] = "rejected: " + (case_dir / "rejected.txt").read_text().strip()
        return row, mask, field
    if not (case_dir / "results.csv").exists():
        row["error"] = error or "no results"
        return row, mask, field
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
        if fields is not None:
            field = crop_field(np.load(case_dir / "fields_mean.npy"), domain, physics, fields)
            if not np.isfinite(field).all():
                row["ok"], row["error"] = False, "non-finite field"
                mask, field = np.zeros_like(mask), np.zeros_like(field)
        if images_dir is not None:
            with Image.open(case_dir / "flow_field.ppm") as img:
                img.save(images_dir / f"{case.index:06d}.png")
    return row, mask, field


def run_case(case: Case, domain: DomainParams, physics: PhysicsParams,
             sampling: SamplingParams, images_dir: Path | None = None,
             fields: FieldParams | None = None) -> tuple[dict, np.ndarray, np.ndarray | None]:
    """Run one case with the CPU solver in a private temp dir."""
    with tempfile.TemporaryDirectory(prefix="cfd_run_") as tmp:
        tmp = Path(tmp)
        error = ""
        try:
            proc = subprocess.run(solver_command(case, tmp, domain, physics,
                                                 mean_fields=fields is not None),
                                  capture_output=True, text=True, timeout=domain.timeout_s)
            if proc.returncode == 2:  # rejected before running
                (tmp / "rejected.txt").write_text(proc.stderr.strip())
            elif proc.returncode not in (0, 3):
                error = f"exit {proc.returncode}"
        except subprocess.TimeoutExpired:
            error = "timeout"
        return collect_case(case, tmp, error, domain, sampling, images_dir, physics, fields)


def run_cuda_batch(cases: list[Case], domain: DomainParams, physics: PhysicsParams,
                   sampling: SamplingParams, gpu: GpuParams, device: int | None = None,
                   images_dir: Path | None = None,
                   fields: FieldParams | None = None) -> list[tuple[dict, np.ndarray, np.ndarray | None]]:
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
                                   *global_args(domain, physics, fields is not None)],
                                  capture_output=True, text=True, timeout=gpu.timeout_s, env=env)
            if proc.returncode != 0:
                error = f"cuda exit {proc.returncode}: {proc.stderr.strip()[-300:]}"
        except subprocess.TimeoutExpired:
            error = "timeout"
        return [collect_case(c, d, error, domain, sampling, images_dir, physics, fields)
                for c, d in zip(cases, dirs)]


def _sample_path(out: Path, index: int) -> Path:
    return out / "samples" / f"{index:06d}.npz"


def _save(out: Path, index: int, row: dict, mask: np.ndarray, field: np.ndarray | None = None) -> None:
    path = _sample_path(out, index)
    tmp = path.with_name(path.stem + ".tmp.npz")
    extra = {} if field is None else {"field": field}
    np.savez_compressed(tmp, mask=mask, meta=np.array(json.dumps(row)), **extra)
    os.replace(tmp, path)  # atomic: a half-written file never counts as done


def _cpu_worker(args) -> int:
    index, seed, out, domain, physics, sampling, images, fields = args
    case = sample_case(seed, index, domain, physics, sampling)
    row, mask, field = run_case(case, domain, physics, sampling, out / "images" if images else None,
                                fields)
    _save(out, index, row, mask, field)
    return index


def count_gpus() -> int:
    try:
        out = subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True, check=True).stdout
        return max(1, sum(line.startswith("GPU ") for line in out.splitlines()))
    except (OSError, subprocess.CalledProcessError):
        return 1


def _config(seed: int, backend: str, domain, physics, sampling, fields=None) -> dict:
    ph = asdict(resolve_precision(physics, backend))
    for k, default in (("walls", "noslip"), ("precision", "double")):
        if ph[k] == default:  # keep configs of earlier datasets unchanged
            del ph[k]
    cfg = {"seed": seed, "backend": backend, "domain": asdict(domain), "physics": ph,
           "sampling": {k: list(v) if isinstance(v, tuple) else v
                        for k, v in asdict(sampling).items()}}
    if fields is not None:  # absent for scalar-only datasets (v1 configs stay unchanged)
        cfg["fields"] = asdict(fields)
    return cfg


def generate(out: Path, n: int, seed: int, workers: int, domain: DomainParams,
             physics: PhysicsParams, sampling: SamplingParams, images: bool = False,
             progress: bool = True, backend: str = "cpu", gpu: GpuParams = GpuParams(),
             gpus: int | None = None, callback=None, fields: FieldParams | None = None) -> pd.DataFrame:
    """Generate cases 0..n-1 (skipping ones already on disk) and consolidate.

    With `fields`, the solver also averages the flow field and fields.npy is written.
    `workers` applies to the cpu back-end; `gpus` (default: all visible) to cuda.
    `callback(done, total)` is called after each finished case (cpu) or batch (cuda).
    """
    exe = {"cpu": SOLVER_EXE, "cuda": SOLVER_CUDA_EXE}[backend]
    if not exe.exists():
        raise FileNotFoundError(f"{exe} not found; run `make {'solver' if backend == 'cpu' else 'cuda'}` first")
    physics = resolve_precision(physics, backend)
    out = Path(out)
    (out / "samples").mkdir(parents=True, exist_ok=True)
    images_dir = out / "images" if images else None
    if images_dir:
        images_dir.mkdir(exist_ok=True)
    if fields is not None:
        field_window(domain, fields)  # fail early if the window does not fit
    cfg = _config(seed, backend, domain, physics, sampling, fields)
    cfg_path = out / "config.json"
    if cfg_path.exists():
        if json.loads(cfg_path.read_text()) != cfg:
            raise ValueError(f"{cfg_path} differs from the requested settings; use a new --out")
    else:
        cfg_path.write_text(json.dumps(cfg, indent=2))

    todo = [i for i in range(n) if not _sample_path(out, i).exists()]
    bar = tqdm(total=len(todo), desc=f"Simulating ({backend})", disable=not progress)
    done, lock = 0, threading.Lock()  # cuda batches finish on several threads

    def advance(k):
        nonlocal done
        with lock:
            done += k
            bar.update(k)
            if callback:
                callback(done, len(todo))
    if backend == "cpu" and todo:
        jobs = [(i, seed, out, domain, physics, sampling, images, fields) for i in todo]
        ctx = multiprocessing.get_context("spawn")  # fork is unsafe in threaded parents
        with ProcessPoolExecutor(max_workers=workers, mp_context=ctx) as pool:
            for _ in pool.map(_cpu_worker, jobs, chunksize=1):
                advance(1)
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
                                         device if n_gpus > 1 else None, images_dir, fields)
            finally:
                free.put(device)
            for c, (row, mask, field) in zip(cases, results):
                _save(out, c.index, row, mask, field)
            advance(len(indices))

        chunks = [todo[i:i + gpu.batch_size] for i in range(0, len(todo), gpu.batch_size)]
        with ThreadPoolExecutor(max_workers=n_gpus) as pool:
            list(pool.map(run_chunk, chunks))
    bar.close()
    return consolidate(out, n, fields)


def consolidate(out: Path, n: int, fields: FieldParams | None = None) -> pd.DataFrame:
    rows, masks = [], []
    big = None
    if fields is not None:  # written row by row: the full array can be ~0.7 GB
        big = np.lib.format.open_memmap(out / "fields.npy.tmp", mode="w+", dtype=np.float16,
                                        shape=(n, 3, fields.height, fields.width))
    for i in range(n):
        with np.load(_sample_path(out, i)) as z:
            rows.append(json.loads(str(z["meta"])))
            masks.append(z["mask"])
            if big is not None:
                big[i] = z["field"]
    df = pd.DataFrame(rows, columns=COLUMNS)
    df.to_csv(out / "data.csv", index=False, float_format="%.10g")
    np.save(out / "masks.npy", np.stack(masks))
    if big is not None:
        big.flush()
        del big
        os.replace(out / "fields.npy.tmp", out / "fields.npy")
    return df


def load(out: Path, only_ok: bool = True) -> tuple[pd.DataFrame, np.ndarray]:
    """Load a consolidated dataset; by default only successful cases."""
    df = pd.read_csv(Path(out) / "data.csv", keep_default_na=False, na_values=["nan", "NaN"])
    masks = np.load(Path(out) / "masks.npy")
    if only_ok:
        keep = df["ok"].to_numpy(bool)
        df, masks = df[keep].reset_index(drop=True), masks[keep]
    return df, masks


def load_fields(out: Path) -> np.ndarray:
    """All rows of fields.npy (memory-mapped, float16); index with data.csv's row order."""
    return np.load(Path(out) / "fields.npy", mmap_mode="r")
