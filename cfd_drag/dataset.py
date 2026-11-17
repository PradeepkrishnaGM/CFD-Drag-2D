"""Dataset generation: run sampled cases in parallel and collect masks + targets.

Layout of a dataset directory:
    config.json          seed and all parameters (a resumed run must match it)
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
import subprocess
import tempfile
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

from .params import DomainParams, PhysicsParams, SamplingParams
from .paths import SOLVER_EXE
from .sampling import Case, sample_case

RESULT_FIELDS = ["tau", "Cd", "Cl", "Cl_rms", "cd_drift", "pressure_loss", "status"]
COLUMNS = (["index", "shape", "size1", "size2", "angle", "re", "D", "blockage"]
           + RESULT_FIELDS + ["ok", "error"])


def solver_command(case: Case, out_dir: Path, domain: DomainParams, physics: PhysicsParams,
                   exe: Path = SOLVER_EXE) -> list[str]:
    return [str(exe), "--out-dir", str(out_dir),
            "--width", str(domain.nx), "--height", str(domain.ny),
            "--shape", case.shape, "--size1", repr(case.size1), "--size2", repr(case.size2),
            "--angle", repr(case.angle), "--re", repr(case.re), "--u", repr(physics.u),
            "--min-tau", repr(physics.min_tau), "--max-blockage", repr(physics.max_blockage),
            "--steady-cl-rms", repr(physics.steady_cl_rms),
            "--warmup", str(physics.warmup_steps(domain.nx)),
            "--avg", str(physics.avg_steps(domain.nx)), "--dump"]


def crop_mask(full_mask: np.ndarray, domain: DomainParams, crop: int) -> np.ndarray:
    cx, cy = domain.centre
    h = crop // 2
    return (full_mask[cy - h:cy - h + crop, cx - h:cx - h + crop] == 1).astype(np.uint8)


def run_case(case: Case, domain: DomainParams, physics: PhysicsParams,
             sampling: SamplingParams, images_dir: Path | None = None) -> tuple[dict, np.ndarray]:
    """Run one case in a private temp dir. Returns (row, cropped mask)."""
    row = {**asdict(case), "blockage": case.D / domain.H, "ok": False, "error": ""}
    row.update({k: math.nan for k in RESULT_FIELDS})
    row["status"] = ""
    mask = np.zeros((sampling.crop, sampling.crop), np.uint8)
    with tempfile.TemporaryDirectory(prefix="cfd_run_") as tmp:
        tmp = Path(tmp)
        try:
            proc = subprocess.run(solver_command(case, tmp, domain, physics),
                                  capture_output=True, text=True, timeout=domain.timeout_s)
        except subprocess.TimeoutExpired:
            row["error"] = "timeout"
            return row, mask
        if proc.returncode == 2:
            row["error"] = "rejected: " + proc.stderr.strip()
            return row, mask
        if not (tmp / "results.csv").exists():
            row["error"] = f"exit {proc.returncode}, no results"
            return row, mask
        with open(tmp / "results.csv") as f:
            res = next(csv.DictReader(f))
        for k in RESULT_FIELDS:
            row[k] = res[k] if k == "status" else float(res[k])
        finite = all(math.isfinite(row[k]) for k in ("Cd", "Cl", "Cl_rms", "pressure_loss"))
        if proc.returncode != 0 or row["status"] == "diverged" or not finite:
            row["error"] = "diverged"
        elif abs(row["Cd"]) > sampling.max_abs_cd:
            row["error"] = "|Cd| above max_abs_cd"
        else:
            row["ok"] = True
            mask = crop_mask(np.load(tmp / "mask.npy"), domain, sampling.crop)
            if images_dir is not None:
                with Image.open(tmp / "flow_field.ppm") as img:
                    img.save(images_dir / f"{case.index:06d}.png")
    return row, mask


def _sample_path(out: Path, index: int) -> Path:
    return out / "samples" / f"{index:06d}.npz"


def _worker(args) -> int:
    index, seed, out, domain, physics, sampling, images = args
    case = sample_case(seed, index, domain, physics, sampling)
    row, mask = run_case(case, domain, physics, sampling, out / "images" if images else None)
    path = _sample_path(out, index)
    tmp = path.with_name(path.stem + ".tmp.npz")
    np.savez_compressed(tmp, mask=mask, meta=np.array(json.dumps(row)))
    os.replace(tmp, path)  # atomic: a half-written file never counts as done
    return index


def _config(seed: int, domain, physics, sampling) -> dict:
    return {"seed": seed, "domain": asdict(domain), "physics": asdict(physics),
            "sampling": {k: list(v) if isinstance(v, tuple) else v
                         for k, v in asdict(sampling).items()}}


def generate(out: Path, n: int, seed: int, workers: int, domain: DomainParams,
             physics: PhysicsParams, sampling: SamplingParams, images: bool = False,
             progress: bool = True) -> pd.DataFrame:
    """Generate cases 0..n-1 (skipping ones already on disk) and consolidate."""
    if not SOLVER_EXE.exists():
        raise FileNotFoundError(f"{SOLVER_EXE} not found; run `make solver` first")
    out = Path(out)
    (out / "samples").mkdir(parents=True, exist_ok=True)
    if images:
        (out / "images").mkdir(exist_ok=True)
    cfg = _config(seed, domain, physics, sampling)
    cfg_path = out / "config.json"
    if cfg_path.exists():
        if json.loads(cfg_path.read_text()) != cfg:
            raise ValueError(f"{cfg_path} differs from the requested settings; use a new --out")
    else:
        cfg_path.write_text(json.dumps(cfg, indent=2))

    todo = [i for i in range(n) if not _sample_path(out, i).exists()]
    jobs = [(i, seed, out, domain, physics, sampling, images) for i in todo]
    if jobs:
        ctx = multiprocessing.get_context("spawn")  # fork is unsafe in threaded parents
        with ProcessPoolExecutor(max_workers=workers, mp_context=ctx) as pool:
            it = pool.map(_worker, jobs, chunksize=1)
            for _ in tqdm(it, total=len(jobs), desc="Simulating", disable=not progress):
                pass
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
