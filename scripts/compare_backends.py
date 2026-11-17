"""Compare the CPU and CUDA solvers on the same sampled cases, and time the CUDA one.

Writes <out>/summary.json. Intended for a GPU machine (Kaggle), e.g.:
    python scripts/compare_backends.py --n 16 --bench 128 --out /kaggle/working/m3

Agreement is expected to be tight for steady cases. For periodic (vortex-shedding) cases
round-off differences shift the shedding phase slightly, so time-averaged values agree
less tightly; both are reported separately.
"""
import argparse
import json
import os
import platform
import subprocess
import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from cfd_drag.dataset import generate  # noqa: E402
from cfd_drag.params import DOMAIN, GPU, PHYSICS, SAMPLING  # noqa: E402


def rel_diff(a, b):
    return np.abs(a - b) / np.maximum(np.abs(b), 1e-12)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=16, help="cases for the CPU/GPU comparison")
    ap.add_argument("--bench", type=int, default=128, help="cases for the CUDA throughput run (0: skip)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=os.cpu_count())
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    out = args.out
    summary = {"python": platform.python_version(), "cpu_count": os.cpu_count()}
    try:
        summary["gpu"] = subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True).stdout.strip()
    except OSError:
        summary["gpu"] = "none"

    t = time.time()
    cpu = generate(out / "cpu", args.n, args.seed, args.workers, DOMAIN, PHYSICS, SAMPLING, progress=False)
    summary["cpu_seconds"] = round(time.time() - t, 1)
    t = time.time()
    gpu = generate(out / "cuda", args.n, args.seed, 1, DOMAIN, PHYSICS, SAMPLING, progress=False,
                   backend="cuda", gpu=GPU)
    summary["cuda_seconds"] = round(time.time() - t, 1)

    ok = cpu["ok"].to_numpy(bool) & gpu["ok"].to_numpy(bool)
    summary["n"] = args.n
    summary["ok_cpu"], summary["ok_cuda"] = int(cpu["ok"].sum()), int(gpu["ok"].sum())
    summary["status_agree"] = int((cpu["status"] == gpu["status"]).sum())
    masks_equal = np.array_equal(np.load(out / "cpu/masks.npy"), np.load(out / "cuda/masks.npy"))
    summary["masks_equal"] = bool(masks_equal)
    for regime in ("steady", "periodic"):
        sel = ok & (cpu["status"] == regime).to_numpy()
        stats = {"count": int(sel.sum())}
        if sel.any():
            for k in ("Cd", "Cl_rms", "pressure_loss"):
                stats[f"{k}_max_rel_diff"] = float(rel_diff(gpu.loc[sel, k].to_numpy(), cpu.loc[sel, k].to_numpy()).max())
            stats["Cl_max_abs_diff"] = float(np.abs(gpu.loc[sel, "Cl"] - cpu.loc[sel, "Cl"]).max())
        summary[regime] = stats

    # Determinism: same cases, different batch size -> identical data.csv
    t = time.time()
    generate(out / "cuda_b3", args.n, args.seed, 1, DOMAIN, PHYSICS, SAMPLING, progress=False,
             backend="cuda", gpu=replace(GPU, batch_size=3))
    summary["cuda_b3_seconds"] = round(time.time() - t, 1)
    summary["cuda_batch_size_independent"] = (out / "cuda/data.csv").read_bytes() == (out / "cuda_b3/data.csv").read_bytes()

    if args.bench:
        t = time.time()
        b = generate(out / "bench", args.bench, args.seed + 1, 1, DOMAIN, PHYSICS, SAMPLING,
                     progress=False, backend="cuda", gpu=GPU)
        secs = time.time() - t
        summary["bench"] = {"cases": args.bench, "ok": int(b["ok"].sum()), "seconds": round(secs, 1),
                            "seconds_per_case": round(secs / args.bench, 3),
                            "errors": b.loc[~b["ok"], "error"].value_counts().to_dict()}

    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
