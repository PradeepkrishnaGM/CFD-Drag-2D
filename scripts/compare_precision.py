"""M10: float vs double population storage -- accuracy and speed.

Runs the same sampled cases (seed 0 = the first rows of dataset v1) with double and float
storage on one back-end, compares Cd / Cl / Cl_rms / status, times both, and runs the
Schafer-Turek 2D-1 cylinder (Cd = 5.5795) in both precisions. Writes <out>/summary.json.

    python scripts/compare_precision.py --backend cuda --n 128 --bench 512 --out /kaggle/working/m10
    python scripts/compare_precision.py --backend cpu --n 4 --bench 0 --tiny --out /tmp/x   # smoke test
"""
import argparse
import json
import os
import subprocess
import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from cfd_drag.dataset import generate  # noqa: E402
from cfd_drag.params import DOMAIN, GPU, PHYSICS, SAMPLING, DomainParams  # noqa: E402
from cfd_drag.paths import SOLVER_CUDA_EXE, SOLVER_EXE  # noqa: E402

ST_ARGS = ["--width", "440", "--height", "84", "--inlet", "parabolic", "--re", "20", "--u", "0.05",
           "--cx", "40", "--cy", "40.5", "--warmup", "20000", "--avg", "2000"]
ST_CD = 5.5795


def schafer_turek(backend: str, precision: str, out: Path) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    case = out / "case"
    case.mkdir(exist_ok=True)
    if backend == "cuda":
        (out / "cases.csv").write_text(f"out_dir,shape,size1,size2,angle,re\n{case},circle,10,10,0,20\n")
        cmd = [str(SOLVER_CUDA_EXE), "--batch", str(out / "cases.csv"), *ST_ARGS]
    else:
        cmd = [str(SOLVER_EXE), "--out-dir", str(case), "--shape", "circle", "--size1", "10", *ST_ARGS]
    t = time.time()
    subprocess.run([*cmd, "--precision", precision], check=True, capture_output=True, text=True)
    import pandas as pd
    r = pd.read_csv(case / "results.csv").iloc[0]
    return {"Cd": float(r["Cd"]), "Cl": float(r["Cl"]), "Cd_rel_err_vs_reference": float(r["Cd"] / ST_CD - 1),
            "status": r["status"], "seconds": round(time.time() - t, 1)}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--backend", choices=["cpu", "cuda"], default="cuda")
    ap.add_argument("--n", type=int, default=128, help="cases compared (seed 0, indices 0..n-1)")
    ap.add_argument("--bench", type=int, default=512, help="cases per precision for the timing run (0: skip)")
    ap.add_argument("--workers", type=int, default=os.cpu_count())
    ap.add_argument("--tiny", action="store_true", help="small domain and short runs (local smoke test)")
    ap.add_argument("--no-st", action="store_true", help="skip the Schafer-Turek cylinder")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    out = args.out
    domain, physics, sampling = DOMAIN, PHYSICS, SAMPLING
    if args.tiny:
        domain = DomainParams(nx=120, ny=40, timeout_s=120)
        physics = replace(PHYSICS, warmup_flowthroughs=1.0, avg_flowthroughs=0.5)
        sampling = replace(SAMPLING, crop=32, d_min=4.0, min_thickness=2.0)
    kw = dict(domain=domain, sampling=sampling, progress=False, backend=args.backend, gpu=GPU)
    summary = {"backend": args.backend, "n": args.n, "tiny": args.tiny}

    runs, secs = {}, {}
    for prec in ("double", "float"):
        t = time.time()
        runs[prec] = generate(out / f"cases_{prec}", args.n, 0, args.workers,
                              physics=replace(physics, precision=prec), **kw)
        secs[prec] = round(time.time() - t, 1)
    d, f = runs["double"], runs["float"]
    ok = d["ok"].to_numpy(bool) & f["ok"].to_numpy(bool)
    summary["ok"] = {"double": int(d["ok"].sum()), "float": int(f["ok"].sum())}
    summary["status_agree"] = int((d["status"] == f["status"]).sum())
    summary["compare_seconds"] = secs
    for regime in ("steady", "periodic"):
        sel = ok & (d["status"] == regime).to_numpy()
        st = {"count": int(sel.sum())}
        if sel.any():
            rel = np.abs(f.loc[sel, "Cd"].to_numpy() / d.loc[sel, "Cd"].to_numpy() - 1)
            st["Cd_rel_diff_median"] = float(np.median(rel))
            st["Cd_rel_diff_max"] = float(rel.max())
            st["Cl_abs_diff_max"] = float(np.abs(f.loc[sel, "Cl"] - d.loc[sel, "Cl"]).max())
            st["Cl_rms_abs_diff_max"] = float(np.abs(f.loc[sel, "Cl_rms"] - d.loc[sel, "Cl_rms"]).max())
        summary[regime] = st

    if args.bench:
        bench = {}
        for prec in ("double", "float"):
            t = time.time()
            b = generate(out / f"bench_{prec}", args.bench, 1, args.workers,
                         physics=replace(physics, precision=prec), **kw)
            s = time.time() - t
            bench[prec] = {"cases": args.bench, "ok": int(b["ok"].sum()), "seconds": round(s, 1),
                           "seconds_per_case": round(s / args.bench, 3)}
        bench["speedup"] = round(bench["double"]["seconds"] / bench["float"]["seconds"], 2)
        summary["bench"] = bench

    if not args.no_st:
        summary["schafer_turek"] = {p: schafer_turek(args.backend, p, out / f"st_{p}") for p in ("double", "float")}

    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
