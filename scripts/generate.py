"""Generate a CFD dataset: seeded cases, solved in parallel, masks + targets.

Example:
    python scripts/generate.py --n 100 --seed 0 --workers 12 --out dataset

Re-running the same command resumes an interrupted run. All physics and sampling
defaults live in cfd_drag/params.py.
"""
import argparse
import os
import sys
import time
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from cfd_drag.dataset import generate  # noqa: E402
from cfd_drag.params import DOMAIN, FIELD, GPU, PHYSICS, SAMPLING  # noqa: E402
from cfd_drag.paths import DEFAULT_DATASET_DIR  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, required=True, help="number of cases (indices 0..n-1)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) // 2),
                    help="parallel solver runs (default: physical cores; hyper-threads do not help)")
    ap.add_argument("--out", type=Path, default=DEFAULT_DATASET_DIR)
    ap.add_argument("--shapes", nargs="+", choices=["circle", "rect", "tri"], default=list(SAMPLING.shapes))
    ap.add_argument("--images", action="store_true", help="also save velocity-magnitude PNGs")
    ap.add_argument("--fields", action="store_true",
                    help="also store time-averaged flow fields (fields.npy, ~150 kB per case)")
    ap.add_argument("--backend", choices=["cpu", "cuda"], default="cpu")
    ap.add_argument("--gpus", type=int, default=None, help="cuda: GPUs to use (default: all visible)")
    ap.add_argument("--batch-size", type=int, default=GPU.batch_size, help="cuda: cases per launch")
    args = ap.parse_args()

    sampling = replace(SAMPLING, shapes=tuple(args.shapes))
    t0 = time.time()
    gpu = replace(GPU, batch_size=args.batch_size)
    df = generate(args.out, args.n, args.seed, args.workers, DOMAIN, PHYSICS, sampling, args.images,
                  backend=args.backend, gpu=gpu, gpus=args.gpus, fields=FIELD if args.fields else None)
    ok = int(df["ok"].sum())
    print(f"{ok}/{len(df)} cases ok in {time.time() - t0:.0f} s -> {args.out}")
    if ok < len(df):
        print(df.loc[~df["ok"], "error"].value_counts().to_string())


if __name__ == "__main__":
    main()
