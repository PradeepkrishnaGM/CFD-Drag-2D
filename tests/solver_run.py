"""Helper to run the solver binary in a temp dir and read its outputs."""
import csv
import subprocess

import numpy as np


def run_solver(exe, out_dir, *args, check=True, timeout=600):
    cmd = [str(exe), "--out-dir", str(out_dir), *map(str, args)]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    if check:
        assert proc.returncode == 0, proc.stderr + proc.stdout
    return proc


def read_results(out_dir):
    with open(out_dir / "results.csv") as f:
        row = next(csv.DictReader(f))
    return {k: (v if k in ("shape_type", "status") else float(v)) for k, v in row.items()}


def load(out_dir, name):
    return np.load(out_dir / f"{name}.npy")
