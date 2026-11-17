"""CUDA solver checks. Skipped unless build/solver_cuda exists and a GPU is visible."""
import shutil
import subprocess
from dataclasses import replace

import numpy as np
import pytest

from cfd_drag.dataset import generate
from cfd_drag.params import PHYSICS, SAMPLING, DomainParams, GpuParams
from cfd_drag.paths import REPO_ROOT, SOLVER_CUDA_EXE
from solver_run import read_results, run_solver


def _gpu_available() -> bool:
    if not SOLVER_CUDA_EXE.exists() or shutil.which("nvidia-smi") is None:
        return False
    return subprocess.run(["nvidia-smi", "-L"], capture_output=True).returncode == 0


pytestmark = [pytest.mark.gpu, pytest.mark.skipif(not _gpu_available(), reason="no CUDA solver or GPU")]

TINY_DOMAIN = DomainParams(nx=120, ny=40, timeout_s=120)
TINY_PHYSICS = replace(PHYSICS, warmup_flowthroughs=1.0, avg_flowthroughs=0.5)
TINY_SAMPLING = replace(SAMPLING, crop=32, d_min=4.0, min_thickness=2.0)
COMMON = ["--width", "120", "--height", "40", "--warmup", "1500", "--avg", "500", "--dump"]
CASES = [("circle", 4.0, 4.0, 0.0, 30.0), ("rect", 3.0, 6.0, 25.0, 45.0), ("tri", 5.0, 7.0, 60.0, 25.0)]


def _run_batch(tmp_path, cases, name):
    root = tmp_path / name
    root.mkdir()
    lines = ["out_dir,shape,size1,size2,angle,re"]
    for i, (shape, s1, s2, a, re) in enumerate(cases):
        (root / str(i)).mkdir()
        lines.append(f"{root / str(i)},{shape},{s1},{s2},{a},{re}")
    (root / "cases.csv").write_text("\n".join(lines) + "\n")
    proc = subprocess.run([str(SOLVER_CUDA_EXE), "--batch", str(root / "cases.csv"), *COMMON],
                          capture_output=True, text=True, timeout=600)
    assert proc.returncode == 0, proc.stderr + proc.stdout
    return root


def test_cuda_matches_cpu(solver_exe, tmp_path):
    root = _run_batch(tmp_path, CASES, "gpu")
    for i, (shape, s1, s2, a, re) in enumerate(CASES):
        cpu = tmp_path / f"cpu{i}"
        cpu.mkdir()
        run_solver(solver_exe, cpu, *COMMON, "--shape", shape, "--size1", s1, "--size2", s2,
                   "--angle", a, "--re", re)
        g, c = read_results(root / str(i)), read_results(cpu)
        for k in ("Cd", "Cl", "pressure_loss"):
            assert g[k] == pytest.approx(c[k], rel=1e-6, abs=1e-9), (shape, k)
        fg, fc = np.load(root / str(i) / "fields.npy"), np.load(cpu / "fields.npy")
        assert np.abs(fg - fc).max() < 1e-9
        assert np.array_equal(np.load(root / str(i) / "mask.npy"), np.load(cpu / "mask.npy"))


def test_batch_composition_does_not_change_results(tmp_path):
    alone = _run_batch(tmp_path, CASES[1:2], "alone")
    together = _run_batch(tmp_path, CASES, "together")
    assert (alone / "0" / "results.csv").read_bytes() == (together / "1" / "results.csv").read_bytes()


def test_rejected_case_is_reported(tmp_path):
    root = _run_batch(tmp_path, [("circle", 30.0, 30.0, 0.0, 30.0), CASES[0]], "rej")
    assert (root / "0" / "rejected.txt").exists()
    assert (root / "1" / "results.csv").exists()


def test_cuda_dataset_independent_of_batch_size(tmp_path):
    kw = dict(domain=TINY_DOMAIN, physics=TINY_PHYSICS, sampling=TINY_SAMPLING, progress=False,
              backend="cuda", gpus=1)
    generate(tmp_path / "b2", 6, 3, 1, gpu=GpuParams(batch_size=2), **kw)
    generate(tmp_path / "b6", 6, 3, 1, gpu=GpuParams(batch_size=6), **kw)
    assert (tmp_path / "b2/data.csv").read_bytes() == (tmp_path / "b6/data.csv").read_bytes()
