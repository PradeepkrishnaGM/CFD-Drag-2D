"""Quick checks that the build and the Python wrappers are wired up correctly."""
import subprocess
import sys

from cfd_drag.paths import REPO_ROOT
from solver_run import read_results, run_solver

TINY = ["--width", "60", "--height", "30", "--shape", "circle", "--size1", "3",
        "--re", "10", "--warmup", "150", "--avg", "50"]


def test_tiny_run_writes_outputs(solver_exe, tmp_path):
    run_solver(solver_exe, tmp_path, *TINY)
    r = read_results(tmp_path)
    assert r["shape_type"] == "circle" and r["status"] in ("steady", "periodic")
    ppm = (tmp_path / "flow_field.ppm").read_bytes()
    header = b"P6\n60 30\n255\n"
    assert ppm.startswith(header) and len(ppm) == len(header) + 60 * 30 * 3


def test_auto_cli_help():
    out = subprocess.run([sys.executable, "scripts/generate.py", "--help"], cwd=REPO_ROOT,
                         capture_output=True, text=True, check=True)
    assert "--out" in out.stdout
