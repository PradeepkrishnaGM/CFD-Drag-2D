"""Quick checks that the build and the Python wrappers are wired up correctly."""
import csv
import subprocess
import sys

from cfd_drag.paths import REPO_ROOT


def run_tiny(solver_exe, cwd):
    cmd = [str(solver_exe), "--width", "60", "--height", "30", "--shape", "circle",
           "--size1", "4", "--velocity", "0.05", "--steps", "200"]
    return subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, timeout=30, check=True)


def test_tiny_run_writes_outputs(solver_exe, tmp_path):
    run_tiny(solver_exe, tmp_path)
    with open(tmp_path / "results.csv") as f:
        row = next(csv.DictReader(f))
    assert row["shape_type"] == "circle"
    float(row["Cd"]), float(row["pressure_loss"])  # parseable numbers

    ppm = (tmp_path / "flow_field.ppm").read_bytes()
    assert ppm.startswith(b"P6\n60 30\n255\n")
    assert len(ppm) == len(b"P6\n60 30\n255\n") + 60 * 30 * 3


def test_auto_cli_help():
    out = subprocess.run([sys.executable, "scripts/auto.py", "--help"], cwd=REPO_ROOT,
                         capture_output=True, text=True, check=True)
    assert "--out" in out.stdout
