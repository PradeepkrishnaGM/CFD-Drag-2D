"""Repository paths, resolved independently of the current working directory."""
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
BUILD_DIR = REPO_ROOT / "build"
SOLVER_EXE = BUILD_DIR / ("solver.exe" if sys.platform == "win32" else "solver")
DEFAULT_DATASET_DIR = REPO_ROOT / "dataset"
