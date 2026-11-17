import subprocess

import pytest

from cfd_drag.paths import REPO_ROOT, SOLVER_EXE


@pytest.fixture(scope="session")
def solver_exe():
    """Build the solver once per test session and return its path."""
    subprocess.run(["make", "-s", "solver"], cwd=REPO_ROOT, check=True)
    assert SOLVER_EXE.exists()
    return SOLVER_EXE
