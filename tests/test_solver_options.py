"""M10 solver options: free-slip channel walls and float population storage."""
import numpy as np
import pytest

from solver_run import load, read_results, run_solver

BOX = ["--box", "--width", "60", "--height", "30", "--shape", "tri", "--size1", "6", "--size2", "8",
       "--angle", "30", "--re", "20", "--warmup", "400", "--avg", "100"]
CYL = ["--width", "120", "--height", "40", "--shape", "circle", "--size1", "4", "--re", "30",
       "--warmup", "4000", "--avg", "1000", "--dump", "--mean-fields"]


@pytest.mark.parametrize("precision,tol", [("double", 1e-12), ("float", 1e-6)])
def test_freeslip_conserves_mass_in_closed_box(solver_exe, tmp_path, precision, tol):
    run_solver(solver_exe, tmp_path, *BOX, "--walls", "freeslip", "--precision", precision)
    assert abs(read_results(tmp_path)["mass_rel_change"]) < tol


def test_freeslip_empty_channel_stays_uniform(solver_exe, tmp_path):
    """No wall friction: a uniform inlet gives uniform flow (start-up sound waves decay slowly)."""
    run_solver(solver_exe, tmp_path, "--width", "60", "--height", "12", "--shape", "none", "--re", "30",
               "--walls", "freeslip", "--warmup", "40000", "--avg", "1", "--dump")
    ux, uy, _ = load(tmp_path, "fields")
    assert np.abs(ux[1:-1] / 0.05 - 1).max() < 1e-3
    assert np.abs(uy[1:-1]).max() < 1e-12


def test_freeslip_lowers_cylinder_drag(solver_exe, tmp_path):
    """Same cylinder: no-slip walls add blockage-induced drag, so free-slip Cd is lower, Cl ~ 0."""
    cd = {}
    for walls in ("noslip", "freeslip"):
        d = tmp_path / walls
        d.mkdir()
        run_solver(solver_exe, d, *CYL, "--walls", walls)
        cd[walls] = read_results(d)["Cd"]
    assert cd["freeslip"] < cd["noslip"]


@pytest.mark.parametrize("walls", ["noslip", "freeslip"])
def test_float_matches_double(solver_exe, tmp_path, walls):
    runs = {}
    for prec in ("double", "float"):
        d = tmp_path / prec
        d.mkdir()
        run_solver(solver_exe, d, *CYL, "--walls", walls, "--precision", prec)
        runs[prec] = d
    a, b = read_results(runs["double"]), read_results(runs["float"])
    assert a["status"] == b["status"]
    assert b["Cd"] == pytest.approx(a["Cd"], rel=1e-3)
    assert abs(b["Cl"] - a["Cl"]) < 1e-3
    ma, mb = load(runs["double"], "fields_mean"), load(runs["float"], "fields_mean")
    assert np.abs(ma[:2] - mb[:2]).max() < 1e-3 * 0.05  # velocity, relative to U
    assert np.abs(ma[2] - mb[2]).max() < 1e-5            # density


def test_float_deterministic(solver_exe, tmp_path):
    for name in ("a", "b"):
        (tmp_path / name).mkdir()
        run_solver(solver_exe, tmp_path / name, *CYL, "--warmup", "500", "--avg", "100", "--precision", "float")
    assert (tmp_path / "a/results.csv").read_bytes() == (tmp_path / "b/results.csv").read_bytes()
    assert (tmp_path / "a/fields.npy").read_bytes() == (tmp_path / "b/fields.npy").read_bytes()


@pytest.mark.parametrize("args", [["--walls", "slippery"], ["--precision", "half"]])
def test_invalid_option_rejected(solver_exe, tmp_path, args):
    proc = run_solver(solver_exe, tmp_path, "--shape", "circle", "--size1", "5", "--re", "20", *args, check=False)
    assert proc.returncode == 2


@pytest.mark.slow
def test_schafer_turek_cylinder_float(solver_exe, tmp_path):
    """2D-1 benchmark with float storage: same tolerance as the double test."""
    run_solver(solver_exe, tmp_path, "--width", "440", "--height", "84", "--shape", "circle",
               "--size1", "10", "--cx", "40", "--cy", "40.5", "--inlet", "parabolic",
               "--re", "20", "--u", "0.05", "--warmup", "20000", "--avg", "2000", "--precision", "float")
    r = read_results(tmp_path)
    assert r["status"] == "steady"
    assert r["Cd"] == pytest.approx(5.5795, rel=0.05)
    assert abs(r["Cl"]) < 0.05
