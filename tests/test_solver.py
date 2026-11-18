"""Invariants and validation of the LBM solver."""
import math

import numpy as np
import pytest

from solver_run import load, read_results, run_solver


def test_deterministic(solver_exe, tmp_path):
    args = ["--width", "80", "--height", "30", "--shape", "rect", "--size1", "6", "--size2", "5",
            "--angle", "20", "--re", "30", "--warmup", "300", "--avg", "100", "--dump"]
    a, b = tmp_path / "a", tmp_path / "b"
    a.mkdir(), b.mkdir()
    run_solver(solver_exe, a, *args)
    run_solver(solver_exe, b, *args)
    assert (a / "results.csv").read_bytes() == (b / "results.csv").read_bytes()
    assert (a / "fields.npy").read_bytes() == (b / "fields.npy").read_bytes()


def test_mass_conserved_in_closed_box(solver_exe, tmp_path):
    run_solver(solver_exe, tmp_path, "--box", "--width", "60", "--height", "30", "--shape", "tri",
               "--size1", "6", "--size2", "8", "--angle", "30", "--re", "20",
               "--warmup", "400", "--avg", "100")
    assert abs(read_results(tmp_path)["mass_rel_change"]) < 1e-12


@pytest.mark.parametrize("shape,s1,s2,angle,expected", [
    ("circle", 7, 7, 0, 14.0),
    ("rect", 10, 20, 0, 20.0),
    ("rect", 10, 20, 90, 10.0),
    ("rect", 10, 20, 30, 10 * math.sin(math.radians(30)) + 20 * math.cos(math.radians(30))),
    ("tri", 12, 20, 0, 12.0),
    ("tri", 12, 20, 90, 20.0),
])
def test_frontal_height(solver_exe, tmp_path, shape, s1, s2, angle, expected):
    run_solver(solver_exe, tmp_path, "--width", "200", "--height", "100", "--shape", shape,
               "--size1", s1, "--size2", s2, "--angle", angle, "--re", "20",
               "--warmup", "1", "--avg", "1")
    assert read_results(tmp_path)["D"] == pytest.approx(expected, rel=1e-9)


def test_solid_nodes_inset_half_cell(solver_exe, tmp_path):
    # A 20x20 square centred on a node: solid nodes span 19x19 (wall half a cell outside).
    run_solver(solver_exe, tmp_path, "--width", "200", "--height", "100", "--shape", "rect",
               "--size1", "20", "--size2", "20", "--re", "20", "--warmup", "1", "--avg", "1", "--dump")
    m = load(tmp_path, "mask")
    assert (m == 1).sum() == 19 * 19
    assert (m[0] == 2).all() and (m[-1] == 2).all()


@pytest.mark.parametrize("args", [
    ["--re", "1000"],                       # tau below min_tau
    ["--size1", "20"],                      # blockage 40/98 > 0.25
    ["--bogus", "1"],                       # unknown argument
    ["--shape", "hexagon"],
])
def test_invalid_config_rejected(solver_exe, tmp_path, args):
    base = ["--width", "100", "--height", "100", "--shape", "circle", "--size1", "5", "--re", "20"]
    proc = run_solver(solver_exe, tmp_path, *base, *args, check=False)
    assert proc.returncode == 2
    assert not (tmp_path / "results.csv").exists()


def test_poiseuille_channel(solver_exe, tmp_path):
    """Empty channel with parabolic inlet stays a parabola with the analytic pressure gradient.

    LBM is weakly compressible: density falls along the channel, so we compare the profile
    shape (u / mean u) and the gradient for the actual mass flux rho * u_mean.
    """
    H, U, re = 20, 0.05, 10.0
    run_solver(solver_exe, tmp_path, "--width", "100", "--height", H + 2, "--shape", "none",
               "--inlet", "parabolic", "--re", re, "--u", U, "--warmup", "6000", "--avg", "10",
               "--dump")
    ux, _, rho = load(tmp_path, "fields")
    s = (np.arange(1, H + 1) - 0.5) / H
    shape_exact = 6 * s * (1 - s)
    u = ux[1:H + 1, 50]
    assert np.abs(u / u.mean() - shape_exact).max() < 0.005 * 1.5

    nu = U * H / re
    flux = (rho[1:H + 1, 50] * u).mean()
    dpdx = -(rho[1:H + 1, 75].mean() - rho[1:H + 1, 25].mean()) / 3 / 50
    assert dpdx == pytest.approx(12 * nu * flux / H**2, rel=0.02)


@pytest.mark.slow
def test_schafer_turek_cylinder(solver_exe, tmp_path):
    """2D-1 benchmark (Schäfer & Turek 1996): Re = 20, Cd = 5.5795, D = 20 cells."""
    run_solver(solver_exe, tmp_path, "--width", "440", "--height", "84", "--shape", "circle",
               "--size1", "10", "--cx", "40", "--cy", "40.5", "--inlet", "parabolic",
               "--re", "20", "--u", "0.05", "--warmup", "20000", "--avg", "2000")
    r = read_results(tmp_path)
    assert r["status"] == "steady"
    assert r["Cd"] == pytest.approx(5.5795, rel=0.05)
    assert abs(r["Cl"]) < 0.05


def test_mean_fields_are_time_averages(solver_exe, tmp_path):
    """fields_mean over a 2-step window = average of the snapshots at those two steps."""
    common = ["--width", "80", "--height", "30", "--shape", "circle", "--size1", "3", "--re", "40",
              "--ramp", "50", "--dump", "--mean-fields"]
    runs = {}
    for name, warmup, avg in [("w1", 200, 1), ("w2", 201, 1), ("pair", 200, 2)]:
        d = tmp_path / name
        d.mkdir()
        run_solver(solver_exe, d, *common, "--warmup", warmup, "--avg", avg)
        runs[name] = d
    # one-step window: the mean is exactly the snapshot
    assert np.array_equal(load(runs["w1"], "fields_mean"), load(runs["w1"], "fields"))
    expected = 0.5 * (load(runs["w1"], "fields") + load(runs["w2"], "fields"))
    assert np.allclose(load(runs["pair"], "fields_mean"), expected, rtol=0, atol=1e-14)
    mean, mask = load(runs["pair"], "fields_mean"), load(runs["pair"], "mask")
    assert mean.shape == (3, 30, 80)
    assert np.all(mean[:2][:, mask != 0] == 0) and np.all(mean[2][mask != 0] == 1)
