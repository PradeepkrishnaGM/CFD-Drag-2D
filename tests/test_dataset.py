"""Sampler invariants, geometry consistency with the solver, and generation determinism."""
import math
from dataclasses import replace

import numpy as np
import pytest

from cfd_drag.dataset import crop_mask, generate, load
from cfd_drag.geometry import extents
from cfd_drag.params import DOMAIN, PHYSICS, SAMPLING, DomainParams, PhysicsParams
from cfd_drag.sampling import sample_case
from solver_run import load as load_npy
from solver_run import read_results, run_solver

# Tiny setup for fast generation tests: 120x40 domain, ~0.2 flow-throughs.
TINY_DOMAIN = DomainParams(nx=120, ny=40, timeout_s=60)
TINY_PHYSICS = replace(PHYSICS, warmup_flowthroughs=0.1, avg_flowthroughs=0.05)
TINY_SAMPLING = replace(SAMPLING, crop=32, d_min=4.0, min_thickness=2.0)


def test_sampler_respects_all_constraints():
    fit = SAMPLING.crop - SAMPLING.crop_margin
    shapes = set()
    for i in range(3000):
        c = sample_case(0, i, DOMAIN, PHYSICS, SAMPLING)
        shapes.add(c.shape)
        assert SAMPLING.re_min <= c.re <= SAMPLING.re_max
        assert SAMPLING.d_min <= c.D <= PHYSICS.max_blockage * DOMAIN.H
        assert PHYSICS.tau(c.re, c.D) >= PHYSICS.min_tau
        assert math.isclose(extents(c.shape, c.size1, c.size2, c.angle)[1], c.D, rel_tol=1e-12)
        assert max(extents(c.shape, c.size1, c.size2, c.angle)) <= fit
        if c.shape != "circle":
            assert min(c.size1, c.size2) >= SAMPLING.min_thickness
    assert shapes == {"circle", "rect", "tri"}


def test_sampler_is_deterministic_per_index():
    a = [sample_case(7, i, DOMAIN, PHYSICS, SAMPLING) for i in range(50)]
    b = [sample_case(7, i, DOMAIN, PHYSICS, SAMPLING) for i in reversed(range(50))][::-1]
    assert a == b
    assert a != [sample_case(8, i, DOMAIN, PHYSICS, SAMPLING) for i in range(50)]


@pytest.mark.parametrize("index", range(6))
def test_python_geometry_matches_solver(solver_exe, tmp_path, index):
    c = sample_case(3, index, DOMAIN, PHYSICS, SAMPLING)
    run_solver(solver_exe, tmp_path, "--width", DOMAIN.nx, "--height", DOMAIN.ny, "--shape", c.shape,
               "--size1", repr(c.size1), "--size2", repr(c.size2), "--angle", repr(c.angle),
               "--re", repr(c.re), "--warmup", 1, "--avg", 1, "--dump")
    r = read_results(tmp_path)
    assert r["D"] == pytest.approx(c.D, rel=1e-9)
    full = load_npy(tmp_path, "mask")
    crop = crop_mask(full, DOMAIN, SAMPLING.crop)
    assert crop.sum() == (full == 1).sum() > 0  # the whole obstacle lies inside the crop


def test_generation_is_independent_of_worker_count(solver_exe, tmp_path):
    args = (8, 1, TINY_DOMAIN, TINY_PHYSICS, TINY_SAMPLING)
    df1 = generate(tmp_path / "w1", args[0], args[1], 1, *args[2:], progress=False)
    df3 = generate(tmp_path / "w3", args[0], args[1], 3, *args[2:], progress=False)
    assert (tmp_path / "w1/data.csv").read_bytes() == (tmp_path / "w3/data.csv").read_bytes()
    assert np.array_equal(np.load(tmp_path / "w1/masks.npy"), np.load(tmp_path / "w3/masks.npy"))
    assert df1["ok"].all(), df1["error"].tolist()
    assert len(df3) == 8


def test_generation_resumes(solver_exe, tmp_path):
    out = tmp_path / "ds"
    args = (1, TINY_DOMAIN, TINY_PHYSICS, TINY_SAMPLING)
    generate(out, 3, 2, *args, progress=False)
    before = (out / "samples/000001.npz").stat().st_mtime_ns
    df = generate(out, 5, 2, *args, progress=False)  # adds cases 3 and 4 only
    assert (out / "samples/000001.npz").stat().st_mtime_ns == before
    assert list(df["index"]) == [0, 1, 2, 3, 4]
    with pytest.raises(ValueError):
        generate(out, 5, 99, *args, progress=False)  # different seed -> refuse to mix


def test_load_returns_aligned_ok_rows(solver_exe, tmp_path):
    out = tmp_path / "ds"
    generate(out, 4, 5, 2, TINY_DOMAIN, TINY_PHYSICS, TINY_SAMPLING, images=True, progress=False)
    df, masks = load(out)
    assert len(df) == len(masks) == 4
    assert masks.shape[1:] == (32, 32) and masks.dtype == np.uint8
    assert all(m.sum() > 0 for m in masks)
    assert len(list((out / "images").glob("*.png"))) == 4
