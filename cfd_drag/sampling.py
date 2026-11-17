"""Seeded sampling of simulation cases that satisfy the solver guards by construction.

Each case depends only on (seed, index), so a dataset is identical whatever the number of
workers or the order in which cases run.
"""
import math
from dataclasses import dataclass

import numpy as np

from .geometry import extents
from .params import DomainParams, PhysicsParams, SamplingParams

EPS = 1e-9  # keep strictly inside the solver's guards despite float rounding


@dataclass(frozen=True)
class Case:
    index: int
    shape: str
    size1: float
    size2: float
    angle: float
    re: float
    D: float


def case_rng(seed: int, index: int) -> np.random.Generator:
    return np.random.default_rng([seed, index])


def sample_case(seed: int, index: int, domain: DomainParams, physics: PhysicsParams,
                sampling: SamplingParams) -> Case:
    """Draw one valid case.

    (Re, D) are drawn uniformly over the feasible region: Re in [re_min, re_max],
    D in [d_min, max_blockage * H], and tau(Re, D) >= min_tau. The shape's aspect ratio
    and angle are drawn next, and its sizes are scaled to hit D exactly. Draws that
    violate the size constraints are rejected and redrawn from the same stream.
    """
    rng = case_rng(seed, index)
    d_max = physics.max_blockage * domain.H - EPS
    fit = sampling.crop - sampling.crop_margin
    for _ in range(10_000):
        shape = sampling.shapes[rng.integers(len(sampling.shapes))]
        re = rng.uniform(sampling.re_min, sampling.re_max)
        D = rng.uniform(sampling.d_min, d_max)
        aspect = math.exp(rng.uniform(-1.0, 1.0) * math.log(sampling.aspect_max))
        angle = rng.uniform(sampling.angle_min, sampling.angle_max)
        if physics.tau(re, D) < physics.min_tau + EPS:
            continue
        if shape == "circle":
            size1 = size2 = D / 2
            angle = 0.0
        else:
            # rect: width/height = aspect; tri: length/base = aspect
            s1, s2 = (aspect, 1.0) if shape == "rect" else (1.0, aspect)
            scale = D / extents(shape, s1, s2, angle)[1]
            size1, size2 = s1 * scale, s2 * scale
            if min(size1, size2) < sampling.min_thickness:
                continue
        if max(extents(shape, size1, size2, angle)) > fit:
            continue
        return Case(index, shape, size1, size2, angle, re, D)
    raise RuntimeError(f"no valid case found for seed={seed}, index={index}")
