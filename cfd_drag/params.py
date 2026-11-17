"""All tunable numbers in one place. Change defaults only with the owner's approval.

Values in M0 are copied unchanged from the original scripts; M1 revises the physics.
"""
from dataclasses import dataclass


@dataclass(frozen=True)
class DomainParams:
    nx: int = 400            # lattice cells along the flow
    ny: int = 100            # lattice cells across the channel
    steps: int = 15000       # max LBM steps per run (solver stops early on convergence)
    timeout_s: int = 60      # wall-clock limit per solver run


@dataclass(frozen=True)
class SamplingParams:
    min_size_frac: float = 0.05   # min obstacle size as a fraction of ny
    max_size_frac: float = 0.35   # max obstacle size as a fraction of ny
    angle_min: float = 0.0        # angle of attack range [deg]
    angle_max: float = 180.0
    velocity_min: float = 0.02    # inlet velocity range [lattice units]
    velocity_max: float = 0.1
    max_abs_cd: float = 100.0     # samples with |Cd| above this are discarded as blow-ups


DOMAIN = DomainParams()
SAMPLING = SamplingParams()
