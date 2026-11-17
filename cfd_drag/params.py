"""All tunable numbers in one place. Change defaults only with the owner's approval.

The solver's built-in defaults (cfd_engine/solver.cpp, struct Config) mirror these values,
but the Python scripts always pass them explicitly, so this file is authoritative.
"""
from dataclasses import dataclass


@dataclass(frozen=True)
class DomainParams:
    nx: int = 400            # lattice cells along the flow
    ny: int = 100            # lattice cells across the channel (fluid width H = ny - 2)
    timeout_s: int = 300     # wall-clock limit per solver run


@dataclass(frozen=True)
class PhysicsParams:
    u: float = 0.05                   # mean inlet velocity [lattice units]; Mach ~ 0.087
    min_tau: float = 0.52             # runs with a lower relaxation time are rejected (stability)
    max_blockage: float = 0.25        # max frontal height D / channel width H
    warmup_flowthroughs: float = 2.0  # steps before averaging = this * nx / u
    avg_flowthroughs: float = 1.0     # averaging window = this * nx / u

    def warmup_steps(self, nx: int) -> int:
        return int(self.warmup_flowthroughs * nx / self.u)

    def avg_steps(self, nx: int) -> int:
        return int(self.avg_flowthroughs * nx / self.u)


@dataclass(frozen=True)
class SamplingParams:
    re_min: float = 20.0          # Reynolds number range, Re = U * D / nu
    re_max: float = 150.0
    min_size_frac: float = 0.05   # min obstacle size as a fraction of ny
    max_size_frac: float = 0.35   # max obstacle size as a fraction of ny (blockage cap still applies)
    angle_min: float = 0.0        # angle of attack range [deg]
    angle_max: float = 180.0
    max_abs_cd: float = 100.0     # samples with |Cd| above this are discarded as blow-ups


DOMAIN = DomainParams()
PHYSICS = PhysicsParams()
SAMPLING = SamplingParams()
