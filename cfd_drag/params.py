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

    @property
    def H(self) -> float:
        return self.ny - 2.0

    @property
    def centre(self) -> tuple[int, int]:
        """Obstacle centre (x, y) in lattice nodes, as the solver places it by default."""
        return self.nx // 4, self.ny // 2


@dataclass(frozen=True)
class PhysicsParams:
    u: float = 0.05                   # mean inlet velocity [lattice units]; Mach ~ 0.087
    min_tau: float = 0.52             # runs with a lower relaxation time are rejected (stability)
    max_blockage: float = 0.25        # max frontal height D / channel width H
    warmup_flowthroughs: float = 2.0  # steps before averaging = this * nx / u
    avg_flowthroughs: float = 1.0     # averaging window = this * nx / u
    steady_cl_rms: float = 0.02       # Cl_rms below this -> regime "steady", else "periodic"

    def warmup_steps(self, nx: int) -> int:
        return int(self.warmup_flowthroughs * nx / self.u)

    def avg_steps(self, nx: int) -> int:
        return int(self.avg_flowthroughs * nx / self.u)

    def tau(self, re: float, D: float) -> float:
        return 3.0 * self.u * D / re + 0.5


@dataclass(frozen=True)
class SamplingParams:
    shapes: tuple[str, ...] = ("circle", "rect", "tri")
    re_min: float = 20.0          # Reynolds number range, Re = U * D / nu (option (a): max 100)
    re_max: float = 100.0
    d_min: float = 5.0            # min frontal height D [cells] (max is set by the blockage cap)
    aspect_max: float = 3.0       # rect/tri side ratio drawn log-uniformly in [1/aspect_max, aspect_max]
    min_thickness: float = 3.0    # min side length [cells] so thin shapes still have solid nodes
    angle_min: float = 0.0        # angle of attack range [deg]; circles always use 0
    angle_max: float = 180.0
    crop: int = 64                # mask crop size [cells] around the obstacle centre (CNN input)
    crop_margin: float = 4.0      # obstacle bounding box must fit in crop - margin
    max_abs_cd: float = 100.0     # samples with |Cd| above this are discarded as blow-ups


@dataclass(frozen=True)
class GpuParams:
    batch_size: int = 256         # cases per solver_cuda launch (~7.4 MB GPU memory per case)
    timeout_s: int = 7200         # wall-clock limit per batch


DOMAIN = DomainParams()
PHYSICS = PhysicsParams()
SAMPLING = SamplingParams()
GPU = GpuParams()
