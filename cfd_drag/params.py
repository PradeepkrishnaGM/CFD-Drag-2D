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
    walls: str = "noslip"             # channel walls: "noslip" (bounce-back) or "freeslip" (specular)
    precision: str = "double"         # numerics, not physics: solver population storage "double" or
                                      # "float" (stores f_i - w_i; see DECISIONS.md, M10)

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
class FieldParams:
    """Time-averaged flow-field window stored with a field dataset (v2, M7).

    The window spans x in [cx - x_before, cx - x_before + width) and
    y in [cy - height/2, cy + height/2) around the obstacle centre (cx, cy); both sizes are
    divisible by 32 for the U-Net. Stored channels: ux / U, uy / U and the pressure coefficient
    Cp = (p - p_out) / (0.5 U^2) with p = rho / 3 and p_out = 1/3 (outlet), as float16.
    """
    width: int = 256          # cells along the flow (near wake up to 7 obstacle sizes downstream)
    height: int = 96          # cells across (channel rows 2..97 for ny = 100; wall rows dropped)
    x_before: int = 32        # cells upstream of the obstacle centre (= half the mask crop)


@dataclass(frozen=True)
class GpuParams:
    batch_size: int = 256         # cases per solver_cuda launch (~7.4 MB GPU memory per case)
    timeout_s: int = 7200         # wall-clock limit per batch


@dataclass(frozen=True)
class CheckParams:
    min_ok_fraction: float = 0.95  # dataset check fails if fewer cases than this solved cleanly


@dataclass(frozen=True)
class TrainParams:
    split_seed: int = 0                     # train/val/test split by sample
    val_fraction: float = 0.15
    test_fraction: float = 0.15
    channels: tuple[int, ...] = (32, 64, 128, 128)  # conv blocks (each halves the image)
    hidden: int = 128                       # dense head width
    loss_weight_cd: float = 1.0             # MSE weights on standardised targets
    loss_weight_cl: float = 0.5
    lr: float = 1e-3                        # AdamW, cosine schedule to 0
    weight_decay: float = 1e-4
    batch_size: int = 128
    epochs: int = 150                       # best epoch by validation Cd MAE is kept
    seed: int = 0                           # weight init and batch order
    augment_flip: bool = False              # M10: random mirror about the channel centreline
                                            # (exact symmetry: Cd kept, Cl -> -Cl), p = 0.5 per sample
    re_weight_bins: int = 0                 # M10: >0 weights training samples by inverse frequency
                                            # of their Re bin (equal-width bins), mean weight 1
    re_weight_max: float = 5.0              # cap on a sample weight before renormalising
    gbm_max_iter: int = 500                 # gradient-boosting baseline
    gbm_learning_rate: float = 0.05


@dataclass(frozen=True)
class FieldTrainParams:
    """U-Net flow-field surrogate (v2, M8). The split is the same as TrainParams (same test rows)."""
    base_channels: int = 32                 # U-Net width at full resolution, doubled per level
    levels: int = 4                         # down-sampling steps (96 x 256 -> 6 x 16)
    sdf_scale: float = 32.0                 # signed distance input is divided by this [cells]
    loss_weights: tuple[float, ...] = (1.0, 1.0, 1.0)  # MSE weights for ux, uy, Cp (standardised)
    lr: float = 1e-3                        # AdamW, cosine schedule to 0
    weight_decay: float = 1e-4
    batch_size: int = 32
    epochs: int = 100                       # best epoch by validation error is kept
    seed: int = 0                           # weight init and batch order
    amp: bool = True                        # float16 mixed precision on CUDA (ignored on CPU)
    augment_flip: bool = False              # M10: random mirror about the channel centreline (uy -> -uy)


DOMAIN = DomainParams()
PHYSICS = PhysicsParams()
SAMPLING = SamplingParams()
GPU = GpuParams()
FIELD = FieldParams()
CHECK = CheckParams()
TRAIN = TrainParams()
FIELD_TRAIN = FieldTrainParams()
