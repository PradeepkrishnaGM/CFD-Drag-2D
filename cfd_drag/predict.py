"""Single-case helpers for the GUI: validity checks, the exact CNN input mask, and the surrogate.

Kept free of Qt so they can be tested headless. The surrogate needs the optional [ml] extra.
"""
import math
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .dataset import crop_field, crop_mask, global_args, window_mask
from .geometry import extents
from .params import FIELD, DOMAIN, PHYSICS, SAMPLING, DomainParams, PhysicsParams, SamplingParams
from .paths import REPO_ROOT, SOLVER_EXE

MODEL_PATH = REPO_ROOT / "models" / "surrogate_v1.pt"
# U-Net weights (31 MB) are a release asset, not tracked in git: scripts/fetch_field_model.py
FIELD_MODEL_PATH = REPO_ROOT / "models" / "field_unet_v1.pt"


@dataclass
class CaseInfo:
    D: float
    blockage: float
    tau: float
    problems: list[str]   # the solver would reject the case
    warnings: list[str]   # solvable, but outside the range the CNN was trained on

    @property
    def solvable(self) -> bool:
        return not self.problems


def case_info(shape: str, size1: float, size2: float, angle: float, re: float,
              domain: DomainParams = DOMAIN, physics: PhysicsParams = PHYSICS,
              sampling: SamplingParams = SAMPLING) -> CaseInfo:
    if shape == "circle":
        size2, angle = size1, 0.0
    dx, D = extents(shape, size1, size2, angle)
    blockage = D / domain.H
    tau = physics.tau(re, D)
    problems, warnings = [], []
    if tau < physics.min_tau:
        problems.append(f"tau {tau:.3f} < {physics.min_tau} (unstable): lower Re or enlarge the shape")
    if blockage > physics.max_blockage:
        problems.append(f"blockage {blockage:.0%} > {physics.max_blockage:.0%}: make the shape smaller")
    if not sampling.re_min <= re <= sampling.re_max:
        warnings.append(f"Re outside the training range {sampling.re_min:g}-{sampling.re_max:g}")
    if D < sampling.d_min:
        warnings.append(f"D = {D:.1f} below the training minimum {sampling.d_min:g} cells")
    if max(dx, D) > sampling.crop - sampling.crop_margin:
        warnings.append("shape larger than the CNN input window")
    if shape != "circle" and min(size1, size2) < sampling.min_thickness:
        warnings.append(f"side below {sampling.min_thickness:g} cells (thinner than any training shape)")
    return CaseInfo(D, blockage, tau, problems, warnings)


def solver_args(shape: str, size1: float, size2: float, angle: float, re: float,
                domain: DomainParams = DOMAIN, physics: PhysicsParams = PHYSICS,
                mean_fields: bool = False) -> list[str]:
    """Solver arguments with the dataset settings (without --out-dir)."""
    if shape == "circle":
        size2, angle = size1, 0.0
    return [*global_args(domain, physics, mean_fields), "--shape", shape, "--size1", repr(float(size1)),
            "--size2", repr(float(size2)), "--angle", repr(float(angle)), "--re", repr(float(re))]


def obstacle_mask(shape: str, size1: float, size2: float, angle: float, re: float,
                  exe: Path = SOLVER_EXE, domain: DomainParams = DOMAIN,
                  physics: PhysicsParams = PHYSICS, sampling: SamplingParams = SAMPLING) -> np.ndarray:
    """The cropped mask exactly as in the dataset, from a one-step solver run (~0.1 s).

    Re only matters for the solver's tau check; an infeasible Re is replaced by a safe one,
    because the mask depends on geometry only.
    """
    info = case_info(shape, size1, size2, angle, re, domain, physics, sampling)
    if info.blockage > physics.max_blockage:
        raise ValueError(info.problems[-1])
    safe_re = re if info.tau >= physics.min_tau else 1.0
    args = solver_args(shape, size1, size2, angle, safe_re, domain, physics)
    i = args.index("--warmup")
    args[i + 1] = "1"
    args[args.index("--avg") + 1] = "1"
    with tempfile.TemporaryDirectory(prefix="cfd_mask_") as tmp:
        proc = subprocess.run([str(exe), "--out-dir", tmp, *args], capture_output=True, text=True, timeout=60)
        if proc.returncode not in (0, 3) or not (Path(tmp) / "mask.npy").exists():
            raise RuntimeError(f"solver failed (exit {proc.returncode}): {proc.stderr.strip()[:200]}")
        return crop_mask(np.load(Path(tmp) / "mask.npy"), domain, sampling.crop)


class Surrogate:
    """The trained CNN on the CPU: predict(mask, re) -> (Cd, Cl)."""

    def __init__(self, path: Path = MODEL_PATH):
        import torch  # optional dependency

        from .surrogate.train import load_model
        torch.set_num_threads(max(1, min(4, torch.get_num_threads())))
        self._torch = torch
        self.model, self.norm, self.ckpt = load_model(Path(path), "cpu")

    def predict(self, mask: np.ndarray, re: float) -> tuple[float, float]:
        x = self._torch.as_tensor(mask[None, None].astype(np.float32))
        r = self._torch.tensor([[(math.log(re) - self.norm.log_re_mean) / self.norm.log_re_std]],
                               dtype=self._torch.float32)
        with self._torch.no_grad():
            y = self.model(x, r).numpy()
        cd, cl = self.norm.y_inverse(y)[0]
        return float(cd), float(cl)


def solver_field(run_dir: Path, domain: DomainParams = DOMAIN, physics: PhysicsParams = PHYSICS) -> np.ndarray:
    """The field window (ux / U, uy / U, Cp) of a solver run made with mean_fields=True."""
    return crop_field(np.load(Path(run_dir) / "fields_mean.npy"), domain, physics, FIELD).astype(np.float32)


class FieldSurrogate:
    """The trained U-Net on the CPU: predict(mask, re) -> (3, H, W) time-averaged (ux / U, uy / U, Cp)
    in the field window; `mask` is the 64 x 64 crop from obstacle_mask()."""

    def __init__(self, path: Path = FIELD_MODEL_PATH):
        import torch  # optional dependency

        from .surrogate.field_train import load_field_model
        torch.set_num_threads(max(1, min(4, torch.get_num_threads())))
        self.model, self.norm, self.params, self.ckpt = load_field_model(Path(path), "cpu")

    def window_mask(self, mask: np.ndarray) -> np.ndarray:
        return window_mask(mask, DOMAIN, SAMPLING, FIELD)

    def predict(self, mask: np.ndarray, re: float) -> np.ndarray:
        from .surrogate.field_train import predict_fields
        return predict_fields(self.model, self.norm, self.params, self.window_mask(mask)[None],
                              np.array([re], float))[0]
