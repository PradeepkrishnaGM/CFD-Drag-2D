"""Dataset split, normalisation and tabular features (numpy only, no torch)."""
from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..params import TRAIN, TrainParams

TARGETS = ["Cd", "Cl"]
SHAPES = ["circle", "rect", "tri"]


def split_indices(n: int, params: TrainParams = TRAIN) -> dict[str, np.ndarray]:
    """Seeded split of rows 0..n-1 into train/val/test (sorted index arrays)."""
    perm = np.random.default_rng(params.split_seed).permutation(n)
    n_test = int(round(params.test_fraction * n))
    n_val = int(round(params.val_fraction * n))
    return {"test": np.sort(perm[:n_test]), "val": np.sort(perm[n_test:n_test + n_val]),
            "train": np.sort(perm[n_test + n_val:])}


@dataclass
class Normaliser:
    """Standardises log(Re) and the targets with statistics from the training rows."""
    log_re_mean: float
    log_re_std: float
    y_mean: list[float]
    y_std: list[float]

    @classmethod
    def fit(cls, df: pd.DataFrame) -> "Normaliser":
        log_re = np.log(df["re"].to_numpy(float))
        y = df[TARGETS].to_numpy(float)
        return cls(float(log_re.mean()), float(log_re.std()),
                   y.mean(axis=0).tolist(), y.std(axis=0).tolist())

    def re(self, df: pd.DataFrame) -> np.ndarray:
        return ((np.log(df["re"].to_numpy(float)) - self.log_re_mean) / self.log_re_std)[:, None]

    def y(self, df: pd.DataFrame) -> np.ndarray:
        return (df[TARGETS].to_numpy(float) - self.y_mean) / self.y_std

    def y_inverse(self, y_std: np.ndarray) -> np.ndarray:
        return np.asarray(y_std) * self.y_std + self.y_mean


def tabular_features(df: pd.DataFrame) -> np.ndarray:
    """Shape parameters for the gradient-boosting baseline.

    Angle enters as sin/cos of 2*angle (rectangles repeat every 180 deg) and of angle.
    """
    a = np.deg2rad(df["angle"].to_numpy(float))
    onehot = np.stack([(df["shape"] == s).to_numpy(float) for s in SHAPES], axis=1)
    cols = [df[k].to_numpy(float) for k in ["size1", "size2", "re", "D", "blockage"]]
    return np.column_stack([onehot, *cols, np.sin(a), np.cos(a), np.sin(2 * a), np.cos(2 * a)])
