"""Flow-field surrogate data: inputs, normalisation, metrics and baselines (numpy only, no torch).

A sample is the time-averaged field window (3, H, W) = (ux / U, uy / U, Cp) of one case.
Inputs: obstacle mask in the window, its signed distance field and standardised log(Re).
Errors are measured on fluid cells only.
"""
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.ndimage import distance_transform_edt

from ..dataset import load, load_fields, window_mask
from ..params import FIELD_TRAIN, DomainParams, FieldParams, FieldTrainParams, SamplingParams
from .data import tabular_features

CHANNELS = ["ux", "uy", "Cp"]


def dataset_params(data_dir: Path) -> tuple[DomainParams, SamplingParams, FieldParams]:
    """Domain, sampling and field-window params a field dataset was made with (config.json)."""
    cfg = json.loads((Path(data_dir) / "config.json").read_text())
    if "fields" not in cfg:
        raise ValueError(f"{data_dir} is not a field dataset (no 'fields' in config.json)")
    sampling = SamplingParams(**{k: tuple(v) if isinstance(v, list) else v for k, v in cfg["sampling"].items()})
    return DomainParams(**cfg["domain"]), sampling, FieldParams(**cfg["fields"])


def load_field_dataset(data_dir: Path) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """ok rows of a field dataset: (df, window masks (n, H, W) uint8, fields (n, 3, H, W) float16)."""
    domain, sampling, fp = dataset_params(data_dir)
    df_all, _ = load(data_dir, only_ok=False)
    keep = np.flatnonzero(df_all["ok"].to_numpy(bool))
    _, masks = load(data_dir, only_ok=True)
    fields = np.asarray(load_fields(data_dir)[keep])
    wmasks = np.stack([window_mask(m, domain, sampling, fp) for m in masks])
    return df_all.iloc[keep].reset_index(drop=True), wmasks, fields


def signed_distance(mask: np.ndarray) -> np.ndarray:
    """Distance [cells] to the obstacle boundary: positive in the fluid, negative inside."""
    solid = mask.astype(bool)
    if not solid.any():
        return np.full(mask.shape, float(max(mask.shape)), np.float32)
    outside = distance_transform_edt(~solid)
    inside = distance_transform_edt(solid)
    return (outside - inside).astype(np.float32)


@dataclass
class FieldNormaliser:
    """Per-channel mean/std of the target fields (fluid cells of the training rows) and log(Re)."""
    mean: list[float]
    std: list[float]
    log_re_mean: float
    log_re_std: float

    @classmethod
    def fit(cls, df: pd.DataFrame, masks: np.ndarray, fields: np.ndarray) -> "FieldNormaliser":
        fluid = masks == 0
        mean, std = [], []
        for c in range(fields.shape[1]):
            v = fields[:, c][fluid].astype(np.float64)
            mean.append(float(v.mean()))
            std.append(float(v.std()))
        log_re = np.log(df["re"].to_numpy(float))
        return cls(mean, std, float(log_re.mean()), float(log_re.std()))

    def re(self, re: np.ndarray) -> np.ndarray:
        return ((np.log(np.asarray(re, float)) - self.log_re_mean) / self.log_re_std).astype(np.float32)

    def to_std(self, fields: np.ndarray) -> np.ndarray:
        m = np.asarray(self.mean, np.float32)[:, None, None]
        s = np.asarray(self.std, np.float32)[:, None, None]
        return (fields.astype(np.float32) - m) / s

    def from_std(self, fields: np.ndarray) -> np.ndarray:
        m = np.asarray(self.mean, np.float32)[:, None, None]
        s = np.asarray(self.std, np.float32)[:, None, None]
        return np.asarray(fields, np.float32) * s + m


def make_inputs(masks: np.ndarray, re: np.ndarray, norm: FieldNormaliser,
                params: FieldTrainParams = FIELD_TRAIN) -> np.ndarray:
    """(n, 3, H, W) float32 network input: mask, scaled signed distance, standardised log(Re)."""
    n, h, w = masks.shape
    x = np.empty((n, 3, h, w), np.float32)
    x[:, 0] = masks
    for i in range(n):
        x[i, 1] = signed_distance(masks[i]) / params.sdf_scale
    x[:, 2] = norm.re(re)[:, None, None]
    return x


# ---------------------------------------------------------------------------------------
# Metrics (physical units: u / U and Cp; fluid cells only)
# ---------------------------------------------------------------------------------------
def field_errors(true: np.ndarray, pred: np.ndarray, masks: np.ndarray, wake_x0: int) -> pd.DataFrame:
    """Per-sample errors on fluid cells.

    u_rel_l2: ||u_pred - u|| / ||u|| over the velocity vector field; cp_rel_l2: the same for Cp;
    u_mae, cp_mae: mean absolute error; *_wake: the same metrics for x >= wake_x0 (downstream of
    the obstacle centre).
    """
    rows = []
    for t, p, m in zip(true.astype(np.float32), pred.astype(np.float32), masks):
        fluid = m == 0
        r = {}
        for suffix, region in [("", fluid), ("_wake", fluid & (np.arange(m.shape[1]) >= wake_x0)[None, :])]:
            du = p[:2, region] - t[:2, region]
            dp = p[2, region] - t[2, region]
            r["u_rel_l2" + suffix] = float(np.sqrt((du ** 2).sum() / (t[:2, region] ** 2).sum()))
            r["cp_rel_l2" + suffix] = float(np.sqrt((dp ** 2).sum() / max((t[2, region] ** 2).sum(), 1e-12)))
            r["u_mae" + suffix] = float(np.abs(du).mean())
            r["cp_mae" + suffix] = float(np.abs(dp).mean())
        rows.append(r)
    return pd.DataFrame(rows)


def summarise(errors: pd.DataFrame, df: pd.DataFrame | None = None) -> dict:
    """Mean of each error column, overall and per shape / flow regime."""
    out = {"all": {**errors.mean().to_dict(), "u_rel_l2_median": float(errors["u_rel_l2"].median()),
                   "u_rel_l2_p95": float(errors["u_rel_l2"].quantile(0.95)), "n": len(errors)}}
    if df is not None:
        for col in ["shape", "status"]:
            for value in sorted(df[col].unique()):
                sel = (df[col] == value).to_numpy()
                out[f"{col}={value}"] = {**errors[sel].mean().to_dict(), "n": int(sel.sum())}
    return out


# ---------------------------------------------------------------------------------------
# Baselines
# ---------------------------------------------------------------------------------------
def mean_field_baseline(train_fields: np.ndarray, n: int) -> np.ndarray:
    """The average training field for every test sample."""
    mean = train_fields.astype(np.float32).mean(axis=0)
    return np.broadcast_to(mean, (n, *mean.shape)).copy()


def nearest_neighbour_baseline(train_df: pd.DataFrame, train_fields: np.ndarray,
                               test_df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Field of the closest training case in standardised shape-parameter space (same features as
    the gradient-boosting baseline of M5). Returns (fields, neighbour indices into train)."""
    a, b = tabular_features(train_df), tabular_features(test_df)
    mu, sd = a.mean(axis=0), a.std(axis=0)
    sd[sd == 0] = 1.0
    a, b = (a - mu) / sd, (b - mu) / sd
    idx = np.array([int(np.argmin(((a - v) ** 2).sum(axis=1))) for v in b])
    return train_fields[idx].astype(np.float32), idx
