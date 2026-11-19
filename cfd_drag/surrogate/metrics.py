"""Regression metrics, overall and per group (shape, flow regime)."""
import numpy as np
import pandas as pd

from .data import TARGETS

RE_BANDS = (20, 40, 60, 80, 100)  # Re breakdown edges (M10); the last band includes 100


def regression(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """MAE and R² for Cd and Cl; relative error for Cd only (Cl is often ~0)."""
    out = {}
    for j, name in enumerate(TARGETS):
        t, p = y_true[:, j], y_pred[:, j]
        ss_res = float(((t - p) ** 2).sum())
        ss_tot = float(((t - t.mean()) ** 2).sum())
        out[f"{name}_mae"] = float(np.abs(t - p).mean())
        out[f"{name}_r2"] = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    rel = np.abs(y_pred[:, 0] - y_true[:, 0]) / np.abs(y_true[:, 0])
    out["Cd_rel_mean"] = float(rel.mean())
    out["Cd_rel_median"] = float(np.median(rel))
    out["Cd_rel_p95"] = float(np.quantile(rel, 0.95))
    out["n"] = int(len(t))
    return out


def report(df: pd.DataFrame, y_pred: np.ndarray) -> dict:
    """Overall metrics plus breakdowns by shape, status (steady/periodic) and Re band."""
    y_true = df[TARGETS].to_numpy(float)
    res = {"all": regression(y_true, y_pred)}
    for col in ["shape", "status"]:
        for value in sorted(df[col].unique()):
            m = (df[col] == value).to_numpy()
            if m.sum() >= 2:
                res[f"{col}={value}"] = regression(y_true[m], y_pred[m])
    re = df["re"].to_numpy(float)
    for lo, hi in zip(RE_BANDS[:-1], RE_BANDS[1:]):
        m = (re >= lo) & ((re < hi) | ((hi == RE_BANDS[-1]) & (re <= hi)))
        if m.sum() >= 2:
            res[f"re={lo}-{hi}"] = regression(y_true[m], y_pred[m])
    return res
