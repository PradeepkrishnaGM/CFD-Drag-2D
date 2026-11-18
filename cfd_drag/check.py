"""Sanity checks and summary statistics for a consolidated dataset (data.csv + masks.npy)."""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .params import CHECK, CheckParams

TARGETS = ["Cd", "Cl", "Cl_rms", "pressure_loss"]
INPUTS = ["shape", "size1", "size2", "angle", "re"]


def _quantiles(s: pd.Series) -> dict:
    q = s.quantile([0.0, 0.05, 0.5, 0.95, 1.0]).to_numpy()
    return dict(zip(["min", "p05", "median", "p95", "max"], (float(v) for v in q)))


def check(df: pd.DataFrame, masks: np.ndarray, config: dict | None = None,
          params: CheckParams = CHECK) -> tuple[dict, list[str]]:
    """Return (summary, problems). An empty problem list means the dataset passes."""
    problems = []
    n = len(df)
    ok = df["ok"].astype(bool).to_numpy()
    good = df[ok]

    if n == 0:
        return {"n": 0}, ["dataset is empty"]
    if not np.array_equal(df["index"].to_numpy(), np.arange(n)):
        problems.append("index column is not 0..n-1 in order")
    if len(masks) != n:
        problems.append(f"{len(masks)} masks for {n} rows")
    ok_fraction = ok.mean()
    if ok_fraction < params.min_ok_fraction:
        problems.append(f"ok fraction {ok_fraction:.3f} below {params.min_ok_fraction}")
    bad_targets = int((~np.isfinite(good[TARGETS].to_numpy(float))).any(axis=1).sum())
    if bad_targets:
        problems.append(f"{bad_targets} ok rows with non-finite targets")
    dups = int(df.duplicated(subset=INPUTS).sum())
    if dups:
        problems.append(f"{dups} duplicate input rows")

    solid = masks.reshape(len(masks), -1).mean(axis=1) if len(masks) else np.zeros(0)
    if len(masks) == n:
        empty = int((solid[ok] == 0).sum())
        if empty:
            problems.append(f"{empty} ok rows with an empty mask")

    if config:
        s, ph = config["sampling"], config["physics"]
        eps = 1e-9
        if ((df["re"] < s["re_min"] - eps) | (df["re"] > s["re_max"] + eps)).any():
            problems.append("Re outside the configured range")
        if (df["blockage"] > ph["max_blockage"] + eps).any():
            problems.append("blockage above the configured cap")
        if (df["D"] < s["d_min"] - eps).any():
            problems.append("D below d_min")
        if masks.ndim == 3 and masks.shape[1:] != (s["crop"], s["crop"]):
            problems.append(f"mask shape {masks.shape[1:]} != crop {s['crop']}")

    summary = {
        "n": n,
        "n_ok": int(ok.sum()),
        "ok_fraction": float(ok_fraction),
        "failures": df.loc[~ok, "error"].value_counts().to_dict(),
        "shapes": good["shape"].value_counts().to_dict(),
        "regimes": good["status"].value_counts().to_dict(),
        "mask_shape": list(masks.shape[1:]),
        "mask_solid_fraction": _quantiles(pd.Series(solid[ok])) if len(masks) == n and ok.any() else None,
        "inputs": {k: _quantiles(good[k]) for k in ["re", "D", "blockage", "angle"]} if ok.any() else {},
        "targets": {k: _quantiles(good[k]) for k in TARGETS} if ok.any() else {},
        "min_ok_fraction": params.min_ok_fraction,
        "problems": problems,
        "passed": not problems,
    }
    return summary, problems


def check_fields(fields: np.ndarray, masks: np.ndarray, df: pd.DataFrame, config: dict,
                 flux_tol: float = 0.05, chunk: int = 256) -> tuple[dict, list[str]]:
    """Checks for fields.npy: shape, finite values, no-slip on the obstacle, zeros for failed rows,
    and mass conservation (the time-averaged volume flux through every column of the window,
    the column mean of ux / U, stays within flux_tol of the flux through the first column)."""
    from .dataset import window_mask
    from .params import DomainParams, FieldParams, SamplingParams

    problems = []
    fp = FieldParams(**config["fields"])
    domain = DomainParams(**config["domain"])
    sampling = SamplingParams(**{k: tuple(v) if isinstance(v, list) else v
                                 for k, v in config["sampling"].items()})
    n = len(df)
    if fields.shape != (n, 3, fp.height, fp.width):
        return {}, [f"fields shape {fields.shape} != {(n, 3, fp.height, fp.width)}"]
    ok = df["ok"].astype(bool).to_numpy()
    nonfinite = slip = nonzero_failed = 0
    flux_dev = np.zeros(n)
    stats = {c: [np.inf, -np.inf] for c in ("ux", "uy", "Cp")}
    for a in range(0, n, chunk):
        f = np.asarray(fields[a:a + chunk], dtype=np.float32)
        nonfinite += int((~np.isfinite(f)).reshape(len(f), -1).any(axis=1).sum())
        for j, i in enumerate(range(a, a + len(f))):
            if not ok[i]:
                nonzero_failed += int(np.any(f[j] != 0))
                continue
            solid = window_mask(masks[i], domain, sampling, fp).astype(bool)
            slip += int(np.any(f[j, :2, solid] != 0))
            flux = f[j, 0].mean(axis=0)  # volume flux through each column (relative units)
            flux_dev[i] = np.abs(flux / flux[0] - 1.0).max()
        g = f[ok[a:a + len(f)]]
        for c, name in enumerate(stats):
            if len(g):
                stats[name] = [min(stats[name][0], float(g[:, c].min())), max(stats[name][1], float(g[:, c].max()))]
    if nonfinite:
        problems.append(f"{nonfinite} rows with non-finite field values")
    if slip:
        problems.append(f"{slip} ok rows with non-zero velocity on obstacle cells")
    if nonzero_failed:
        problems.append(f"{nonzero_failed} failed rows with a non-zero field")
    bad_flux = int((flux_dev[ok] > flux_tol).sum())
    if bad_flux:
        problems.append(f"{bad_flux} ok rows whose volume flux deviates more than {flux_tol} from the inlet")
    summary = {"shape": list(fields.shape), "range": stats, "flux_tol": flux_tol,
               "max_flux_deviation": float(flux_dev[ok].max()) if ok.any() else None,
               "median_flux_deviation": float(np.median(flux_dev[ok])) if ok.any() else None}
    return summary, problems


def check_dir(out: Path, params: CheckParams = CHECK) -> tuple[dict, list[str]]:
    """Check a dataset directory and write summary.json next to data.csv."""
    out = Path(out)
    df = pd.read_csv(out / "data.csv", keep_default_na=False, na_values=["nan", "NaN"])
    masks = np.load(out / "masks.npy")
    cfg_path = out / "config.json"
    config = json.loads(cfg_path.read_text()) if cfg_path.exists() else None
    summary, problems = check(df, masks, config, params)
    if config and "fields" in config:
        if not (out / "fields.npy").exists():
            problems.append("config has fields but fields.npy is missing")
        else:
            fsum, fprob = check_fields(np.load(out / "fields.npy", mmap_mode="r"), masks, df, config)
            summary["fields"] = fsum
            problems += fprob
        summary["problems"], summary["passed"] = problems, not problems
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    return summary, problems


def spot_check(out: Path, k: int, workers: int, rng_seed: int = 0, rel_tol: float = 1e-8) -> dict:
    """Re-solve k random saved cases with the CPU solver and compare with the stored rows.

    Uses the default params; refuses a dataset whose config.json was made with other settings.
    """
    from concurrent.futures import ProcessPoolExecutor

    from .dataset import _config, run_case
    from .params import DOMAIN, FIELD, PHYSICS, SAMPLING
    from .sampling import sample_case

    out = Path(out)
    cfg = json.loads((out / "config.json").read_text())
    fields = FIELD if "fields" in cfg else None
    expected = _config(cfg["seed"], "cpu", DOMAIN, PHYSICS, SAMPLING, fields)
    expected.pop("backend")  # pre-M3 datasets have no backend key; it does not affect results
    if {k: v for k, v in cfg.items() if k != "backend"} != expected:
        raise ValueError("config.json differs from the default params; spot check not possible")
    df = pd.read_csv(out / "data.csv", keep_default_na=False, na_values=["nan", "NaN"])
    masks = np.load(out / "masks.npy")
    idx = sorted(np.random.default_rng(rng_seed).choice(len(df), size=min(k, len(df)), replace=False).tolist())
    cases = [sample_case(cfg["seed"], i, DOMAIN, PHYSICS, SAMPLING) for i in idx]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        results = list(pool.map(run_case, cases, [DOMAIN] * len(cases), [PHYSICS] * len(cases),
                                [SAMPLING] * len(cases), [None] * len(cases), [fields] * len(cases)))
    stored_fields = np.load(out / "fields.npy", mmap_mode="r") if fields else None
    rows = []
    for i, (row, mask, field) in zip(idx, results):
        stored = df.loc[i]
        diffs = {t: abs(row[t] - stored[t]) / max(abs(stored[t]), 1e-12) for t in TARGETS}
        rows.append({"index": i, "ok_same": bool(row["ok"]) == bool(stored["ok"]),
                     "status_same": row["status"] == stored["status"],
                     "mask_same": bool(np.array_equal(mask, masks[i])),
                     "max_rel_diff": float(np.nanmax(list(diffs.values())))})
        if fields:  # float16 storage: identical unless a value sits on a rounding boundary
            d = np.abs(field.astype(np.float32) - np.asarray(stored_fields[i], np.float32))
            rows[-1]["field_max_abs_diff"] = float(d.max())
    passed = all(r["ok_same"] and r["status_same"] and r["mask_same"]
                 and not r["max_rel_diff"] > rel_tol and r.get("field_max_abs_diff", 0.0) <= 1e-2
                 for r in rows)
    return {"indices": idx, "rel_tol": rel_tol, "cases": rows, "passed": passed}
