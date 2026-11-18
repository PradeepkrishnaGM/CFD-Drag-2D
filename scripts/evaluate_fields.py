"""Evaluate a trained U-Net field surrogate on the test split and compare it with the baselines.

Example:
    python scripts/evaluate_fields.py --data dataset/fields_v1/cfd_drag_2d_fields --run runs/fields_v1

Writes into --run: metrics.json (U-Net, mean-field and nearest-neighbour baselines, overall,
in the wake and per shape / regime), errors.csv (per test row) and fields.png (examples).
Exits 1 if the U-Net does not beat both baselines on the mean velocity and Cp relative L2 errors.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from cfd_drag.params import TrainParams  # noqa: E402
from cfd_drag.surrogate.data import split_indices  # noqa: E402
from cfd_drag.surrogate.field_train import load_field_model, predict_fields  # noqa: E402
from cfd_drag.surrogate.fields import (dataset_params, field_errors, load_field_dataset,  # noqa: E402
                                       mean_field_baseline, nearest_neighbour_baseline, summarise)


def example_plot(test, true, pred, masks, errors, path: Path):
    """Rows: test cases at the 10th, 50th and 90th percentile of the U-Net velocity error."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    order = np.argsort(errors["u_rel_l2"].to_numpy())
    picks = [order[int(q * (len(order) - 1))] for q in (0.1, 0.5, 0.9)]
    fig, axes = plt.subplots(len(picks), 4, figsize=(16, 2.3 * len(picks)), squeeze=False)
    for row, i in zip(axes, picks):
        solid = masks[i] != 0
        speed = [np.hypot(f[0], f[1]) for f in (true[i], pred[i])]
        vmax = float(max(s.max() for s in speed))
        cp_lim = float(np.abs(true[i, 2]).max())
        panels = [(speed[0], "|u|/U solver", "viridis", 0, vmax),
                  (speed[1], "|u|/U U-Net", "viridis", 0, vmax),
                  (np.abs(speed[1] - speed[0]), "| |u| error |", "magma", 0, 0.25 * vmax),
                  (pred[i, 2] - true[i, 2], "Cp error", "RdBu_r", -0.25 * cp_lim, 0.25 * cp_lim)]
        for ax, (img, title, cmap, lo, hi) in zip(row, panels):
            im = ax.imshow(np.ma.masked_where(solid, img), origin="lower", cmap=cmap, vmin=lo, vmax=hi)
            ax.set_xticks([]), ax.set_yticks([])
            ax.set_title(title, fontsize=9)
            fig.colorbar(im, ax=ax, fraction=0.025, pad=0.01)
        r = test.iloc[i]
        row[0].set_ylabel(f"{r['shape']}, Re {r['re']:.0f}\nerr {errors['u_rel_l2'].iloc[i]:.1%}", fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--run", type=Path, required=True)
    args = ap.parse_args()
    df, masks, fields = load_field_dataset(args.data)
    model, norm, params, ckpt = load_field_model(args.run / "model.pt")
    if ckpt["n_rows"] != len(df):
        sys.exit(f"model was trained on {ckpt['n_rows']} rows, dataset has {len(df)}")
    split = split_indices(len(df), TrainParams(**{**ckpt["split_params"],
                                                  "channels": tuple(ckpt["split_params"]["channels"])}))
    tr, te = split["train"], split["test"]
    test = df.iloc[te].reset_index(drop=True)
    true, m_te = fields[te].astype(np.float32), masks[te]
    wake_x0 = dataset_params(args.data)[2].x_before

    pred = predict_fields(model, norm, params, m_te, test["re"].to_numpy(float))
    nn_pred, _ = nearest_neighbour_baseline(df.iloc[tr], fields[tr], test)
    preds = {"unet": pred, "mean": mean_field_baseline(fields[tr], len(te)), "nearest": nn_pred}
    errs = {k: field_errors(true, p, m_te, wake_x0) for k, p in preds.items()}
    metrics = {k: summarise(e, test) for k, e in errs.items()}
    metrics.update(best_epoch=ckpt["best_epoch"], train_seconds=ckpt["train_seconds"],
                   device=ckpt["device"], n_parameters=ckpt["n_parameters"])
    a = {k: metrics[k]["all"] for k in preds}
    metrics["passed"] = all(a["unet"][c] < a[b][c] for b in ("mean", "nearest") for c in ("u_rel_l2", "cp_rel_l2"))
    (args.run / "metrics.json").write_text(json.dumps(metrics, indent=1))
    out = test[["index", "shape", "re", "D", "status"]].copy()
    for k, e in errs.items():
        for c in ("u_rel_l2", "cp_rel_l2"):
            out[f"{k}_{c}"] = e[c].to_numpy()
    out.to_csv(args.run / "errors.csv", index=False, float_format="%.6g")
    example_plot(test, true, pred, m_te, errs["unet"], args.run / "fields.png")

    print(f"{'test (n=' + str(len(te)) + ')':14s} {'u relL2':>8s} {'Cp relL2':>9s} {'u wake':>8s} {'Cp wake':>8s} {'u MAE':>7s} {'Cp MAE':>7s}")
    for k in ["mean", "nearest", "unet"]:
        r = a[k]
        print(f"{k:14s} {r['u_rel_l2']:8.2%} {r['cp_rel_l2']:9.2%} {r['u_rel_l2_wake']:8.2%} "
              f"{r['cp_rel_l2_wake']:8.2%} {r['u_mae']:7.4f} {r['cp_mae']:7.4f}")
    print("PASSED: U-Net beats both baselines on velocity and Cp errors" if metrics["passed"]
          else "FAILED: U-Net does not beat both baselines")
    sys.exit(0 if metrics["passed"] else 1)


if __name__ == "__main__":
    main()
