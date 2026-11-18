"""Evaluate a trained surrogate on the test split and compare it with the baselines.

Example:
    python scripts/evaluate.py --data dataset/kaggle_v1/cfd_drag_2d --run runs/v1

Writes into --run: metrics.json (CNN, mean and gradient-boosting baselines, per shape and
regime), predictions.csv (test rows) and parity.png. Exits 1 if the CNN does not beat both
baselines on test Cd MAE.
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from cfd_drag.dataset import load  # noqa: E402
from cfd_drag.surrogate.baselines import run_baselines  # noqa: E402
from cfd_drag.surrogate.data import split_indices  # noqa: E402
from cfd_drag.surrogate.metrics import report  # noqa: E402
from cfd_drag.surrogate.train import load_model, predict_df  # noqa: E402
from cfd_drag.params import TRAIN  # noqa: E402


def parity_plot(test, pred, path: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(9, 4.2))
    for ax, j, name in zip(axes, [0, 1], ["Cd", "Cl"]):
        t = test[name].to_numpy()
        for shape, marker in [("circle", "o"), ("rect", "s"), ("tri", "^")]:
            m = (test["shape"] == shape).to_numpy()
            ax.scatter(t[m], pred[m, j], s=8, alpha=0.6, marker=marker, label=shape)
        lo, hi = min(t.min(), pred[:, j].min()), max(t.max(), pred[:, j].max())
        ax.plot([lo, hi], [lo, hi], "k--", lw=1)
        ax.set_xlabel(f"{name} (LBM solver)")
        ax.set_ylabel(f"{name} (CNN)")
        ax.set_title(f"Test set: {name}")
        ax.legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--run", type=Path, required=True)
    args = ap.parse_args()
    df, masks = load(args.data)
    model, norm, ckpt = load_model(args.run / "model.pt")
    if ckpt["n_rows"] != len(df):
        sys.exit(f"model was trained on {ckpt['n_rows']} rows, dataset has {len(df)}")
    params = TRAIN.__class__(**{**ckpt["params"], "channels": tuple(ckpt["params"]["channels"])})
    split = split_indices(len(df), params)
    test = df.iloc[split["test"]].reset_index(drop=True)
    pred = predict_df(model, norm, test, masks[split["test"]])

    metrics = {"cnn": report(test, pred), **run_baselines(df, split, params),
               "best_epoch": ckpt["best_epoch"], "train_seconds": ckpt["train_seconds"],
               "device": ckpt["device"]}
    cd = {k: metrics[k]["all"]["Cd_mae"] for k in ["cnn", "gbm", "mean"]}
    metrics["passed"] = cd["cnn"] < cd["gbm"] and cd["cnn"] < cd["mean"]
    (args.run / "metrics.json").write_text(json.dumps(metrics, indent=1))
    out = test[["index", "shape", "re", "D", "status", "Cd", "Cl"]].copy()
    out["Cd_pred"], out["Cl_pred"] = pred[:, 0], pred[:, 1]
    out.to_csv(args.run / "predictions.csv", index=False, float_format="%.6g")
    parity_plot(test, pred, args.run / "parity.png")

    print(f"{'test (n=' + str(len(test)) + ')':14s} {'Cd MAE':>8s} {'Cd R2':>7s} {'Cd rel':>7s} {'Cl MAE':>8s} {'Cl R2':>7s}")
    for k in ["mean", "gbm", "cnn"]:
        a = metrics[k]["all"]
        print(f"{k:14s} {a['Cd_mae']:8.4f} {a['Cd_r2']:7.3f} {a['Cd_rel_mean']:7.2%} {a['Cl_mae']:8.4f} {a['Cl_r2']:7.3f}")
    print("PASSED: CNN beats both baselines on test Cd MAE" if metrics["passed"]
          else "FAILED: CNN does not beat both baselines on test Cd MAE")
    sys.exit(0 if metrics["passed"] else 1)


if __name__ == "__main__":
    main()
