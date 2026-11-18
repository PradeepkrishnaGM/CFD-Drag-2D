"""Train the CNN surrogate (mask + Re -> Cd, Cl) on a consolidated dataset.

Example:
    python scripts/train.py --data dataset/kaggle_v1/cfd_drag_2d --out runs/v1
    python scripts/train.py --data example_data --out runs/smoke --epochs 2

Writes model.pt, history.json and split.json into --out. Defaults: cfd_drag/params.py (TrainParams).
"""
import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from cfd_drag.dataset import load  # noqa: E402
from cfd_drag.params import TRAIN  # noqa: E402
from cfd_drag.surrogate.data import split_indices  # noqa: E402
from cfd_drag.surrogate.train import train  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--epochs", type=int, default=TRAIN.epochs)
    ap.add_argument("--device", default="auto", help="auto, cpu or cuda")
    ap.add_argument("--seed", type=int, default=TRAIN.seed, help="weight init and batch order (split stays fixed)")
    args = ap.parse_args()
    params = replace(TRAIN, epochs=args.epochs, seed=args.seed)
    df, masks = load(args.data)
    args.out.mkdir(parents=True, exist_ok=True)
    split = split_indices(len(df), params)
    (args.out / "split.json").write_text(json.dumps(
        {"data": str(args.data), "n_rows": len(df), **{k: v.tolist() for k, v in split.items()}}))
    ckpt = train(df, masks, args.out, params, args.device)
    print(f"best epoch {ckpt['best_epoch']}, val Cd MAE {ckpt['best_val_cd_mae']:.4f}, "
          f"{ckpt['train_seconds']:.0f} s on {ckpt['device']} -> {args.out / 'model.pt'}")


if __name__ == "__main__":
    main()
