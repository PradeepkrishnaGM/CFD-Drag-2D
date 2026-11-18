"""Train the U-Net flow-field surrogate (mask + Re -> time-averaged ux, uy, Cp).

Example:
    python scripts/train_fields.py --data dataset/fields_v1/cfd_drag_2d_fields --out runs/fields_v1
    python scripts/train_fields.py --data <tiny field dataset> --out runs/smoke --epochs 2

Writes model.pt and history.json into --out. Defaults: cfd_drag/params.py (FieldTrainParams);
the train/val/test split is the one of the M5 CNN (TrainParams.split_seed and fractions).
"""
import argparse
import sys
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from cfd_drag.params import FIELD_TRAIN  # noqa: E402
from cfd_drag.surrogate.field_train import train_fields  # noqa: E402
from cfd_drag.surrogate.fields import load_field_dataset  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--epochs", type=int, default=FIELD_TRAIN.epochs)
    ap.add_argument("--device", default="auto", help="auto, cpu or cuda")
    ap.add_argument("--seed", type=int, default=FIELD_TRAIN.seed, help="weight init and batch order (split stays fixed)")
    args = ap.parse_args()
    params = replace(FIELD_TRAIN, epochs=args.epochs, seed=args.seed)
    df, masks, fields = load_field_dataset(args.data)
    ckpt = train_fields(df, masks, fields, args.out, params, device=args.device)
    print(f"best epoch {ckpt['best_epoch']}, val error {ckpt['best_val_error']:.4f}, "
          f"{ckpt['train_seconds']:.0f} s on {ckpt['device']} -> {args.out / 'model.pt'}")


if __name__ == "__main__":
    main()
