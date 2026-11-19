"""Training loop, checkpoint I/O and prediction (CPU or CUDA, same code)."""
import json
import os
import time
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from ..params import TRAIN, TrainParams
from .data import TARGETS, Normaliser, flip_y_std, re_weights, split_indices
from .model import SurrogateCNN


def pick_device(name: str = "auto") -> torch.device:
    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.device(name)


def _tensors(df, masks, norm: Normaliser, device):
    x = torch.as_tensor(masks[:, None].astype(np.float32), device=device)
    re = torch.as_tensor(norm.re(df).astype(np.float32), device=device)
    has_y = all(t in df.columns for t in TARGETS)  # prediction-only rows have no targets
    y = torch.as_tensor(norm.y(df).astype(np.float32), device=device) if has_y else None
    return x, re, y


@torch.no_grad()
def predict(model, x, re, batch: int = 512) -> np.ndarray:
    model.eval()
    return torch.cat([model(x[i:i + batch], re[i:i + batch]) for i in range(0, len(x), batch)]).cpu().numpy()


def train(df: pd.DataFrame, masks: np.ndarray, out_dir: Path, params: TrainParams = TRAIN,
          device: str = "auto", log=print) -> dict:
    """Train on the split's train rows, keep the epoch with the lowest validation Cd MAE.

    Writes out_dir/model.pt and out_dir/history.json; returns the checkpoint dict.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    dev = pick_device(device)
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.use_deterministic_algorithms(True, warn_only=dev.type == "cuda")
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(params.seed)

    split = split_indices(len(df), params)
    tr, va = df.iloc[split["train"]], df.iloc[split["val"]]
    norm = Normaliser.fit(tr)
    x_tr, re_tr, y_tr = _tensors(tr, masks[split["train"]], norm, dev)
    x_va, re_va, _ = _tensors(va, masks[split["val"]], norm, dev)
    cd_va = va["Cd"].to_numpy(float)

    model = SurrogateCNN(params).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=params.lr, weight_decay=params.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=params.epochs)
    w = torch.tensor([params.loss_weight_cd, params.loss_weight_cl], device=dev)
    gen = torch.Generator().manual_seed(params.seed)
    sw_tr = None
    if params.re_weight_bins > 0:
        sw_tr = torch.as_tensor(re_weights(tr["re"].to_numpy(float), params.re_weight_bins,
                                           params.re_weight_max).astype(np.float32), device=dev)

    best, best_state, history = float("inf"), None, []
    t0 = time.time()
    for epoch in range(1, params.epochs + 1):
        model.train()
        perm = torch.randperm(len(x_tr), generator=gen).to(dev)
        total = 0.0
        for i in range(0, len(perm), params.batch_size):
            b = perm[i:i + params.batch_size]
            if len(b) < 2:  # BatchNorm needs more than one sample
                continue
            xb, yb = x_tr[b], y_tr[b]
            if params.augment_flip:  # mirror about the channel centreline: Cd kept, Cl -> -Cl
                flip = (torch.rand(len(b), generator=gen) < 0.5).to(dev)
                xb = torch.where(flip[:, None, None, None], xb.flip(-2), xb)
                yb = torch.stack([yb[:, 0], torch.where(flip, flip_y_std(yb[:, 1], norm.y_mean[1], norm.y_std[1]),
                                                        yb[:, 1])], dim=1)
            sq = ((model(xb, re_tr[b]) - yb) ** 2) * w
            loss = (sq.mean(dim=1) * sw_tr[b]).mean() if sw_tr is not None else sq.mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            total += loss.item() * len(b)
        sched.step()
        val_cd = norm.y_inverse(predict(model, x_va, re_va))[:, 0]
        val_mae = float(np.abs(val_cd - cd_va).mean())
        history.append({"epoch": epoch, "train_loss": total / len(perm), "val_cd_mae": val_mae,
                        "seconds": round(time.time() - t0, 1)})
        if val_mae < best:
            best = val_mae
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            best_epoch = epoch
        if epoch == 1 or epoch % 10 == 0 or epoch == params.epochs:
            log(f"epoch {epoch:4d}  train loss {history[-1]['train_loss']:.4f}  "
                f"val Cd MAE {val_mae:.4f}  best {best:.4f} (epoch {best_epoch})  {history[-1]['seconds']:.0f} s")

    ckpt = {"state_dict": best_state, "params": asdict(params), "normaliser": asdict(norm),
            "best_epoch": best_epoch, "best_val_cd_mae": best, "n_rows": len(df),
            "train_seconds": round(time.time() - t0, 1), "device": str(dev),
            "torch": torch.__version__}
    torch.save(ckpt, out_dir / "model.pt")
    (out_dir / "history.json").write_text(json.dumps(history, indent=1))
    return ckpt


def load_model(path: Path, device: str = "cpu") -> tuple[SurrogateCNN, Normaliser, dict]:
    ckpt = torch.load(path, map_location=device, weights_only=False)
    p = dict(ckpt["params"])
    p["channels"] = tuple(p["channels"])
    model = SurrogateCNN(replace(TRAIN, **p)).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model, Normaliser(**ckpt["normaliser"]), ckpt


def predict_df(model, norm: Normaliser, df: pd.DataFrame, masks: np.ndarray, device: str = "cpu") -> np.ndarray:
    """Cd, Cl predictions in physical units for the given rows."""
    x, re, _ = _tensors(df, masks, norm, torch.device(device))
    return norm.y_inverse(predict(model, x, re))
