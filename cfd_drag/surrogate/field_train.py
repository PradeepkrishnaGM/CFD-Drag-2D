"""Training loop, checkpoint I/O and prediction for the U-Net field surrogate (CPU or CUDA)."""
import json
import os
import time
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from ..params import FIELD_TRAIN, TRAIN, FieldTrainParams, TrainParams
from .data import split_indices
from .fields import FieldNormaliser, make_inputs
from .train import pick_device
from .unet import FieldUNet, n_parameters


def _masked_mse(pred, target, fluid, weights):
    """Mean over fluid cells of the channel-weighted squared error."""
    err = ((pred - target) ** 2 * weights[None, :, None, None]).sum(dim=1)
    return (err * fluid).sum() / fluid.sum().clamp(min=1)


@torch.no_grad()
def predict_std(model, x: torch.Tensor, batch: int = 64, amp: bool = False) -> torch.Tensor:
    """Standardised predictions for inputs x (on the model's device), float32."""
    model.eval()
    outs = []
    for i in range(0, len(x), batch):
        with torch.autocast(x.device.type, dtype=torch.float16, enabled=amp):
            outs.append(model(x[i:i + batch].float()).float())
    return torch.cat(outs)


def _val_error(pred_std, y_std, fluid, norm: FieldNormaliser) -> float:
    """Validation score: mean over samples of (velocity rel. L2 + Cp rel. L2) / 2, physical units."""
    m = torch.tensor(norm.mean, device=pred_std.device)[None, :, None, None]
    s = torch.tensor(norm.std, device=pred_std.device)[None, :, None, None]
    p, t = pred_std * s + m, y_std.float() * s + m
    f = fluid[:, None]
    du = (((p[:, :2] - t[:, :2]) ** 2) * f).sum(dim=(1, 2, 3)) / ((t[:, :2] ** 2) * f).sum(dim=(1, 2, 3))
    dp = (((p[:, 2:] - t[:, 2:]) ** 2) * f).sum(dim=(1, 2, 3)) / ((t[:, 2:] ** 2) * f).sum(dim=(1, 2, 3)).clamp(min=1e-12)
    return float(((du.sqrt() + dp.sqrt()) / 2).mean())


def train_fields(df: pd.DataFrame, masks: np.ndarray, fields: np.ndarray, out_dir: Path,
                 params: FieldTrainParams = FIELD_TRAIN, split_params: TrainParams = TRAIN,
                 device: str = "auto", log=print) -> dict:
    """Train on the train rows (same split as the M5 CNN); keep the epoch with the lowest
    validation error. Writes out_dir/model.pt and out_dir/history.json; returns the checkpoint."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    dev = pick_device(device)
    amp = params.amp and dev.type == "cuda"
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.use_deterministic_algorithms(True, warn_only=True)  # some up-sampling kernels have no deterministic CUDA version
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(params.seed)

    split = split_indices(len(df), split_params)
    tr, va = split["train"], split["val"]
    norm = FieldNormaliser.fit(df.iloc[tr], masks[tr], fields[tr])
    store = torch.float16 if dev.type == "cuda" else torch.float32  # keeps the whole set on a 16 GB GPU

    def tensors(idx):
        x = torch.as_tensor(make_inputs(masks[idx], df["re"].to_numpy(float)[idx], norm, params)).to(dev, store)
        y = torch.as_tensor(norm.to_std(fields[idx])).to(dev, store)
        fluid = torch.as_tensor(masks[idx] == 0).to(dev, torch.float32)
        return x, y, fluid

    x_tr, y_tr, f_tr = tensors(tr)
    x_va, y_va, f_va = tensors(va)
    model = FieldUNet(params).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=params.lr, weight_decay=params.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=params.epochs)
    scaler = torch.amp.GradScaler(dev.type, enabled=amp)
    w = torch.tensor(params.loss_weights, device=dev)
    gen = torch.Generator().manual_seed(params.seed)
    log(f"U-Net {n_parameters(model) / 1e6:.2f} M parameters | train {len(tr)} | val {len(va)} | "
        f"{dev} | amp {amp}")

    best, best_state, best_epoch, history = float("inf"), None, 0, []
    t0 = time.time()
    for epoch in range(1, params.epochs + 1):
        model.train()
        perm = torch.randperm(len(tr), generator=gen).to(dev)
        total, count = 0.0, 0
        for i in range(0, len(perm), params.batch_size):
            b = perm[i:i + params.batch_size]
            if len(b) < 2:  # BatchNorm needs more than one sample
                continue
            with torch.autocast(dev.type, dtype=torch.float16, enabled=amp):
                pred = model(x_tr[b].float())
            loss = _masked_mse(pred.float(), y_tr[b].float(), f_tr[b], w)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            total += loss.item() * len(b)
            count += len(b)
        sched.step()
        val = _val_error(predict_std(model, x_va, amp=amp), y_va, f_va, norm)
        history.append({"epoch": epoch, "train_loss": total / max(count, 1), "val_error": val,
                        "seconds": round(time.time() - t0, 1)})
        if val < best:
            best, best_epoch = val, epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        if epoch == 1 or epoch % 5 == 0 or epoch == params.epochs:
            log(f"epoch {epoch:4d}  train loss {history[-1]['train_loss']:.4f}  val error {val:.4f}  "
                f"best {best:.4f} (epoch {best_epoch})  {history[-1]['seconds']:.0f} s")

    ckpt = {"state_dict": best_state, "params": asdict(params), "split_params": asdict(split_params),
            "normaliser": asdict(norm), "best_epoch": best_epoch, "best_val_error": best,
            "n_rows": len(df), "n_parameters": n_parameters(model),
            "train_seconds": round(time.time() - t0, 1), "device": str(dev), "torch": torch.__version__}
    torch.save(ckpt, out_dir / "model.pt")
    (out_dir / "history.json").write_text(json.dumps(history, indent=1))
    return ckpt


def load_field_model(path: Path, device: str = "cpu") -> tuple[FieldUNet, FieldNormaliser, FieldTrainParams, dict]:
    ckpt = torch.load(path, map_location=device, weights_only=False)
    p = dict(ckpt["params"])
    p["loss_weights"] = tuple(p["loss_weights"])
    params = replace(FIELD_TRAIN, **p)
    model = FieldUNet(params).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model, FieldNormaliser(**ckpt["normaliser"]), params, ckpt


def predict_fields(model, norm: FieldNormaliser, params: FieldTrainParams, masks: np.ndarray,
                   re: np.ndarray, device: str = "cpu", batch: int = 64) -> np.ndarray:
    """(n, 3, H, W) float32 fields in physical units (ux / U, uy / U, Cp); 0 on obstacle cells."""
    x = torch.as_tensor(make_inputs(masks, np.asarray(re, float), norm, params), device=device)
    pred = norm.from_std(predict_std(model, x, batch).cpu().numpy())
    pred[np.broadcast_to((masks != 0)[:, None], pred.shape)] = 0.0
    return pred
