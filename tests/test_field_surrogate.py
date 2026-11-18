"""Flow-field surrogate (M8): inputs, metrics, baselines, U-Net shapes, training determinism."""
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from cfd_drag.params import FIELD_TRAIN, TrainParams
from cfd_drag.surrogate.fields import (FieldNormaliser, field_errors, make_inputs,
                                       nearest_neighbour_baseline, signed_distance)

H, W = 32, 64


def _synthetic(n=12, seed=0):
    """Random circles in a 32 x 64 window with a smooth fake field around them."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:H, 0:W]
    masks, fields, rows = [], [], []
    for i in range(n):
        r, re = rng.uniform(3, 7), rng.uniform(20, 100)
        m = ((xx - 16) ** 2 + (yy - 16) ** 2 <= r * r).astype(np.uint8)
        d = np.hypot(xx - 16, yy - 16)
        f = np.stack([1 - np.exp(-d / r) * (1 + re / 100), 0.1 * np.sin(xx / 8) * r / 7, np.exp(-d / (2 * r))])
        f[:, m == 1] = 0
        masks.append(m)
        fields.append(f.astype(np.float16))
        rows.append({"index": i, "shape": "circle", "size1": r, "size2": r, "angle": 0.0, "re": re,
                     "D": 2 * r, "blockage": 2 * r / 98, "status": "steady"})
    return pd.DataFrame(rows), np.stack(masks), np.stack(fields)


def test_signed_distance_sign_and_scale():
    m = np.zeros((H, W), np.uint8)
    m[10:20, 20:30] = 1
    sdf = signed_distance(m)
    assert (sdf[m == 1] < 0).all() and (sdf[m == 0] > 0).all()
    assert sdf[15, 40] == pytest.approx(11.0)  # 11 cells right of the last solid column


def test_field_errors():
    df, masks, fields = _synthetic(4)
    true = fields.astype(np.float32)
    zero = field_errors(true, true, masks, wake_x0=16)
    assert np.allclose(zero.to_numpy(), 0.0)
    scaled = field_errors(true, 1.1 * true, masks, wake_x0=16)
    assert np.allclose(scaled["u_rel_l2"], 0.1, rtol=1e-5) and np.allclose(scaled["cp_rel_l2"], 0.1, rtol=1e-5)


def test_nearest_neighbour_returns_itself_on_training_rows():
    df, _, fields = _synthetic(6)
    pred, idx = nearest_neighbour_baseline(df, fields, df)
    assert np.array_equal(idx, np.arange(6)) and np.array_equal(pred, fields.astype(np.float32))


def test_unet_shapes():
    torch = pytest.importorskip("torch")
    from cfd_drag.surrogate.unet import FieldUNet
    model = FieldUNet(replace(FIELD_TRAIN, base_channels=4, levels=2))
    assert model(torch.zeros(2, 3, H, W)).shape == (2, 3, H, W)
    with pytest.raises(ValueError):
        model(torch.zeros(1, 3, 30, W))


def test_training_is_deterministic_and_reloads(tmp_path):
    torch = pytest.importorskip("torch")
    from cfd_drag.surrogate.field_train import load_field_model, predict_fields, train_fields
    df, masks, fields = _synthetic(12)
    params = replace(FIELD_TRAIN, base_channels=4, levels=2, epochs=2, batch_size=4)
    split = TrainParams(val_fraction=0.25, test_fraction=0.25)
    a = train_fields(df, masks, fields, tmp_path / "a", params, split, device="cpu", log=lambda *_: None)
    b = train_fields(df, masks, fields, tmp_path / "b", params, split, device="cpu", log=lambda *_: None)
    assert all(torch.equal(a["state_dict"][k], b["state_dict"][k]) for k in a["state_dict"])
    model, norm, p, _ = load_field_model(tmp_path / "a" / "model.pt")
    pred = predict_fields(model, norm, p, masks[:3], df["re"].to_numpy()[:3])
    assert pred.shape == (3, 3, H, W) and np.isfinite(pred).all()
    assert (pred[:, :, masks[0] == 1][0] == 0).all()  # obstacle cells are zeroed


def test_inputs_and_normaliser():
    df, masks, fields = _synthetic(5)
    norm = FieldNormaliser.fit(df, masks, fields)
    std = norm.to_std(fields)
    for c in range(3):
        v = std[:, c][masks == 0]
        assert abs(v.mean()) < 1e-3 and abs(v.std() - 1) < 1e-3
    assert np.allclose(norm.from_std(std), fields.astype(np.float32), atol=1e-5)
    x = make_inputs(masks, df["re"].to_numpy(), norm)
    assert x.shape == (5, 3, H, W) and np.array_equal(x[:, 0], masks)
    assert np.allclose(x[:, 2].mean(axis=(1, 2)), norm.re(df["re"].to_numpy()))
