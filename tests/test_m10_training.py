"""M10 training extras: mirror symmetry behind flip augmentation, Re weights, determinism."""
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from cfd_drag.surrogate.data import Normaliser, flip_y_std, re_weights
from solver_run import load, read_results, run_solver


@pytest.mark.parametrize("shape,s1,s2,angle,re", [("tri", 6.0, 9.0, 35.0, 30.0), ("rect", 4.0, 8.0, 20.0, 60.0)])
def test_mirrored_case_has_mirrored_mask_same_cd_opposite_cl(solver_exe, tmp_path, shape, s1, s2, angle, re):
    """Mirror about the channel centreline y = (ny - 1) / 2: obstacle centre ny/2 -> ny/2 - 1,
    angle -> -angle. Basis of the flip augmentation: crops about the centreline flip exactly."""
    ny = 40
    common = ["--width", "120", "--height", ny, "--shape", shape, "--size1", s1, "--size2", s2,
              "--re", re, "--warmup", "2000", "--avg", "800", "--dump"]
    a, b = tmp_path / "a", tmp_path / "b"
    a.mkdir(), b.mkdir()
    run_solver(solver_exe, a, *common, "--angle", angle)
    run_solver(solver_exe, b, *common, "--angle", -angle, "--cy", ny // 2 - 1)
    assert np.array_equal(load(a, "mask")[::-1], load(b, "mask"))
    ra, rb = read_results(a), read_results(b)
    assert rb["Cd"] == pytest.approx(ra["Cd"], rel=1e-9)
    assert rb["Cl"] == pytest.approx(-ra["Cl"], rel=1e-9, abs=1e-12)


def test_re_weights_favour_rare_re_and_have_mean_one():
    re = np.concatenate([np.full(90, 30.0), np.full(10, 90.0)])
    w = re_weights(re, bins=4, w_max=100.0)
    assert w.mean() == pytest.approx(1.0)
    assert w[-1] / w[0] == pytest.approx(9.0)
    capped = re_weights(re, bins=4, w_max=2.0)
    assert capped.mean() == pytest.approx(1.0) and capped[-1] / capped[0] < 9.0


def test_flip_y_std_negates_physical_value():
    df = pd.DataFrame({"re": [20.0, 50, 80], "Cd": [2.0, 1.5, 1.2], "Cl": [0.3, -0.1, 0.05]})
    norm = Normaliser.fit(df)
    neg = df.assign(Cl=-df["Cl"])
    assert np.allclose(flip_y_std(norm.y(df)[:, 1], norm.y_mean[1], norm.y_std[1]), norm.y(neg)[:, 1])


def test_cnn_training_with_flip_and_weights_is_deterministic(tmp_path):
    torch = pytest.importorskip("torch")
    from cfd_drag.dataset import load as load_dataset
    from cfd_drag.params import TRAIN
    from cfd_drag.paths import REPO_ROOT
    from cfd_drag.surrogate.train import train
    df, masks = load_dataset(REPO_ROOT / "example_data")
    p = replace(TRAIN, channels=(8, 16), hidden=16, epochs=2, batch_size=8, augment_flip=True, re_weight_bins=4)
    a = train(df, masks, tmp_path / "a", p, "cpu", log=lambda *_: None)
    b = train(df, masks, tmp_path / "b", p, "cpu", log=lambda *_: None)
    assert all(torch.equal(a["state_dict"][k], b["state_dict"][k]) for k in a["state_dict"])
    c = train(df, masks, tmp_path / "c", replace(p, augment_flip=False), "cpu", log=lambda *_: None)
    assert not all(torch.equal(a["state_dict"][k], c["state_dict"][k]) for k in a["state_dict"])


def test_unet_training_with_flip_is_deterministic(tmp_path):
    torch = pytest.importorskip("torch")
    from cfd_drag.params import FIELD_TRAIN, TrainParams
    from cfd_drag.surrogate.field_train import train_fields
    from test_field_surrogate import _synthetic
    df, masks, fields = _synthetic(12)
    params = replace(FIELD_TRAIN, base_channels=4, levels=2, epochs=2, batch_size=4, augment_flip=True)
    split = TrainParams(val_fraction=0.25, test_fraction=0.25)
    a = train_fields(df, masks, fields, tmp_path / "a", params, split, device="cpu", log=lambda *_: None)
    b = train_fields(df, masks, fields, tmp_path / "b", params, split, device="cpu", log=lambda *_: None)
    assert all(torch.equal(a["state_dict"][k], b["state_dict"][k]) for k in a["state_dict"])
