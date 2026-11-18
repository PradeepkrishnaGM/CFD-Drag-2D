"""Single-case prediction helpers and the GUI (headless, Qt offscreen platform)."""
import os

import numpy as np
import pytest

from cfd_drag.dataset import load
from cfd_drag.paths import REPO_ROOT
from cfd_drag.predict import MODEL_PATH, case_info, obstacle_mask

EXAMPLE = REPO_ROOT / "example_data"


def test_case_info_flags():
    ok = case_info("rect", 8, 16, 30, 60)
    assert ok.solvable and not ok.warnings
    assert case_info("circle", 5, 0, 0, 100).problems  # tau too low
    assert case_info("rect", 40, 30, 0, 50).problems   # blockage above the cap
    assert any("Re" in w for w in case_info("circle", 8, 0, 0, 150).warnings)
    assert case_info("circle", 8, 99, 45, 50).D == 16  # circles ignore size2 and angle


def test_mask_matches_dataset(solver_exe):
    df, masks = load(EXAMPLE)
    for i in range(4):
        r = df.iloc[i]
        assert np.array_equal(obstacle_mask(r["shape"], r.size1, r.size2, r.angle, r.re), masks[i])


def test_surrogate_predicts_example_cases(solver_exe):
    pytest.importorskip("torch")
    from cfd_drag.predict import Surrogate
    assert MODEL_PATH.exists()
    s = Surrogate()
    df, masks = load(EXAMPLE)
    pred = np.array([s.predict(masks[i], df.re[i]) for i in range(len(df))])
    # example cases are part of the training data, so this is a wiring check, not an accuracy claim
    rel = np.abs(pred[:, 0] - df.Cd) / df.Cd
    assert rel.mean() < 0.05


def test_gui_live_prediction_and_validation(solver_exe):
    pytest.importorskip("torch")
    pytest.importorskip("PyQt6")  # not installed on Kaggle
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PyQt6.QtWidgets import QApplication
    import sys
    sys.path.insert(0, str(REPO_ROOT / "gui"))
    import app as gui

    qapp = QApplication.instance() or QApplication([])
    w = gui.LBMSolverGUI()
    assert w.surrogate is not None
    w.update_case()
    assert w.cnn_pred is not None and "C<sub>d</sub> =" in w.cnn_cd.text()
    assert w.sim_btn.isEnabled()
    w.shape_combo.setCurrentText("Circle")
    w.size1_spin.setValue(20)  # D = 40 cells: blockage 41 % > 25 %
    w.update_case()
    assert not w.sim_btn.isEnabled() and w.cnn_pred is None
    assert "blockage" in w.case_label.text()
    w.size1_spin.setValue(5)
    w.re_spin.setValue(100)  # tau below the limit: the full message must survive HTML rendering
    w.update_case()
    assert "&lt; 0.52 (unstable)" in w.case_label.text()
    w.close()
    del qapp


def _tiny_field_model(path):
    """A small untrained U-Net checkpoint with the real window size (for wiring tests)."""
    torch = pytest.importorskip("torch")
    from dataclasses import asdict, replace

    from cfd_drag.params import FIELD_TRAIN, TRAIN
    from cfd_drag.surrogate.fields import FieldNormaliser
    from cfd_drag.surrogate.unet import FieldUNet, n_parameters
    params = replace(FIELD_TRAIN, base_channels=4)
    torch.manual_seed(0)
    model = FieldUNet(params)
    norm = FieldNormaliser([0.9, 0.0, 0.2], [0.4, 0.2, 0.6], 3.9, 0.45)
    torch.save({"state_dict": model.state_dict(), "params": asdict(params), "split_params": asdict(TRAIN),
                "normaliser": asdict(norm), "best_epoch": 0, "best_val_error": 1.0, "n_rows": 0,
                "n_parameters": n_parameters(model), "train_seconds": 0, "device": "cpu", "torch": ""}, path)


def test_field_surrogate_and_gui_views(solver_exe, tmp_path):
    pytest.importorskip("torch")
    pytest.importorskip("PyQt6")
    from cfd_drag.params import FIELD
    from cfd_drag.predict import FieldSurrogate
    _tiny_field_model(tmp_path / "unet.pt")
    fs = FieldSurrogate(tmp_path / "unet.pt")
    mask = obstacle_mask("rect", 8, 16, 30, 60)
    field = fs.predict(mask, 60)
    solid = fs.window_mask(mask).astype(bool)
    assert field.shape == (3, FIELD.height, FIELD.width) and np.isfinite(field).all()
    assert (field[:, solid] == 0).all() and solid.sum() == mask.sum()

    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PyQt6.QtWidgets import QApplication
    import sys
    sys.path.insert(0, str(REPO_ROOT / "gui"))
    import app as gui
    qapp = QApplication.instance() or QApplication([])
    w = gui.LBMSolverGUI()
    w.field_surrogate = fs
    w.update_case()
    assert w.unet_field is not None and "U-Net field" in w.cnn_time.text()
    w.view_combo.setCurrentText("U-Net prediction")
    assert w.canvas_caption.text().startswith("U-Net prediction")
    w.view_combo.setCurrentText("Solver result")  # no solver run yet: falls back to the preview
    assert "RUN SOLVER" in w.canvas_caption.text()
    w.solved_inputs, w.solver_fields = w.inputs(), field + 0.1
    w.view_combo.setCurrentText("Difference (U-Net - solver)")
    assert w.canvas_caption.text().startswith("U-Net minus solver")
    w.close()
    del qapp
