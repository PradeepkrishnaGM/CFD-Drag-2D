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
