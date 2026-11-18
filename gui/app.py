"""LBM Flow Studio: design an obstacle, get an instant CNN prediction, verify it with the solver.

Run:  python gui/app.py
      python gui/app.py --screenshot docs/gui.png   (runs a demo case, saves a screenshot, exits)
"""
import html
import math
import os
import sys
import tempfile
import time
from dataclasses import replace
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from cfd_drag.dataset import collect_case, field_window, generate  # noqa: E402
from cfd_drag.geometry import vertices  # noqa: E402
from cfd_drag.params import DOMAIN, FIELD, PHYSICS, SAMPLING  # noqa: E402
from cfd_drag.paths import SOLVER_EXE  # noqa: E402
from cfd_drag.predict import (FIELD_MODEL_PATH, MODEL_PATH, case_info, obstacle_mask,  # noqa: E402
                              solver_args, solver_field)
from cfd_drag.sampling import Case  # noqa: E402

from PyQt6.QtWidgets import (  # noqa: E402
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QGridLayout, QLabel, QPushButton, QComboBox, QDoubleSpinBox,
    QGroupBox, QFileDialog, QMessageBox, QRadioButton, QSpinBox, QCheckBox,
    QGraphicsView, QGraphicsScene, QGraphicsPixmapItem
)
from PyQt6.QtCore import Qt, QProcess, QThread, QTimer, pyqtSignal, QRectF, QPointF  # noqa: E402
from PyQt6.QtGui import (  # noqa: E402
    QPixmap, QPainter, QColor, QPen, QBrush, QPolygonF, QCursor, QWheelEvent, QImage
)

SHAPES = {"Circle": "circle", "Rectangle": "rect", "Triangle": "tri"}
SCALE = 2.0  # screen pixels per lattice cell
DEFAULT_WORKERS = max(1, (os.cpu_count() or 2) // 2)  # physical cores, as in generate.py

# Field view: (label, colour range, colormap). Fixed ranges so U-Net and solver views compare 1:1.
VIEWS = ["Shape preview", "U-Net prediction", "Solver result", "Difference (U-Net - solver)"]
QUANTITIES = {"Speed |u|/U": ((0.0, 1.8), "viridis"), "ux/U": ((-0.4, 1.8), "viridis"),
              "uy/U": ((-1.0, 1.0), "RdBu_r"), "Cp": ((-2.5, 2.5), "RdBu_r")}
DIFF_RANGE = {"Speed |u|/U": 0.3, "ux/U": 0.3, "uy/U": 0.3, "Cp": 0.5}


def quantity(field, name):
    """One scalar map from a (3, H, W) field (ux / U, uy / U, Cp)."""
    if name == "Speed |u|/U":
        return np.hypot(field[0], field[1])
    return field[{"ux/U": 0, "uy/U": 1, "Cp": 2}[name]]


def colorize(values, lo, hi, cmap):
    """float map -> uint8 RGB with a matplotlib colormap (grey if matplotlib is missing)."""
    t = np.clip((values - lo) / (hi - lo), 0.0, 1.0)
    try:
        import matplotlib
        rgb = matplotlib.colormaps[cmap](t)[..., :3]
    except ImportError:
        rgb = np.repeat(t[..., None], 3, axis=2)
    return (rgb * 255).astype(np.uint8)


def field_errors_simple(pred, true, solid):
    """(velocity, Cp) relative L2 errors on fluid cells."""
    fl = ~solid
    du = np.sqrt(((pred[:2, fl] - true[:2, fl]) ** 2).sum() / (true[:2, fl] ** 2).sum())
    dp = np.sqrt(((pred[2, fl] - true[2, fl]) ** 2).sum() / max((true[2, fl] ** 2).sum(), 1e-12))
    return float(du), float(dp)


# ==========================================
# Worker thread for dataset generation
# ==========================================
class BatchWorker(QThread):
    progress = pyqtSignal(int, int)  # done, total
    finished_ok = pyqtSignal(str)
    error = pyqtSignal(str)

    def __init__(self, out_dir, n, seed, workers, shapes, images):
        super().__init__()
        self.out_dir, self.n, self.seed, self.workers = Path(out_dir), n, seed, workers
        self.shapes, self.images = tuple(shapes), images

    def run(self):
        try:
            df = generate(self.out_dir, self.n, self.seed, self.workers, DOMAIN, PHYSICS,
                          replace(SAMPLING, shapes=self.shapes), images=self.images, progress=False,
                          callback=lambda done, total: self.progress.emit(done, total))
            self.finished_ok.emit(f"{int(df['ok'].sum())} of {len(df)} cases solved.\n"
                                  f"Saved data.csv and masks.npy in {self.out_dir}")
        except Exception as e:  # shown to the user
            self.error.emit(str(e))


# ==========================================
# Custom Circular Progress Widget
# ==========================================
class CircularProgress(QWidget):
    def __init__(self):
        super().__init__()
        self.val = 0
        self.max_val = 100
        # Fix: Base initialization is clean, removing the hardcoded visual duplication
        self.step_text = "0/0 steps" 
        self.setMinimumSize(140, 160)

    def update_progress(self, val, max_val, text):
        self.val = val
        self.max_val = max_val
        self.step_text = text
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        rect = self.rect()

        size = 80 
        x = int((rect.width() - size) / 2)
        y = int((rect.height() - size) / 2) - 25
        circle_rect = QRectF(x, y, size, size)

        # Background circle
        pen = QPen(QColor("#1e1e2e"), 8)
        pen.setCapStyle(Qt.PenCapStyle.RoundCap)
        painter.setPen(pen)
        painter.drawArc(circle_rect, 0, 360 * 16)

        # Active progress
        if self.max_val > 0:
            span_angle = int((self.val / self.max_val) * 360 * 16)
            pen.setColor(QColor("#00ffff"))
            painter.setPen(pen)
            painter.drawArc(circle_rect, 90 * 16, -span_angle)

        # Text below the circle
        painter.setPen(QColor("#a6adc8"))
        font = painter.font()
        font.setPointSize(9)
        painter.setFont(font)
        
        pct = int((self.val / self.max_val) * 100) if self.max_val > 0 else 0
        text_y = y + size + 15
        
        # Draw the percentage and the dynamic text cleanly
        painter.drawText(QRectF(0, text_y, rect.width(), 20), Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignTop, f"{pct} %")
        painter.drawText(QRectF(0, text_y + 20, rect.width(), 30), Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignTop, self.step_text)
        
        painter.end()

# ==========================================
# Interactive Graphics View for Zoom & Pan
# ==========================================
class InteractiveGraphicsView(QGraphicsView):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.scene = QGraphicsScene(self)
        self.setScene(self.scene)
        self.pixmap_item = QGraphicsPixmapItem()
        self.scene.addItem(self.pixmap_item)
        
        # Setup interactive panning
        self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        
        self.setStyleSheet("background-color: #1e1e2e; border: 1px solid #313244; border-radius: 8px;")

    def setPixmap(self, pixmap):
        self.pixmap_item.setPixmap(pixmap)
        self.scene.setSceneRect(self.pixmap_item.boundingRect())
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)

    def reset_view(self):
        self.resetTransform()

    def wheelEvent(self, event: QWheelEvent):
        # Zoom Factors
        zoom_in_factor = 1.15
        zoom_out_factor = 1.0 / zoom_in_factor

        # Set Anchors so it zooms exactly where the mouse is
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)

        # Zoom in or out
        if event.angleDelta().y() > 0:
            zoom_factor = zoom_in_factor
        else:
            zoom_factor = zoom_out_factor
            
        self.scale(zoom_factor, zoom_factor)

# ==========================================
# ==========================================
# Main GUI Application
# ==========================================
class LBMSolverGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("LBM Flow Studio: CFD solver + CNN surrogate")
        self.resize(1280, 760)
        self.executable = str(SOLVER_EXE)
        self.process = None
        self.run_dir = None
        self.batch_worker = None
        self.cnn_pred = None      # (Cd, Cl) for the current inputs
        self.unet_field = None    # (3, H, W) U-Net field for the current inputs
        self.field_solid = None   # (H, W) obstacle cells in the field window
        self.solver_fields = None  # (3, H, W) mean field of the last solver run (solved_inputs)
        self.solved_inputs = None  # inputs of the last solver run
        self.surrogate, self.surrogate_error = None, ""
        try:
            from cfd_drag.predict import Surrogate
            self.surrogate = Surrogate(MODEL_PATH)
        except Exception as e:  # torch missing or no model file: GUI still works without the CNN
            self.surrogate_error = f"{type(e).__name__}: {e}"
        self.field_surrogate, self.field_error = None, ""
        try:
            from cfd_drag.predict import FieldSurrogate
            self.field_surrogate = FieldSurrogate(FIELD_MODEL_PATH)
        except Exception as e:  # no U-Net weights (scripts/fetch_field_model.py): no field prediction
            self.field_error = f"{type(e).__name__}: {e}"

        self.update_timer = QTimer(self)
        self.update_timer.setSingleShot(True)
        self.update_timer.setInterval(150)
        self.update_timer.timeout.connect(self.update_case)

        self.init_ui()
        self.apply_dark_theme()
        self.update_case()

    # ---------- layout ----------
    def init_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QVBoxLayout(central)
        main_layout.setContentsMargins(20, 20, 20, 20)
        main_layout.setSpacing(16)

        top = QHBoxLayout()
        top.setSpacing(16)
        left = QVBoxLayout()
        view_row = QHBoxLayout()
        self.view_combo = QComboBox()
        self.view_combo.addItems(VIEWS)
        self.qty_combo = QComboBox()
        self.qty_combo.addItems(list(QUANTITIES))
        for label, combo in [("View:", self.view_combo), ("Quantity:", self.qty_combo)]:
            view_row.addWidget(QLabel(label))
            view_row.addWidget(combo)
            combo.currentIndexChanged.connect(lambda *_: self.refresh_canvas())
        view_row.addStretch(1)
        left.addLayout(view_row)
        self.canvas = InteractiveGraphicsView()
        self.canvas.setMinimumSize(820, 230)
        left.addWidget(self.canvas, stretch=1)
        self.canvas_caption = QLabel("")
        self.canvas_caption.setStyleSheet("color: #a6adc8; font-size: 12px;")
        left.addWidget(self.canvas_caption)
        top.addLayout(left, stretch=4)

        right = QVBoxLayout()
        right.setSpacing(12)

        cnn_group = QGroupBox("CNN prediction (instant)")
        cnn_group.setProperty("header", "blue")
        cnn = QGridLayout(cnn_group)
        self.mask_label = QLabel()
        self.mask_label.setFixedSize(96, 96)
        self.mask_label.setToolTip("CNN input: 64x64 obstacle mask")
        self.cnn_cd = QLabel("C<sub>d</sub> --")
        self.cnn_cl = QLabel("C<sub>l</sub> --")
        self.cnn_time = QLabel("")
        for w, color, size in [(self.cnn_cd, "#a6e3a1", 20), (self.cnn_cl, "#f9e2af", 16),
                               (self.cnn_time, "#a6adc8", 11)]:
            w.setTextFormat(Qt.TextFormat.RichText)
            w.setStyleSheet(f"color: {color}; font-size: {size}px; font-weight: bold;")
        cnn.addWidget(self.mask_label, 0, 0, 3, 1)
        cnn.addWidget(self.cnn_cd, 0, 1)
        cnn.addWidget(self.cnn_cl, 1, 1)
        cnn.addWidget(self.cnn_time, 2, 1)
        right.addWidget(cnn_group)

        solver_group = QGroupBox("LBM solver (ground truth)")
        solver_group.setProperty("header", "blue")
        sol = QGridLayout(solver_group)
        self.sol_cd = QLabel("C<sub>d</sub> --")
        self.sol_cl = QLabel("C<sub>l</sub> --")
        self.sol_info = QLabel("Press RUN SOLVER to verify")
        self.sol_err = QLabel("")
        for w, color, size in [(self.sol_cd, "#a6e3a1", 20), (self.sol_cl, "#f9e2af", 16),
                               (self.sol_info, "#a6adc8", 11), (self.sol_err, "#89dceb", 12)]:
            w.setTextFormat(Qt.TextFormat.RichText)
            w.setStyleSheet(f"color: {color}; font-size: {size}px; font-weight: bold;")
        sol.addWidget(self.sol_cd, 0, 0)
        sol.addWidget(self.sol_cl, 0, 1)
        sol.addWidget(self.sol_info, 1, 0, 1, 2)
        sol.addWidget(self.sol_err, 2, 0, 1, 2)
        right.addWidget(solver_group)

        progress_group = QGroupBox("progress")
        pl = QVBoxLayout(progress_group)
        self.progress_widget = CircularProgress()
        pl.addWidget(self.progress_widget)
        right.addWidget(progress_group, stretch=1)
        top.addLayout(right, stretch=2)
        main_layout.addLayout(top, stretch=3)

        bottom = QHBoxLayout()
        bottom.setSpacing(14)

        shape_group = QGroupBox("Geometry [lattice cells]")
        shape_group.setProperty("header", "blue")
        sl = QVBoxLayout(shape_group)
        self.shape_combo = QComboBox()
        self.shape_combo.addItems(list(SHAPES))
        self.shape_combo.setCurrentText("Rectangle")
        self.shape_combo.currentTextChanged.connect(self.inputs_changed)
        sl.addWidget(self.shape_combo)
        self.size1_label, self.size1_spin = self.create_control_row("", 1.0, 60.0, 8.0, sl, 0.5)
        self.size2_label, self.size2_spin = self.create_control_row("", 1.0, 60.0, 16.0, sl, 0.5)
        _, self.angle_spin = self.create_control_row("Angle (°):", 0.0, 180.0, 30.0, sl, 5.0)
        bottom.addWidget(shape_group)

        flow_group = QGroupBox("Flow")
        flow_group.setProperty("header", "blue")
        fl = QVBoxLayout(flow_group)
        _, self.re_spin = self.create_control_row("Reynolds number:", 1.0, 200.0, 60.0, fl, 5.0)
        self.case_label = QLabel("")
        self.case_label.setWordWrap(True)
        self.case_label.setTextFormat(Qt.TextFormat.RichText)
        self.case_label.setStyleSheet("font-size: 12px;")
        fl.addWidget(self.case_label)
        fl.addStretch()
        bottom.addWidget(flow_group, stretch=1)

        batch_group = QGroupBox("Mode")
        bl = QGridLayout(batch_group)
        self.radio_single = QRadioButton("Single case")
        self.radio_single.setChecked(True)
        self.radio_batch = QRadioButton("Generate dataset:")
        self.batch_count_spin = self.int_spin(1, 100000, 50)
        bl.addWidget(self.radio_single, 0, 0, 1, 2)
        bl.addWidget(self.radio_batch, 1, 0)
        bl.addWidget(self.batch_count_spin, 1, 1)
        bl.setRowStretch(2, 1)
        bottom.addWidget(batch_group)

        self.tuning_group = QGroupBox("Dataset settings")
        self.tuning_group.setEnabled(False)
        tl = QGridLayout(self.tuning_group)
        self.seed_spin = self.int_spin(0, 10**6, 0)
        self.workers_spin = self.int_spin(1, 64, DEFAULT_WORKERS)
        self.tune_shape = {}
        shapes_row = QHBoxLayout()
        for name in SHAPES:
            cb = QCheckBox(name)
            cb.setChecked(True)
            cb.setCursor(QCursor(Qt.CursorShape.PointingHandCursor))
            shapes_row.addWidget(cb)
            self.tune_shape[name] = cb
        self.images_check = QCheckBox("Save flow images")
        tl.addWidget(QLabel("Seed:"), 0, 0); tl.addWidget(self.seed_spin, 0, 1)
        tl.addWidget(QLabel("Workers:"), 0, 2); tl.addWidget(self.workers_spin, 0, 3)
        tl.addWidget(QLabel("Shapes:"), 1, 0); tl.addLayout(shapes_row, 1, 1, 1, 3)
        tl.addWidget(self.images_check, 2, 0, 1, 4)
        bottom.addWidget(self.tuning_group)
        self.radio_single.toggled.connect(self.toggle_modes)

        self.sim_btn = QPushButton("RUN SOLVER")
        self.sim_btn.setMinimumSize(190, 60)
        font = self.sim_btn.font()
        font.setPointSize(12)
        font.setBold(True)
        self.sim_btn.setFont(font)
        self.sim_btn.clicked.connect(self.start_dispatcher)
        self.sim_btn.setCursor(QCursor(Qt.CursorShape.PointingHandCursor))
        bottom.addWidget(self.sim_btn, alignment=Qt.AlignmentFlag.AlignBottom | Qt.AlignmentFlag.AlignRight)
        main_layout.addLayout(bottom, stretch=1)

    def create_control_row(self, label_text, min_val, max_val, default_val, layout, step=1.0):
        row = QWidget()
        rl = QHBoxLayout(row)
        rl.setContentsMargins(0, 0, 0, 0)
        label = QLabel(label_text)
        spin = QDoubleSpinBox()
        spin.setRange(min_val, max_val)
        spin.setValue(default_val)
        spin.setSingleStep(step)
        spin.valueChanged.connect(self.inputs_changed)
        rl.addWidget(label, stretch=1)
        rl.addWidget(spin, stretch=1)
        layout.addWidget(row)
        return label, spin

    @staticmethod
    def int_spin(lo, hi, value):
        s = QSpinBox()
        s.setRange(lo, hi)
        s.setValue(value)
        s.setMinimumWidth(70)
        return s

    def toggle_modes(self):
        batch = self.radio_batch.isChecked()
        self.tuning_group.setEnabled(batch)
        self.sim_btn.setText("GENERATE" if batch else "RUN SOLVER")
        self.update_case()

    # ---------- current inputs ----------
    def inputs(self):
        shape = SHAPES[self.shape_combo.currentText()]
        s1, s2, angle = self.size1_spin.value(), self.size2_spin.value(), self.angle_spin.value()
        if shape == "circle":
            s2, angle = s1, 0.0
        return shape, s1, s2, angle, self.re_spin.value()

    def inputs_changed(self, *_):
        self.update_timer.start()  # debounce: recompute 150 ms after the last change

    # ---------- live case info + CNN ----------
    def update_case(self):
        shape, s1, s2, angle, re = self.inputs()
        circle = shape == "circle"
        self.size1_label.setText("Radius:" if circle else ("Width:" if shape == "rect" else "Base:"))
        self.size2_label.setText("Height:" if shape == "rect" else "Length:")
        self.size2_spin.setEnabled(not circle)
        self.angle_spin.setEnabled(not circle)

        info = case_info(shape, s1, s2, angle, re)
        lines = [f"D = {info.D:.1f} cells · blockage {info.blockage:.0%} · τ = {info.tau:.3f}"]
        lines += [f"<span style='color:#f38ba8'>✖ {html.escape(p)}</span>" for p in info.problems]
        lines += [f"<span style='color:#fab387'>⚠ {html.escape(w)}</span>" for w in info.warnings]
        if info.solvable and not info.warnings:
            lines.append("<span style='color:#a6e3a1'>✔ inside the training range</span>")
        self.case_label.setText("<br>".join(lines))
        if not self.is_busy():
            self.sim_btn.setEnabled(info.solvable or self.radio_batch.isChecked())

        mask = None
        self.cnn_pred = None
        self.unet_field = None
        if info.blockage <= PHYSICS.max_blockage and Path(self.executable).exists():
            try:
                t = time.perf_counter()
                mask = obstacle_mask(shape, s1, s2, angle, re, exe=Path(self.executable))
                t_mask = time.perf_counter() - t
                if self.surrogate is not None:
                    t = time.perf_counter()
                    self.cnn_pred = self.surrogate.predict(mask, re)
                    t_cnn = time.perf_counter() - t
                if self.field_surrogate is not None:
                    t = time.perf_counter()
                    self.unet_field = self.field_surrogate.predict(mask, re)
                    self.field_solid = self.field_surrogate.window_mask(mask).astype(bool)
                    t_unet = time.perf_counter() - t
            except Exception as e:
                self.cnn_time.setText(f"mask error: {e}"[:80])
        self.show_mask(mask)
        if self.cnn_pred:
            cd, cl = self.cnn_pred
            self.cnn_cd.setText(f"C<sub>d</sub> = {cd:.3f}")
            self.cnn_cl.setText(f"C<sub>l</sub> = {cl:+.3f}")
            note = " · outside training range" if info.warnings else ""
            unet = f" · U-Net field {t_unet * 1000:.0f} ms" if self.unet_field is not None else ""
            self.cnn_time.setText(f"{t_cnn * 1000:.0f} ms (+{t_mask * 1000:.0f} ms mask){unet}{note}")
        else:
            self.cnn_cd.setText("C<sub>d</sub> --")
            self.cnn_cl.setText("C<sub>l</sub> --")
            if self.surrogate is None:
                self.cnn_time.setText("CNN unavailable (pip install -e \".[ml]\")")
                self.cnn_time.setToolTip(self.surrogate_error)
            elif mask is None:
                self.cnn_time.setText("no prediction for this shape")

        if self.solved_inputs is not None and self.solved_inputs == self.inputs():
            self.update_comparison()
        elif not self.is_busy():
            self.clear_solver_panel()
        self.refresh_canvas()

    # ---------- canvas: shape preview or flow fields ----------
    def refresh_canvas(self):
        view, qty = self.view_combo.currentText(), self.qty_combo.currentText()
        shape, s1, s2, angle, re = self.inputs()
        solved = self.solver_fields if self.solved_inputs == self.inputs() else None
        unet = self.unet_field
        if view == VIEWS[1] and unet is not None:
            self.show_field(quantity(unet, qty), *QUANTITIES[qty], f"U-Net prediction (time-averaged {qty}), "
                            f"{shape}, Re = {re:g}")
        elif view == VIEWS[2] and solved is not None:
            self.show_field(quantity(solved, qty), *QUANTITIES[qty], f"LBM solver (time-averaged {qty}), "
                            f"{shape}, Re = {re:g}")
        elif view == VIEWS[3] and solved is not None and unet is not None:
            d = DIFF_RANGE[qty]
            self.show_field(quantity(unet, qty) - quantity(solved, qty), (-d, d), "RdBu_r",
                            f"U-Net minus solver ({qty}), {shape}, Re = {re:g}")
        else:
            self.draw_preview(shape, s1, s2, angle)
            if view == VIEWS[1]:
                reason = ("U-Net weights missing: python scripts/fetch_field_model.py"
                          if self.field_surrogate is None else "no prediction for this shape")
                self.canvas_caption.setText(f"{reason}. Showing the shape preview.")
            elif view != VIEWS[0]:
                self.canvas_caption.setText("Press RUN SOLVER to compute the solver field. Showing the shape preview.")

    def show_field(self, values, rng, cmap, caption):
        """Draw a field-window map into the channel picture (obstacle in black)."""
        rows, cols = field_window(DOMAIN, FIELD)
        img = np.zeros((DOMAIN.ny, DOMAIN.nx, 3), np.uint8)
        img[:] = (30, 30, 46)
        img[[0, -1]] = (137, 180, 250)  # channel walls
        rgb = colorize(values, rng[0], rng[1], cmap)
        if self.field_solid is not None:
            rgb[self.field_solid] = (17, 17, 27)
        img[rows, cols] = rgb
        img = np.ascontiguousarray(np.flipud(img))  # lattice y points up
        qimg = QImage(img.data, img.shape[1], img.shape[0], 3 * img.shape[1], QImage.Format.Format_RGB888)
        pix = QPixmap.fromImage(qimg.copy()).scaled(int(DOMAIN.nx * SCALE), int(DOMAIN.ny * SCALE),
                                                     Qt.AspectRatioMode.IgnoreAspectRatio,
                                                     Qt.TransformationMode.SmoothTransformation)
        self.canvas.reset_view()
        self.canvas.setPixmap(pix)
        self.canvas_caption.setText(f"{caption}. Colour range {rng[0]:g} to {rng[1]:g}; "
                                    f"window {FIELD.width} x {FIELD.height} cells. Scroll to zoom, drag to pan.")

    def show_mask(self, mask):
        if mask is None:
            self.mask_label.clear()
            return
        img = np.flipud(mask).astype(np.uint8) * 255  # row 0 = bottom in lattice y
        img = np.ascontiguousarray(np.where(img > 0, 230, 40).astype(np.uint8))
        qimg = QImage(img.data, img.shape[1], img.shape[0], img.shape[1], QImage.Format.Format_Grayscale8)
        self.mask_label.setPixmap(QPixmap.fromImage(qimg.copy()).scaled(96, 96))

    def draw_preview(self, shape, s1, s2, angle):
        w, h = int(DOMAIN.nx * SCALE), int(DOMAIN.ny * SCALE)
        pixmap = QPixmap(w, h)
        pixmap.fill(QColor("#1e1e2e"))
        p = QPainter(pixmap)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        p.setPen(QPen(QColor("#89b4fa"), 3))
        p.drawLine(0, 1, w, 1)
        p.drawLine(0, h - 2, w, h - 2)
        p.setPen(QPen(QColor("#45475a"), 2))
        for y in range(20, h, 30):  # inlet arrows
            p.drawLine(10, y, 40, y)
            p.drawLine(30, y - 5, 40, y)
            p.drawLine(30, y + 5, 40, y)
        cx, cy = DOMAIN.centre
        X = lambda x: x * SCALE
        Y = lambda y: h - y * SCALE  # lattice y points up
        half = SAMPLING.crop / 2
        p.setPen(QPen(QColor("#585b70"), 1, Qt.PenStyle.DashLine))
        p.setBrush(Qt.BrushStyle.NoBrush)
        p.drawRect(QRectF(X(cx - half), Y(cy + half), SAMPLING.crop * SCALE, SAMPLING.crop * SCALE))
        p.setBrush(QBrush(QColor("#f38ba8")))
        p.setPen(QPen(QColor("#11111b"), 2))
        if shape == "circle":
            p.drawEllipse(QPointF(X(cx), Y(cy)), s1 * SCALE, s1 * SCALE)
        else:
            t = math.radians(angle)
            pts = [QPointF(X(cx + lx * math.cos(t) - ly * math.sin(t)),
                           Y(cy + lx * math.sin(t) + ly * math.cos(t))) for lx, ly in vertices(shape, s1, s2)]
            p.drawPolygon(QPolygonF(pts))
        p.end()
        self.canvas.reset_view()
        self.canvas.setPixmap(pixmap)
        self.canvas_caption.setText(f"Preview: {DOMAIN.nx} x {DOMAIN.ny} channel, flow left to right. "
                                    f"Dashed square = CNN input window ({SAMPLING.crop} x {SAMPLING.crop}).")

    # ---------- solver run ----------
    def is_busy(self):
        return self.process is not None or self.batch_worker is not None

    def clear_solver_panel(self):
        self.sol_cd.setText("C<sub>d</sub> --")
        self.sol_cl.setText("C<sub>l</sub> --")
        self.sol_info.setText("Press RUN SOLVER to verify")
        self.sol_err.setText("")

    def start_dispatcher(self):
        if self.radio_single.isChecked():
            self.start_simulation()
        else:
            self.run_batch_mode()

    def start_simulation(self):
        if not Path(self.executable).exists():
            QMessageBox.critical(self, "Error", f"Solver '{self.executable}' not found. Run `make solver`.")
            return
        shape, s1, s2, angle, re = self.inputs()
        if not case_info(shape, s1, s2, angle, re).solvable:
            return
        self.run_inputs = (shape, s1, s2, angle, re)
        self.run_dir = tempfile.TemporaryDirectory(prefix="cfd_gui_")
        self.total_steps = PHYSICS.warmup_steps(DOMAIN.nx) + PHYSICS.avg_steps(DOMAIN.nx)
        self.sim_btn.setEnabled(False)
        self.sim_btn.setText("SOLVING...")
        self.sol_info.setText("solving...")
        self.progress_widget.update_progress(0, self.total_steps, f"0/{self.total_steps} steps")
        self.t_start = time.perf_counter()
        self.process = QProcess(self)
        self.process.readyReadStandardOutput.connect(self.handle_stdout)
        self.process.finished.connect(self.simulation_finished)
        self.process.start(self.executable, ["--out-dir", self.run_dir.name,
                                             *solver_args(shape, s1, s2, angle, re, mean_fields=True)])

    def handle_stdout(self):
        out = self.process.readAllStandardOutput().data().decode(errors="replace")
        for line in out.replace("\r", "\n").split("\n"):
            parts = line.split()
            if len(parts) >= 4 and parts[0] == "Step" and parts[2] == "/":
                try:
                    step, total = int(parts[1]), int(parts[3])
                    self.progress_widget.update_progress(step, total, f"{step}/{total} steps")
                except ValueError:
                    pass

    def simulation_finished(self):
        seconds = time.perf_counter() - self.t_start
        code = self.process.exitCode()
        crashed = self.process.exitStatus() != QProcess.ExitStatus.NormalExit
        self.process = None
        self.sim_btn.setText("RUN SOLVER")
        self.sim_btn.setEnabled(True)
        self.progress_widget.update_progress(1, 1, f"done in {seconds:.0f} s")
        shape, s1, s2, angle, re = self.run_inputs
        case = Case(0, shape, s1, s2, angle, re, case_info(shape, s1, s2, angle, re).D)
        run_dir = Path(self.run_dir.name)
        error = "crashed" if crashed else ("" if code in (0, 3) else f"exit {code}")
        row, _, _ = collect_case(case, run_dir, error, DOMAIN, SAMPLING, None)
        self.solver_fields = solver_field(run_dir) if row["ok"] and (run_dir / "fields_mean.npy").exists() else None
        self.run_dir.cleanup()
        self.run_dir = None
        self.last_row = row
        if row["ok"]:
            self.solved_inputs = self.run_inputs
            if self.view_combo.currentText() == VIEWS[0]:
                self.view_combo.setCurrentText(VIEWS[2])
            self.sol_cd.setText(f"C<sub>d</sub> = {row['Cd']:.3f}")
            self.sol_cl.setText(f"C<sub>l</sub> = {row['Cl']:+.3f}")
            regime = "vortex shedding" if row["status"] == "periodic" else "steady"
            self.sol_info.setText(f"{regime} · C<sub>l,rms</sub> = {row['Cl_rms']:.3f} · {seconds:.0f} s")
            self.update_comparison()
        else:
            self.solved_inputs = None
            self.sol_info.setText(f"<span style='color:#f38ba8'>failed: {html.escape(row['error'])}</span>")
        self.update_case()

    def update_comparison(self):
        row = self.last_row
        if not self.cnn_pred or not row["ok"]:
            self.sol_err.setText("")
            return
        cd, cl = self.cnn_pred
        err = abs(cd - row["Cd"]) / abs(row["Cd"])
        text = f"CNN error: C<sub>d</sub> {err:.1%} · C<sub>l</sub> {abs(cl - row['Cl']):.3f} (abs)"
        if self.unet_field is not None and self.solver_fields is not None:
            du, dp = field_errors_simple(self.unet_field, self.solver_fields, self.field_solid)
            text += f"<br>U-Net field error: velocity {du:.1%} · C<sub>p</sub> {dp:.1%}"
        self.sol_err.setText(text)

    # ---------- dataset generation ----------
    def run_batch_mode(self):
        if not Path(self.executable).exists():
            QMessageBox.critical(self, "Error", f"Solver '{self.executable}' not found. Run `make solver`.")
            return
        shapes = [SHAPES[name] for name, cb in self.tune_shape.items() if cb.isChecked()]
        if not shapes:
            QMessageBox.warning(self, "Selection required", "Select at least one shape.")
            return
        out = QFileDialog.getExistingDirectory(self, "Output folder (re-use a folder to resume)")
        if not out:
            return
        n = self.batch_count_spin.value()
        self.sim_btn.setEnabled(False)
        self.progress_widget.update_progress(0, n, f"0/{n} cases")
        self.batch_worker = BatchWorker(out, n, self.seed_spin.value(), self.workers_spin.value(),
                                        shapes, self.images_check.isChecked())
        self.batch_worker.progress.connect(
            lambda d, t: self.progress_widget.update_progress(d, t, f"{d}/{t} cases"))
        self.batch_worker.finished_ok.connect(self.batch_finished)
        self.batch_worker.error.connect(self.batch_error)
        self.batch_worker.start()

    def batch_finished(self, msg):
        self.batch_cleanup()
        QMessageBox.information(self, "Dataset ready", msg)

    def batch_error(self, err):
        self.batch_cleanup()
        QMessageBox.critical(self, "Generation error", err)

    def batch_cleanup(self):
        self.batch_worker.wait()
        self.batch_worker = None
        self.sim_btn.setEnabled(True)
        self.progress_widget.update_progress(1, 1, "done")

    def apply_dark_theme(self):
        self.setStyleSheet("""
            QMainWindow, QWidget {
                background-color: #11111b;
                color: #cdd6f4;
            }
            QLabel {
                background-color: transparent;
                font-family: 'Segoe UI', Arial, sans-serif;
            }
            QGroupBox {
                background-color: transparent;
                border: 1px solid #313244;
                border-radius: 8px;
                margin-top: 18px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                subcontrol-position: top left;
                left: 15px;
                padding: 0 5px;
                color: #a6adc8;
            }
            QGroupBox[header="blue"]::title {
                color: #89b4fa;
                font-weight: bold;
                font-size: 14px;
            }
            QComboBox, QDoubleSpinBox, QSpinBox {
                background-color: #1e1e2e;
                color: #cdd6f4;
                border: 1px solid #313244;
                border-radius: 4px;
                padding: 5px;
            }
            QComboBox:hover, QDoubleSpinBox:hover, QSpinBox:hover {
                border: 1px solid #89b4fa;
            }
            QComboBox::drop-down {
                border-left: 1px solid #313244;
            }
            QPushButton {
                background-color: #89b4fa;
                color: #11111b;
                border: none;
                border-radius: 6px;
                padding: 10px;
            }
            QPushButton:hover {
                background-color: #b4befe;
            }
            QPushButton:disabled {
                background-color: #45475a;
                color: #a6adc8;
            }
            QRadioButton, QCheckBox {
                background-color: transparent;
                color: #cdd6f4;
                spacing: 8px;
            }
            QRadioButton::indicator {
                width: 14px;
                height: 14px;
                border-radius: 7px;
                border: 2px solid #585b70;
                background-color: #1e1e2e;
            }
            QRadioButton::indicator:checked {
                background-color: #89b4fa;
                border: 3px solid #11111b;
            }
            /* Explicit high-visibility CheckBox Style */
            QCheckBox::indicator {
                width: 16px;
                height: 16px;
                border-radius: 4px;
                border: 2px solid #585b70;
                background-color: #1e1e2e;
            }
            QCheckBox::indicator:hover {
                border: 2px solid #a6e3a1;
            }
            QCheckBox::indicator:checked {
                background-color: #a6e3a1;
                border: 2px solid #a6e3a1;
            }
            QCheckBox:disabled {
                color: #45475a;
            }
        """)


def take_screenshot(window, path, inputs=("Rectangle", 8.0, 16.0, 30.0, 60.0), view=None):
    """Run one demo case with the solver, then save a screenshot of the window (in `view`) and quit."""
    shape, s1, s2, angle, re = inputs
    window.shape_combo.setCurrentText(shape)
    window.size1_spin.setValue(s1)
    window.size2_spin.setValue(s2)
    window.angle_spin.setValue(angle)
    window.re_spin.setValue(re)
    window.update_case()
    window.start_simulation()

    def poll():
        if window.process is None:
            if view:
                window.view_combo.setCurrentText(view)
            QApplication.processEvents()
            window.grab().save(str(path))
            print(f"saved {path}")
            QApplication.quit()
        else:
            QTimer.singleShot(500, poll)
    QTimer.singleShot(500, poll)


def main():
    app = QApplication(sys.argv)
    window = LBMSolverGUI()
    window.show()
    if "--screenshot" in sys.argv:
        view = sys.argv[sys.argv.index("--view") + 1] if "--view" in sys.argv else None
        take_screenshot(window, Path(sys.argv[sys.argv.index("--screenshot") + 1]), view=view)
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
