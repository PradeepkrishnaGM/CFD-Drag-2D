import sys
import os
import csv
import math
import random
import subprocess
from pathlib import Path
from PIL import Image

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
    QGridLayout, QLabel, QPushButton, QComboBox, QSlider, QDoubleSpinBox, 
    QGroupBox, QProgressBar, QStatusBar, QMenuBar, QFileDialog, 
    QInputDialog, QMessageBox, QRadioButton, QSpinBox, QCheckBox,
    QGraphicsView, QGraphicsScene, QGraphicsPixmapItem
)
from PyQt6.QtCore import Qt, QProcess, QThread, pyqtSignal, QRectF, QPointF
from PyQt6.QtGui import QPixmap, QPainter, QColor, QPen, QBrush, QAction, QPolygonF, QCursor, QWheelEvent

# ==========================================
# Worker Thread for Batch Automation
# ==========================================
class BatchWorker(QThread):
    progress = pyqtSignal(int, int) # current, total
    finished = pyqtSignal(str)
    error = pyqtSignal(str)

    def __init__(self, executable, output_dir, num_samples, tuning_params):
        super().__init__()
        self.executable = executable
        self.output_dir = Path(output_dir)
        self.num_samples = num_samples
        self.tuning = tuning_params
        self.is_running = True

    def generate_random_params(self, allowed_shapes, ny):
        """Replicates logic directly from the provided python automation script."""
        shape = random.choice(allowed_shapes)
        
        max_size = ny * 0.35 
        min_size = ny * 0.05
        
        size1 = random.uniform(min_size, max_size)
        
        if shape == "circle":
            size2 = size1
        else:
            size2 = random.uniform(min_size, max_size)
            
        angle = random.uniform(self.tuning['ang_min'], self.tuning['ang_max'])
        velocity = random.uniform(self.tuning['vel_min'], self.tuning['vel_max'])
        
        return {
            "shape": shape,
            "size1": round(size1, 3),
            "size2": round(size2, 3),
            "angle": round(angle, 2),
            "velocity": round(velocity, 4)
        }

    def run(self):
        nx, ny = 400, 100
        images_dir = self.output_dir / "images"
        master_csv = self.output_dir / "training_data.csv"

        try:
            images_dir.mkdir(parents=True, exist_ok=True)
            if not master_csv.exists():
                with open(master_csv, mode='w', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow([
                        "sample_id", "shape", "size1", "size2", 
                        "angle", "velocity", "Cd", "pressure_loss", "image_path"
                    ])

            successful_samples = 0
            while successful_samples < self.num_samples and self.is_running:
                # 1. Generate Parameters
                params = self.generate_random_params(self.tuning['shapes'], ny)

                # 2. Cleanup & Run
                for temp_file in ["results.csv", "flow_field.ppm"]:
                    if os.path.exists(temp_file): os.remove(temp_file)

                cmd = [
                    self.executable, "--width", str(nx), "--height", str(ny),
                    "--shape", params["shape"], "--size1", str(params["size1"]),
                    "--size2", str(params["size2"]), "--angle", str(params["angle"]),
                    "--velocity", str(params["velocity"]), "--steps", "15000"
                ]

                try:
                    subprocess.run(cmd, capture_output=True, text=True, timeout=60, check=True)
                except (subprocess.TimeoutExpired, subprocess.CalledProcessError):
                    continue

                if not os.path.exists("results.csv") or not os.path.exists("flow_field.ppm"):
                    continue

                # 3. Parse CSV
                try:
                    with open("results.csv", mode='r') as f:
                        reader = csv.DictReader(f)
                        row = next(reader)
                    Cd = float(row["Cd"])
                    dP = float(row["pressure_loss"])
                    if math.isnan(Cd) or math.isinf(Cd) or math.isnan(dP) or math.isinf(dP) or abs(Cd) > 100.0:
                        continue
                except Exception:
                    continue

                # 4. Process Image
                sample_id = f"{successful_samples:05d}"
                img_name = f"sample_{sample_id}.png"
                img_path = images_dir / img_name
                try:
                    with Image.open("flow_field.ppm") as img:
                        img.save(img_path, format="PNG")
                except Exception:
                    continue

                # 5. Save Record
                with open(master_csv, mode='a', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow([
                        sample_id, params["shape"], params["size1"], params["size2"],
                        params["angle"], params["velocity"], Cd, dP, str(img_path.as_posix())
                    ])

                successful_samples += 1
                self.progress.emit(successful_samples, self.num_samples)

            if self.is_running:
                self.finished.emit(f"Successfully generated {successful_samples} samples.")
                
        except Exception as e:
            self.error.emit(str(e))

    def stop(self):
        self.is_running = False

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
# Main GUI Application
# ==========================================
class LBMSolverGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("LBM Flow Studio - Advanced Dataset Generator")
        self.resize(1200, 650)
        
        # Domain configuration
        self.domain_nx = 400
        self.domain_ny = 100
        self.executable = "./solver.exe" if sys.platform == "win32" else "./solver"
        
        self.process = None
        self.batch_worker = None
        
        self.init_ui()
        self.apply_dark_theme()
        self.update_preview()

    def init_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        main_layout.setContentsMargins(20, 20, 20, 20)
        main_layout.setSpacing(20)

        # --- TOP SECTION ---
        top_layout = QHBoxLayout()
        top_layout.setSpacing(20)

        # Canvas (Now using InteractiveGraphicsView for Zoom/Pan)
        self.canvas = InteractiveGraphicsView()
        self.canvas.setMinimumSize(800, 300)
        top_layout.addWidget(self.canvas, stretch=4)

        # Results & Progress
        right_layout = QVBoxLayout()
        right_layout.setSpacing(15)

        # Result Box
        result_group = QGroupBox("result")
        result_layout = QGridLayout(result_group)
        result_layout.setSpacing(10)
        
        self.cd_label = QLabel("C<sub>d</sub>: --")
        self.cd_label.setTextFormat(Qt.TextFormat.RichText) # Force HTML parsing for subscripts
        self.dp_label = QLabel("ΔP: --")
        
        self.br_label = QLabel("Blockage: --")
        self.re_label = QLabel("Est. Re: --")
        
        self.cd_label.setStyleSheet("color: #a6e3a1; font-size: 18px; font-weight: bold;")
        self.dp_label.setStyleSheet("color: #f9e2af; font-size: 18px; font-weight: bold;")
        self.br_label.setStyleSheet("color: #89b4fa; font-size: 13px;")
        self.re_label.setStyleSheet("color: #cba6f7; font-size: 13px;")
        
        result_layout.addWidget(self.cd_label, 0, 0)
        result_layout.addWidget(self.dp_label, 0, 1)
        result_layout.addWidget(self.br_label, 1, 0, 1, 2)
        result_layout.addWidget(self.re_label, 2, 0, 1, 2)
        
        right_layout.addWidget(result_group, stretch=1)

        # Progress Box
        progress_group = QGroupBox("progress")
        progress_layout = QVBoxLayout(progress_group)
        self.progress_widget = CircularProgress()
        progress_layout.addWidget(self.progress_widget)
        right_layout.addWidget(progress_group, stretch=1)

        top_layout.addLayout(right_layout, stretch=1)
        main_layout.addLayout(top_layout, stretch=3)

        # --- BOTTOM SECTION ---
        bottom_layout = QHBoxLayout()
        bottom_layout.setSpacing(15)

        # Geometry Group
        shape_group = QGroupBox("Geometry")
        shape_group.setProperty("header", "blue")
        shape_layout = QVBoxLayout(shape_group)
        self.shape_combo = QComboBox()
        self.shape_combo.addItems(["Circle", "Rectangle", "Triangle"])
        self.shape_combo.currentTextChanged.connect(self.update_preview)
        shape_layout.addWidget(self.shape_combo)
        
        self.size1_spin = self.create_control_row("Size 1 (Rad/W):", 5.0, 50.0, 10.0, shape_layout)
        self.size2_spin = self.create_control_row("Size 2 (Height):", 5.0, 50.0, 10.0, shape_layout)
        self.angle_spin = self.create_control_row("Angle (°):", 0.0, 180.0, 0.0, shape_layout)
        bottom_layout.addWidget(shape_group)

        # Fluid Dynamics Group
        fluid_group = QGroupBox("Fluid Dynamics")
        fluid_group.setProperty("header", "blue")
        fluid_layout = QVBoxLayout(fluid_group)
        self.velocity_spin = self.create_control_row("Inlet Velocity:", 0.01, 0.20, 0.10, fluid_layout, step=0.01)
        fluid_layout.addStretch()
        bottom_layout.addWidget(fluid_group)

        # Batch Run Group
        batch_group = QGroupBox("Batch run")
        batch_layout = QGridLayout(batch_group)
        batch_layout.setVerticalSpacing(15)
        
        self.radio_single = QRadioButton("Single Run")
        self.radio_single.setChecked(True)
        self.radio_batch = QRadioButton("Automated Run :")
        
        self.batch_count_spin = QSpinBox()
        self.batch_count_spin.setRange(1, 10000)
        self.batch_count_spin.setValue(10)
        self.batch_count_spin.setEnabled(False)
        self.batch_count_spin.setMinimumWidth(70)
        
        batch_layout.addWidget(self.radio_single, 0, 0, 1, 2)
        batch_layout.addWidget(self.radio_batch, 1, 0)
        batch_layout.addWidget(self.batch_count_spin, 1, 1)
        batch_layout.setRowStretch(2, 1)
        bottom_layout.addWidget(batch_group)
        
        # Automated Tuning Group
        self.tuning_group = QGroupBox("Automated Tuning")
        self.tuning_group.setEnabled(False) # Tied to batch radio
        tuning_layout = QGridLayout(self.tuning_group)
        
        self.tune_vel_min = self.create_tuning_spin(0.01, 0.20, 0.02, 0.01)
        self.tune_vel_max = self.create_tuning_spin(0.01, 0.20, 0.10, 0.01)
        self.tune_ang_min = self.create_tuning_spin(0.0, 180.0, 0.0, 1.0)
        self.tune_ang_max = self.create_tuning_spin(0.0, 180.0, 180.0, 1.0)
        
        # Shape Opt-in Checkboxes
        self.tune_shape_circle = QCheckBox("Circle")
        self.tune_shape_rect = QCheckBox("Rectangle")
        self.tune_shape_tri = QCheckBox("Triangle")
        self.tune_shape_circle.setChecked(True)
        self.tune_shape_rect.setChecked(True)
        self.tune_shape_tri.setChecked(True)
        
        # Apply hand cursors for clear interactivity
        for cb in [self.tune_shape_circle, self.tune_shape_rect, self.tune_shape_tri]:
            cb.setCursor(QCursor(Qt.CursorShape.PointingHandCursor))
        
        shape_layout_h = QHBoxLayout()
        shape_layout_h.addWidget(self.tune_shape_circle)
        shape_layout_h.addWidget(self.tune_shape_rect)
        shape_layout_h.addWidget(self.tune_shape_tri)
        
        tuning_layout.addWidget(QLabel("Vel Min:"), 0, 0); tuning_layout.addWidget(self.tune_vel_min, 0, 1)
        tuning_layout.addWidget(QLabel("Vel Max:"), 0, 2); tuning_layout.addWidget(self.tune_vel_max, 0, 3)
        tuning_layout.addWidget(QLabel("Ang Min:"), 1, 0); tuning_layout.addWidget(self.tune_ang_min, 1, 1)
        tuning_layout.addWidget(QLabel("Ang Max:"), 1, 2); tuning_layout.addWidget(self.tune_ang_max, 1, 3)
        tuning_layout.addWidget(QLabel("Shapes:"), 2, 0); tuning_layout.addLayout(shape_layout_h, 2, 1, 1, 3)
        bottom_layout.addWidget(self.tuning_group)
        
        self.radio_single.toggled.connect(self.toggle_modes)

        # Simulate Button
        self.sim_btn = QPushButton("RUN SIMULATION")
        self.sim_btn.setMinimumSize(200, 60)
        font = self.sim_btn.font()
        font.setPointSize(12)
        font.setBold(True)
        self.sim_btn.setFont(font)
        self.sim_btn.clicked.connect(self.start_simulation_dispatcher)
        self.sim_btn.setCursor(QCursor(Qt.CursorShape.PointingHandCursor))
        
        bottom_layout.addWidget(self.sim_btn, alignment=Qt.AlignmentFlag.AlignBottom | Qt.AlignmentFlag.AlignRight)
        main_layout.addLayout(bottom_layout, stretch=1)

    def create_control_row(self, label_text, min_val, max_val, default_val, layout, step=1.0):
        row_widget = QWidget()
        row_layout = QHBoxLayout(row_widget)
        row_layout.setContentsMargins(0, 0, 0, 0)
        
        label = QLabel(label_text)
        spinbox = QDoubleSpinBox()
        spinbox.setRange(min_val, max_val)
        spinbox.setValue(default_val)
        spinbox.setSingleStep(step)
        spinbox.valueChanged.connect(self.update_preview)
        
        row_layout.addWidget(label, stretch=1)
        row_layout.addWidget(spinbox, stretch=1)
        layout.addWidget(row_widget)
        return spinbox

    def create_tuning_spin(self, min_val, max_val, default_val, step):
        spinbox = QDoubleSpinBox()
        spinbox.setRange(min_val, max_val)
        spinbox.setValue(default_val)
        spinbox.setSingleStep(step)
        spinbox.setMinimumWidth(60)
        return spinbox

    def toggle_modes(self):
        is_batch = self.radio_batch.isChecked()
        self.batch_count_spin.setEnabled(is_batch)
        self.tuning_group.setEnabled(is_batch)

    def update_preview(self):
        w, h = 800, 200
        pixmap = QPixmap(w, h)
        pixmap.fill(QColor("#1e1e2e"))
        
        painter = QPainter(pixmap)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        painter.setPen(QPen(QColor("#89b4fa"), 2, Qt.PenStyle.DashLine))
        painter.drawLine(0, 2, w, 2)
        painter.drawLine(0, h-2, w, h-2)

        painter.setPen(QPen(QColor("#45475a"), 2))
        for y in range(20, h, 30):
            painter.drawLine(10, y, 40, y)
            painter.drawLine(30, y-5, 40, y)
            painter.drawLine(30, y+5, 40, y)

        scale = 2.0
        cx = (self.domain_nx / 4.0) * scale
        cy = (self.domain_ny / 2.0) * scale

        shape_display = self.shape_combo.currentText()
        shape_map = {"Circle": "circle", "Rectangle": "rect", "Triangle": "tri"}
        shape = shape_map.get(shape_display, "circle")

        s1 = self.size1_spin.value() * scale
        s2 = self.size2_spin.value() * scale
        angle = self.angle_spin.value()
        vel = self.velocity_spin.value()

        # Update Live Readouts
        L = s2 if shape != "circle" else (s1 * 2)
        re_est = (vel * (L / scale)) / 0.0333
        br_est = (L / scale) / self.domain_ny * 100.0
        
        self.re_label.setText(f"Est. Re: {re_est:.1f}")
        self.br_label.setText(f"Blockage: {br_est:.1f}%")

        painter.translate(cx, cy)
        painter.rotate(angle)
        painter.setBrush(QBrush(QColor("#f38ba8")))
        painter.setPen(QPen(QColor("#11111b"), 2))

        if shape == "circle":
            painter.drawEllipse(QPointF(0, 0), s1, s1)
        elif shape == "rect":
            painter.drawRect(QRectF(-s1/2, -s2/2, s1, s2))
        elif shape == "tri":
            polygon = QPolygonF([
                QPointF(s2/2, 0),
                QPointF(-s2/2, s1/2),
                QPointF(-s2/2, -s1/2)
            ])
            painter.drawPolygon(polygon)

        painter.resetTransform()
        painter.setPen(QColor(255, 255, 255, 20))
        font = painter.font()
        font.setPointSize(75)
        font.setBold(True)
        painter.setFont(font)
        painter.drawText(QRectF(0, 0, w, h), Qt.AlignmentFlag.AlignCenter, "PREVIEW")

        painter.end()
        
        # Reset zoom and display the preview
        self.canvas.reset_view()
        self.canvas.setPixmap(pixmap)

    def start_simulation_dispatcher(self):
        if self.radio_single.isChecked():
            self.start_simulation()
        else:
            self.run_batch_mode()

    def start_simulation(self):
        if not os.path.exists(self.executable):
            QMessageBox.critical(self, "Error", f"Executable '{self.executable}' not found.")
            return

        for f in ["results.csv", "flow_field.ppm", "flow_field.png"]:
            if os.path.exists(f): os.remove(f)

        self.sim_btn.setEnabled(False)
        self.sim_btn.setText("SIMULATING...")
        self.progress_widget.update_progress(0, 15000, "0/15000 steps")

        shape_display = self.shape_combo.currentText()
        shape_map = {"Circle": "circle", "Rectangle": "rect", "Triangle": "tri"}
        shape = shape_map.get(shape_display, "circle")
        
        s1 = str(self.size1_spin.value())
        s2 = str(self.size2_spin.value())
        angle = str(self.angle_spin.value())
        vel = str(self.velocity_spin.value())

        self.process = QProcess(self)
        self.process.readyReadStandardOutput.connect(self.handle_stdout)
        self.process.finished.connect(self.simulation_finished)

        args = [
            "--width", str(self.domain_nx), "--height", str(self.domain_ny),
            "--shape", shape, "--size1", s1, "--size2", s2,
            "--angle", angle, "--velocity", vel, "--steps", "15000"
        ]
        
        self.process.start(self.executable, args)

    def handle_stdout(self):
        output = self.process.readAllStandardOutput().data().decode()
        for line in output.replace('\r', '\n').split('\n'):
            if "Step" in line and "/" in line:
                try:
                    parts = line.strip().split()
                    step = int(parts[1])
                    total = int(parts[3])
                    self.progress_widget.update_progress(step, total, f"{step}/{total} steps")
                except:
                    pass
            elif "Steady-state convergence reached" in line:
                self.progress_widget.update_progress(15000, 15000, "15000/15000 steps")

    def simulation_finished(self):
        self.sim_btn.setEnabled(True)
        self.sim_btn.setText("RUN SIMULATION")

        if self.process.exitStatus() != QProcess.ExitStatus.NormalExit:
            QMessageBox.warning(self, "Error", "Simulation crashed or was terminated.")
            return

        if os.path.exists("results.csv"):
            try:
                with open("results.csv", 'r') as f:
                    reader = csv.DictReader(f)
                    row = next(reader)
                    cd_val = float(row["Cd"])
                    dp_val = float(row["pressure_loss"])
                    self.cd_label.setText(f"C<sub>d</sub>: {cd_val:.4f}")
                    self.dp_label.setText(f"ΔP: {dp_val:.5f}")
            except Exception as e:
                self.cd_label.setText("C<sub>d</sub>: Error")

        if os.path.exists("flow_field.ppm"):
            try:
                with Image.open("flow_field.ppm") as img:
                    img.save("flow_field.png", format="PNG")
                
                pixmap = QPixmap("flow_field.png")
                scaled_pixmap = pixmap.scaled(800, 200, Qt.AspectRatioMode.IgnoreAspectRatio, Qt.TransformationMode.SmoothTransformation)
                
                # Reset zoom and display the result image
                self.canvas.reset_view()
                self.canvas.setPixmap(scaled_pixmap)
            except Exception as e:
                QMessageBox.warning(self, "Image Error", f"Failed to render flow field: {e}")

    def run_batch_mode(self):
        if not os.path.exists(self.executable):
            QMessageBox.critical(self, "Error", f"Executable '{self.executable}' not found.")
            return

        selected_shapes = []
        if self.tune_shape_circle.isChecked(): selected_shapes.append("circle")
        if self.tune_shape_rect.isChecked(): selected_shapes.append("rect")
        if self.tune_shape_tri.isChecked(): selected_shapes.append("tri")

        if len(selected_shapes) == 0:
            QMessageBox.warning(self, "Selection Required", "Please select at least one shape variant to generate.")
            return

        dir_path = QFileDialog.getExistingDirectory(self, "Select Output Directory for Dataset")
        if not dir_path: return

        count = self.batch_count_spin.value()
        
        tuning_params = {
            'vel_min': self.tune_vel_min.value(),
            'vel_max': self.tune_vel_max.value(),
            'ang_min': self.tune_ang_min.value(),
            'ang_max': self.tune_ang_max.value(),
            'shapes': selected_shapes
        }

        self.sim_btn.setEnabled(False)
        self.progress_widget.update_progress(0, count, f"0/{count} samples")

        self.batch_worker = BatchWorker(self.executable, dir_path, count, tuning_params)
        self.batch_worker.progress.connect(self.update_batch_progress)
        self.batch_worker.finished.connect(self.batch_finished)
        self.batch_worker.error.connect(self.batch_error)
        self.batch_worker.start()

    def update_batch_progress(self, current, total):
        self.progress_widget.update_progress(current, total, f"{current}/{total} samples")

    def batch_finished(self, msg):
        self.batch_cleanup()
        QMessageBox.information(self, "Batch Complete", msg)

    def batch_error(self, err):
        self.batch_cleanup()
        QMessageBox.critical(self, "Batch Error", f"An error occurred: {err}")

    def batch_cleanup(self):
        self.sim_btn.setEnabled(True)
        self.progress_widget.update_progress(100, 100, "Done")
        self.batch_worker = None

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

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = LBMSolverGUI()
    window.show()
    sys.exit(app.exec())
