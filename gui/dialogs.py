"""Dialogs: palette editor, custom formula, high-res image save, video exports."""
from __future__ import annotations

import os
import time

import numpy as np
from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtGui import QColor, QImage, QPixmap
from PySide6.QtWidgets import (QCheckBox, QColorDialog, QComboBox, QDialog,
                               QDialogButtonBox, QDoubleSpinBox, QFileDialog,
                               QFormLayout, QGridLayout, QGroupBox, QHBoxLayout,
                               QLabel, QLineEdit, QMessageBox, QProgressBar,
                               QPushButton, QSpinBox, QTabWidget, QVBoxLayout,
                               QWidget)

from core import engine as eng
from core import video as vid
from core.formulas import EscapeFormula, FormulaError, NewtonFormula, parse_formula
from core.palette import (Palette, gradient_palette, palette_from_recipe,
                          palette_strip_image, sphere_great_circle_palette,
                          sphere_spiral_palette)


def _strip_pixmap(colors: np.ndarray, width: int = 420, height: int = 26) -> QPixmap:
    strip = palette_strip_image(colors, height)
    img = QImage(strip.tobytes(), strip.shape[1], height, strip.shape[1] * 3,
                 QImage.Format.Format_RGB888)
    return QPixmap.fromImage(img).scaled(width, height)


# ============================================================================= palette editor

class PaletteEditor(QDialog):
    """Two parametric tabs (gradient stops, sphere paths) with live preview."""

    def __init__(self, parent, current: Palette):
        super().__init__(parent)
        self.setWindowTitle("Palette Editor")
        self.result_palette: Palette | None = None

        v = QVBoxLayout(self)
        self.name_edit = QLineEdit(current.name)
        nrow = QHBoxLayout()
        nrow.addWidget(QLabel("Name:")); nrow.addWidget(self.name_edit)
        v.addLayout(nrow)

        self.tabs = QTabWidget()
        v.addWidget(self.tabs)
        self._build_gradient_tab(current)
        self._build_sphere_tab(current)

        self.preview = QLabel()
        self.preview.setFixedHeight(28)
        v.addWidget(self.preview)

        io = QHBoxLayout()
        b_load = QPushButton("Load JSON…"); b_load.clicked.connect(self._load)
        b_save = QPushButton("Save JSON…"); b_save.clicked.connect(self._save)
        io.addWidget(b_load); io.addWidget(b_save); io.addStretch(1)
        v.addLayout(io)

        bb = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok |
                              QDialogButtonBox.StandardButton.Cancel)
        bb.accepted.connect(self._accept); bb.rejected.connect(self.reject)
        v.addWidget(bb)

        if current.recipe and current.recipe.get("kind", "").startswith("sphere"):
            self.tabs.setCurrentIndex(1)
        self.tabs.currentChanged.connect(lambda _idx: self._refresh())
        self._refresh()

    # -- gradient tab ---------------------------------------------------------
    def _build_gradient_tab(self, current: Palette):
        w = QWidget(); lay = QVBoxLayout(w)
        self.stop_rows: list[tuple[QDoubleSpinBox, QPushButton, QColor]] = []
        self.stops_box = QVBoxLayout()
        lay.addLayout(self.stops_box)
        btns = QHBoxLayout()
        add = QPushButton("Add stop"); add.clicked.connect(lambda: (self._add_stop(0.5, QColor(255, 255, 255)), self._refresh()))
        rem = QPushButton("Remove last"); rem.clicked.connect(self._remove_stop)
        btns.addWidget(add); btns.addWidget(rem); btns.addStretch(1)
        lay.addLayout(btns)
        lay.addStretch(1)
        self.tabs.addTab(w, "Gradient stops")

        stops = None
        if current.recipe and current.recipe.get("kind") == "gradient":
            stops = [(p, QColor(*c)) for p, c in current.recipe["stops"]]
        if not stops:
            stops = [(0.0, QColor(10, 0, 60)), (0.35, QColor(40, 120, 255)),
                     (0.65, QColor(255, 230, 80)), (1.0, QColor(255, 60, 20))]
        for p, c in stops:
            self._add_stop(p, c)

    def _add_stop(self, pos: float, color: QColor):
        row = QHBoxLayout()
        sp = QDoubleSpinBox(); sp.setRange(0.0, 1.0); sp.setSingleStep(0.05)
        sp.setDecimals(3); sp.setValue(pos)
        sp.valueChanged.connect(lambda _: self._refresh())
        btn = QPushButton(); btn.setFixedWidth(70)
        entry = [sp, btn, QColor(color)]
        def pick():
            c = QColorDialog.getColor(entry[2], self, "Stop color")
            if c.isValid():
                entry[2] = c
                btn.setStyleSheet(f"background-color: {c.name()}")
                self._refresh()
        btn.clicked.connect(pick)
        btn.setStyleSheet(f"background-color: {color.name()}")
        row.addWidget(QLabel("pos")); row.addWidget(sp); row.addWidget(btn)
        holder = QWidget(); holder.setLayout(row)
        self.stops_box.addWidget(holder)
        self.stop_rows.append(entry)

    def _remove_stop(self):
        if len(self.stop_rows) <= 2:
            return
        self.stop_rows.pop()
        item = self.stops_box.takeAt(self.stops_box.count() - 1)
        if item and item.widget():
            item.widget().deleteLater()
        self._refresh()

    # -- sphere tab -----------------------------------------------------------
    def _build_sphere_tab(self, current: Palette):
        w = QWidget(); form = QFormLayout(w)
        self.sphere_kind = QComboBox()
        self.sphere_kind.addItems(["Great circle", "Spiral (pole to pole)"])
        self.sp_tilt = QDoubleSpinBox(); self.sp_tilt.setRange(0, 180); self.sp_tilt.setValue(55)
        self.sp_az = QDoubleSpinBox(); self.sp_az.setRange(0, 360); self.sp_az.setValue(30)
        self.sp_turns = QDoubleSpinBox(); self.sp_turns.setRange(0.25, 40); self.sp_turns.setValue(3)
        self.sp_phase = QDoubleSpinBox(); self.sp_phase.setRange(0, 1); self.sp_phase.setSingleStep(0.05)
        self.sp_axis = QComboBox(); self.sp_axis.addItems(["gray", "r", "g", "b"])
        form.addRow("Path", self.sphere_kind)
        form.addRow("Tilt (deg)", self.sp_tilt)
        form.addRow("Azimuth (deg)", self.sp_az)
        form.addRow("Turns", self.sp_turns)
        form.addRow("Phase", self.sp_phase)
        form.addRow("Pole axis", self.sp_axis)
        note = QLabel("Paths on the sphere inscribed in the RGB cube —\n"
                      "vibrant colors that blend smoothly and wrap seamlessly.")
        note.setStyleSheet("color: gray")
        form.addRow(note)
        self.tabs.addTab(w, "Sphere path")
        for c in (self.sp_tilt, self.sp_az, self.sp_turns, self.sp_phase):
            c.valueChanged.connect(lambda _: self._refresh())
        self.sphere_kind.currentIndexChanged.connect(lambda _: self._refresh())
        self.sp_axis.currentIndexChanged.connect(lambda _: self._refresh())
        r = current.recipe or {}
        if r.get("kind") == "sphere_spiral":
            self.sphere_kind.setCurrentIndex(1)
            self.sp_turns.setValue(r.get("turns", 3.0))
            self.sp_phase.setValue(r.get("phase", 0.0))
        elif r.get("kind") == "sphere_great_circle":
            self.sp_tilt.setValue(r.get("tilt_deg", 55.0))
            self.sp_az.setValue(r.get("azimuth_deg", 30.0))
            self.sp_phase.setValue(r.get("phase", 0.0))

    # -- shared ---------------------------------------------------------------
    def _current_recipe(self) -> dict:
        if self.tabs.currentIndex() == 0:
            stops = sorted(((sp.value(), (c.red(), c.green(), c.blue()))
                            for sp, _b, c in self.stop_rows), key=lambda s: s[0])
            return {"kind": "gradient", "stops": [[p, list(c)] for p, c in stops]}
        if self.sphere_kind.currentIndex() == 0:
            return {"kind": "sphere_great_circle", "tilt_deg": self.sp_tilt.value(),
                    "azimuth_deg": self.sp_az.value(), "phase": self.sp_phase.value()}
        return {"kind": "sphere_spiral", "turns": self.sp_turns.value(),
                "phase": self.sp_phase.value(), "pole_axis": self.sp_axis.currentText()}

    def _current_palette(self) -> Palette:
        recipe = self._current_recipe()
        return Palette(self.name_edit.text() or "Custom",
                       palette_from_recipe(recipe), recipe)

    def _refresh(self):
        try:
            self.preview.setPixmap(_strip_pixmap(self._current_palette().colors))
        except Exception:
            pass

    def _accept(self):
        self.result_palette = self._current_palette()
        self.accept()

    def _save(self):
        path, _ = QFileDialog.getSaveFileName(self, "Save palette", "", "Palette (*.json)")
        if path:
            self._current_palette().save(path)

    def _load(self):
        path, _ = QFileDialog.getOpenFileName(self, "Load palette", "", "Palette (*.json)")
        if path:
            try:
                self.result_palette = Palette.load(path)
                self.accept()
            except Exception as e:
                QMessageBox.warning(self, "Load failed", str(e))


# ============================================================================= custom formula

class FormulaDialog(QDialog):
    """Enter/validate a custom escape-time formula or Newton coefficients."""

    def __init__(self, parent):
        super().__init__(parent)
        self.setWindowTitle("Custom formula")
        self.escape_result: EscapeFormula | None = None
        self.newton_result: NewtonFormula | None = None
        v = QVBoxLayout(self)
        self.tabs = QTabWidget(); v.addWidget(self.tabs)

        # escape tab
        w1 = QWidget(); f1 = QFormLayout(w1)
        self.name1 = QLineEdit("My formula")
        self.expr = QLineEdit("z**2 + c")
        self.expr.setPlaceholderText("expression in z and c, e.g.  z**3 + c*sin(z)")
        self.degree = QDoubleSpinBox(); self.degree.setRange(1.01, 64); self.degree.setValue(2.0)
        self.bailout = QDoubleSpinBox(); self.bailout.setRange(4, 1e9)
        self.bailout.setDecimals(0); self.bailout.setValue(1000)
        self.z0 = QLineEdit("0")
        f1.addRow("Name", self.name1)
        f1.addRow("f(z, c) =", self.expr)
        f1.addRow("Degree (for smoothing)", self.degree)
        f1.addRow("Bailout radius", self.bailout)
        f1.addRow("z\u2080 (mandelbrot plane)", self.z0)
        hint = QLabel("Allowed: z, c, numbers, + - * / **, sin cos tan sinh cosh tanh\n"
                      "exp log sqrt asin acos atan abs conj re im")
        hint.setStyleSheet("color: gray")
        f1.addRow(hint)
        self.tabs.addTab(w1, "Escape-time f(z, c)")

        # newton tab
        w2 = QWidget(); f2 = QFormLayout(w2)
        self.name2 = QLineEdit("My Newton")
        self.coeffs = QLineEdit("1, 0, 0, -1")
        self.coeffs.setPlaceholderText("polynomial coefficients, highest degree first")
        f2.addRow("Name", self.name2)
        f2.addRow("p(z) coeffs", self.coeffs)
        f2.addRow(QLabel("e.g.  z\u00b3 \u2212 1  \u2192  1, 0, 0, -1     (complex ok: 1, 0, 1j, -1)"))
        self.tabs.addTab(w2, "Newton basins p(z)")

        bb = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok |
                              QDialogButtonBox.StandardButton.Cancel)
        bb.accepted.connect(self._accept); bb.rejected.connect(self.reject)
        v.addWidget(bb)

    def _accept(self):
        try:
            if self.tabs.currentIndex() == 0:
                parse_formula(self.expr.text())
                self.escape_result = EscapeFormula(
                    self.name1.text() or self.expr.text(), self.expr.text().strip(),
                    self.degree.value(), self.bailout.value(),
                    complex(self.z0.text().replace(" ", "") or "0"))
            else:
                coeffs = NewtonFormula.parse_coeffs(self.coeffs.text())
                self.newton_result = NewtonFormula(self.name2.text() or "Newton", coeffs)
            self.accept()
        except (FormulaError, ValueError) as e:
            QMessageBox.warning(self, "Invalid formula", str(e))


# ============================================================================= background jobs

class _JobThread(QThread):
    progressed = Signal(float, str)
    done = Signal(object)

    def __init__(self, fn):
        super().__init__()
        self.fn = fn
        self.cancel = eng.CancelToken()

    def run(self):
        try:
            out = self.fn(self.cancel, lambda p, msg="": self.progressed.emit(p, msg))
        except Exception as e:                     # surface errors to the dialog
            out = e
        self.done.emit(out)


class _ExportDialogBase(QDialog):
    """Shared progress/cancel plumbing for long exports."""

    def _add_progress_ui(self, v: QVBoxLayout):
        self.pbar = QProgressBar(); self.pbar.setRange(0, 1000)
        self.status = QLabel("")
        v.addWidget(self.pbar); v.addWidget(self.status)
        row = QHBoxLayout()
        self.go = QPushButton("Render"); self.go.clicked.connect(self._start)
        self.stop = QPushButton("Cancel"); self.stop.setEnabled(False)
        self.stop.clicked.connect(self._cancel)
        close = QPushButton("Close"); close.clicked.connect(self.reject)
        row.addStretch(1); row.addWidget(self.go); row.addWidget(self.stop); row.addWidget(close)
        v.addLayout(row)
        self.thread: _JobThread | None = None
        self._t0 = 0.0

    def _start(self):
        fn = self._make_job()
        if fn is None:
            return
        self._t0 = time.time()
        self.thread = _JobThread(fn)
        self.thread.progressed.connect(self._on_progress)
        self.thread.done.connect(self._on_done)
        self.go.setEnabled(False); self.stop.setEnabled(True)
        self.thread.start()

    def _cancel(self):
        if self.thread:
            self.thread.cancel.cancel()
            self.status.setText("cancelling…")

    def _on_progress(self, p: float, msg: str):
        self.pbar.setValue(int(p * 1000))
        el = time.time() - self._t0
        eta = el / p - el if p > 1e-6 else 0.0
        self.status.setText(f"{msg}   elapsed {el:.0f}s   eta {eta:.0f}s")

    def _on_done(self, out):
        self.go.setEnabled(True); self.stop.setEnabled(False)
        if isinstance(out, Exception):
            QMessageBox.critical(self, "Render failed", str(out))
            self.status.setText("failed")
        elif out is False or out is None:
            self.status.setText("cancelled")
        else:
            self.status.setText("done ✓")
        self.thread = None

    def _make_job(self):
        raise NotImplementedError

    def closeEvent(self, ev):
        if self.thread and self.thread.isRunning():
            self.thread.cancel.cancel()
            self.thread.wait(5000)
        super().closeEvent(ev)


def _res_combo() -> QComboBox:
    cb = QComboBox()
    for k in vid.RESOLUTIONS:
        cb.addItem(k)
    cb.setCurrentIndex(2)  # 4K default
    return cb


# ============================================================================= high-res still

class SaveImageDialog(_ExportDialogBase):
    def __init__(self, parent, settings, view, palette, cs):
        super().__init__(parent)
        self.setWindowTitle("Save high-resolution image")
        self.args = (settings, view, palette, cs)
        v = QVBoxLayout(self)
        form = QFormLayout()
        self.wspin = QSpinBox(); self.wspin.setRange(64, 65536); self.wspin.setValue(3840)
        self.hspin = QSpinBox(); self.hspin.setRange(64, 65536); self.hspin.setValue(2160)
        self.ss = QComboBox(); self.ss.addItems(["1", "2", "3", "4"]); self.ss.setCurrentIndex(1)
        self.path = QLineEdit(os.path.expanduser("~/fractal.png"))
        pick = QPushButton("…"); pick.setFixedWidth(30)
        pick.clicked.connect(self._pick)
        prow = QHBoxLayout(); prow.addWidget(self.path); prow.addWidget(pick)
        form.addRow("Width", self.wspin)
        form.addRow("Height", self.hspin)
        form.addRow("Supersample", self.ss)
        form.addRow("File", prow)
        v.addLayout(form)
        self._add_progress_ui(v)

    def _pick(self):
        p, _ = QFileDialog.getSaveFileName(self, "Save image", self.path.text(),
                                           "PNG (*.png)")
        if p:
            self.path.setText(p)

    def _make_job(self):
        settings, view, palette, cs = self.args
        W, H, ss = self.wspin.value(), self.hspin.value(), int(self.ss.currentText())
        path = self.path.text()
        if not path:
            return None

        def job(cancel, progress):
            img = eng.render_highres_tiled(settings, view, palette, cs, W, H, ss,
                                           progress=lambda p: progress(p, f"{W}x{H}"),
                                           cancel=cancel)
            if img is None:
                return False
            import imageio.v2 as iio
            from core.engine import location_to_dict
            import json as _json
            iio.imwrite(path, img)
            # sidecar with full parameters so the still is reproducible
            with open(path + ".location.json", "w") as f:
                _json.dump(location_to_dict(settings, view, palette, cs,
                                            os.path.basename(path)), f, indent=1)
            return True
        return job


# ============================================================================= zoom video

class ZoomVideoDialog(_ExportDialogBase):
    def __init__(self, parent, settings, view, palette, cs, home_span: float):
        super().__init__(parent)
        self.setWindowTitle("Export zoom-in video")
        self.args = (settings, view, palette, cs)
        v = QVBoxLayout(self)
        form = QFormLayout()
        self.res = _res_combo()
        self.fps = QSpinBox(); self.fps.setRange(10, 120); self.fps.setValue(30)
        self.rate = QDoubleSpinBox(); self.rate.setRange(1.05, 64.0)
        self.rate.setSingleStep(0.1); self.rate.setValue(2.0)
        self.start_span = QDoubleSpinBox(); self.start_span.setDecimals(6)
        self.start_span.setRange(1e-13, 1e6); self.start_span.setValue(home_span)
        self.ss = QComboBox(); self.ss.addItems(["1", "2", "3"]); self.ss.setCurrentIndex(1)
        self.hold = QDoubleSpinBox(); self.hold.setRange(0, 30); self.hold.setValue(1.0)
        self.cycle = QCheckBox("Cycle palette during zoom")
        self.cycle_speed = QDoubleSpinBox(); self.cycle_speed.setRange(1, 500)
        self.cycle_speed.setValue(cs.cycle_speed)
        self.keep_png = QCheckBox("Also keep PNG frame sequence")
        self.path = QLineEdit(os.path.expanduser("~/fractal_zoom.mp4"))
        pick = QPushButton("…"); pick.setFixedWidth(30); pick.clicked.connect(self._pick)
        prow = QHBoxLayout(); prow.addWidget(self.path); prow.addWidget(pick)
        self.info = QLabel("")
        form.addRow("Resolution", self.res)
        form.addRow("FPS", self.fps)
        form.addRow("Zoom rate (× per second)", self.rate)
        form.addRow("Start span (home width)", self.start_span)
        form.addRow("Supersample", self.ss)
        form.addRow("Hold at end (s)", self.hold)
        form.addRow(self.cycle)
        form.addRow("Cycle speed (idx/s)", self.cycle_speed)
        form.addRow(self.keep_png)
        form.addRow("Output MP4", prow)
        form.addRow(self.info)
        v.addLayout(form)
        self._add_progress_ui(v)
        for wgt in (self.fps, self.rate, self.start_span):
            wgt.valueChanged.connect(self._update_info)
        self._update_info()

    def _spec(self) -> vid.ZoomVideoSpec:
        settings, view, palette, cs = self.args
        W, H = vid.RESOLUTIONS[self.res.currentText()]
        png_dir = None
        if self.keep_png.isChecked():
            png_dir = os.path.splitext(self.path.text())[0] + "_frames"
        return vid.ZoomVideoSpec(end_view=view, start_span=self.start_span.value(),
                                 rate_per_sec=self.rate.value(), fps=self.fps.value(),
                                 width=W, height=H, supersample=int(self.ss.currentText()),
                                 hold_seconds=self.hold.value(),
                                 cycle_colors=self.cycle.isChecked(),
                                 cycle_speed=self.cycle_speed.value(), png_dir=png_dir)

    def _update_info(self):
        s = self._spec()
        self.info.setText(f"{s.n_zoom_frames()} frames, "
                          f"≈{s.duration_seconds():.1f} s of video")

    def _pick(self):
        p, _ = QFileDialog.getSaveFileName(self, "Save video", self.path.text(),
                                           "MP4 (*.mp4)")
        if p:
            self.path.setText(p)

    def _make_job(self):
        settings, view, palette, cs = self.args
        spec = self._spec()
        path = self.path.text()
        if not path:
            return None
        return lambda cancel, progress: vid.render_zoom_video(
            spec, settings, palette, cs, path, progress=progress, cancel=cancel)


# ============================================================================= julia morph video

class JuliaMorphDialog(_ExportDialogBase):
    def __init__(self, parent, settings, view, palette, cs):
        super().__init__(parent)
        self.setWindowTitle("Export Julia morph video")
        self.args = (settings, view, palette, cs)
        v = QVBoxLayout(self)
        form = QFormLayout()
        self.path_kind = QComboBox()
        self.path_kind.addItems(["circle (seamless loop)", "spiral outward", "line to c\u2081"])
        c0 = settings.julia_c
        self.c0 = QLineEdit(f"{c0.real:+.9f}{c0.imag:+.9f}j")
        self.radius = QDoubleSpinBox(); self.radius.setDecimals(6)
        self.radius.setRange(1e-6, 2.0); self.radius.setSingleStep(0.005)
        self.radius.setValue(0.02)
        self.turns = QDoubleSpinBox(); self.turns.setRange(0.25, 40); self.turns.setValue(1.0)
        self.c1 = QLineEdit("0+0j")
        self.roundtrip = QCheckBox("There and back (loops)"); self.roundtrip.setChecked(True)
        self.duration = QDoubleSpinBox(); self.duration.setRange(1, 600); self.duration.setValue(12)
        self.fps = QSpinBox(); self.fps.setRange(10, 120); self.fps.setValue(30)
        self.res = _res_combo()
        self.ss = QComboBox(); self.ss.addItems(["1", "2", "3"]); self.ss.setCurrentIndex(1)
        self.combine_zoom = QCheckBox("Combine with zoom to current view")
        self.cycle = QCheckBox("Cycle palette")
        self.keep_png = QCheckBox("Also keep PNG frame sequence")
        self.path = QLineEdit(os.path.expanduser("~/julia_morph.mp4"))
        pick = QPushButton("…"); pick.setFixedWidth(30); pick.clicked.connect(self._pick)
        prow = QHBoxLayout(); prow.addWidget(self.path); prow.addWidget(pick)
        form.addRow("Path", self.path_kind)
        form.addRow("c\u2080", self.c0)
        form.addRow("Radius", self.radius)
        form.addRow("Turns", self.turns)
        form.addRow("c\u2081 (line)", self.c1)
        form.addRow(self.roundtrip)
        form.addRow("Duration (s)", self.duration)
        form.addRow("FPS", self.fps)
        form.addRow("Resolution", self.res)
        form.addRow("Supersample", self.ss)
        form.addRow(self.combine_zoom)
        form.addRow(self.cycle)
        form.addRow(self.keep_png)
        form.addRow("Output MP4", prow)
        v.addLayout(form)
        self._add_progress_ui(v)

    def _pick(self):
        p, _ = QFileDialog.getSaveFileName(self, "Save video", self.path.text(),
                                           "MP4 (*.mp4)")
        if p:
            self.path.setText(p)

    def _make_job(self):
        settings, view, palette, cs = self.args
        kinds = {0: "circle", 1: "spiral", 2: "line"}
        W, H = vid.RESOLUTIONS[self.res.currentText()]
        try:
            c0 = complex(self.c0.text().replace(" ", ""))
            c1 = complex(self.c1.text().replace(" ", ""))
        except ValueError:
            QMessageBox.warning(self, "Bad constant", "c must look like -0.7269+0.1889j")
            return None
        png_dir = None
        if self.keep_png.isChecked():
            png_dir = os.path.splitext(self.path.text())[0] + "_frames"
        home_view = eng.ViewState(view.center, settings.formula.default_span) \
            if self.combine_zoom.isChecked() else view
        spec = vid.JuliaMorphSpec(
            c0=c0, path=kinds[self.path_kind.currentIndex()],
            radius=self.radius.value(), turns=self.turns.value(), c1=c1,
            there_and_back=self.roundtrip.isChecked(),
            duration=self.duration.value(), fps=self.fps.value(),
            width=W, height=H, supersample=int(self.ss.currentText()),
            view=home_view,
            zoom_end_span=view.span if self.combine_zoom.isChecked() else None,
            cycle_colors=self.cycle.isChecked(), cycle_speed=cs.cycle_speed,
            png_dir=png_dir)
        path = self.path.text()
        if not path:
            return None
        return lambda cancel, progress: vid.render_julia_morph_video(
            spec, settings, palette, cs, path, progress=progress, cancel=cancel)
