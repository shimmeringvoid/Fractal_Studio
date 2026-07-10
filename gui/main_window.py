"""Main window: interactive fractal canvas + settings dock.

Mouse controls
    left-click            zoom in by the click factor (default x10), centered there
    shift + left-click    zoom out by the same factor
    left-drag             pan
    scroll wheel          zoom in/out x1.5 per notch, anchored at the cursor
    right-click           open the Julia set with c = clicked point

Rendering is progressive: a fast low-res preview appears immediately, then the
full-resolution pass fills in (strip by strip, cancellable). The smooth-iteration
field is cached, so palette edits, density/offset changes, and color cycling
recolor instantly without re-rendering.
"""
from __future__ import annotations

import copy
import glob
import os
import time

import numpy as np
from PySide6.QtCore import QPoint, Qt, QThread, QTimer, Signal
from PySide6.QtGui import QAction, QImage, QKeySequence, QPainter, QPixmap
from PySide6.QtWidgets import (QCheckBox, QComboBox, QDockWidget, QDoubleSpinBox,
                               QFileDialog, QFormLayout, QHBoxLayout, QInputDialog,
                               QLabel, QLineEdit, QMainWindow, QMessageBox,
                               QProgressBar, QPushButton, QSlider, QSpinBox,
                               QVBoxLayout, QWidget)

from core import engine as eng
from core.formulas import NEWTON_PRESETS, PRESETS, EscapeFormula, NewtonFormula
from core.palette import Palette, builtin_palettes
from gui.dialogs import (FormulaDialog, JuliaMorphDialog, PaletteEditor,
                         SaveImageDialog, ZoomVideoDialog)

LOCATIONS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                             "locations")


# ----------------------------------------------------------------------------- render thread

class RenderThread(QThread):
    preview_ready = Signal(object, int)          # rgb ndarray, generation
    field_ready = Signal(object, object, int)    # RenderResult, rgb ndarray, generation
    progressed = Signal(float, int)

    def __init__(self, settings, view, palette, cs, width, height, supersample, gen):
        super().__init__()
        self.args = (copy.deepcopy(settings), view, palette, cs, width, height,
                     supersample, gen)
        self.cancel = eng.CancelToken()

    def run(self):
        settings, view, palette, cs, W, H, ss, gen = self.args
        # fast preview pass
        pw, ph = max(64, W // 6), max(48, H // 6)
        res = eng.render_field(settings, view, pw, ph, 1, cancel=self.cancel)
        if res is None:
            return
        rgb = eng.colorize(res, palette, cs)
        self.preview_ready.emit(rgb, gen)
        # full pass
        res = eng.render_field(settings, view, W, H, ss,
                               progress=lambda p: self.progressed.emit(p, gen),
                               cancel=self.cancel)
        if res is None:
            return
        rgb = eng.downsample_rgb(eng.colorize(res, palette, cs), res.supersample)
        self.field_ready.emit(res, rgb, gen)


# ----------------------------------------------------------------------------- canvas

class FractalView(QWidget):
    clicked = Signal(complex, object)     # point, Qt modifiers
    right_clicked = Signal(complex)
    wheel_zoom = Signal(complex, float)   # anchor point, factor
    panned = Signal(int, int)             # dx, dy pixels (on release)

    def __init__(self, win: "MainWindow"):
        super().__init__()
        self.win = win
        self.setMinimumSize(320, 240)
        self.setMouseTracking(True)
        self.pix: QPixmap | None = None
        self._press: QPoint | None = None
        self._drag = QPoint(0, 0)
        self._dragging = False

    def set_image(self, rgb: np.ndarray):
        h, w, _ = rgb.shape
        img = QImage(rgb.tobytes(), w, h, 3 * w, QImage.Format.Format_RGB888)
        self.pix = QPixmap.fromImage(img.copy())
        self._drag = QPoint(0, 0)
        self.update()

    def paintEvent(self, ev):
        p = QPainter(self)
        p.fillRect(self.rect(), Qt.GlobalColor.black)
        if self.pix is not None:
            p.drawPixmap(self._drag, self.pix.scaled(self.size()))

    def _pt(self, pos) -> complex:
        return self.win.view.complex_at(pos.x(), pos.y(), self.width(), self.height())

    def mousePressEvent(self, ev):
        if ev.button() in (Qt.MouseButton.LeftButton, Qt.MouseButton.MiddleButton):
            self._press = ev.position().toPoint()
            self._dragging = False

    def mouseMoveEvent(self, ev):
        c = self._pt(ev.position())
        self.win.show_cursor_coord(c)
        if self._press is not None:
            d = ev.position().toPoint() - self._press
            if self._dragging or d.manhattanLength() > 6:
                self._dragging = True
                self._drag = d
                self.update()

    def mouseReleaseEvent(self, ev):
        if ev.button() == Qt.MouseButton.RightButton:
            self.right_clicked.emit(self._pt(ev.position()))
            return
        if self._press is None:
            return
        d = ev.position().toPoint() - self._press
        self._press = None
        if self._dragging:
            self._dragging = False
            self.panned.emit(d.x(), d.y())
        elif ev.button() == Qt.MouseButton.LeftButton:
            self.clicked.emit(self._pt(ev.position()), ev.modifiers())

    def wheelEvent(self, ev):
        notches = ev.angleDelta().y() / 120.0
        if notches:
            self.wheel_zoom.emit(self._pt(ev.position()), 1.5 ** notches)


# ----------------------------------------------------------------------------- main window

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Fractal Studio")
        self.resize(1600, 950)

        self.settings = eng.RenderSettings()
        self.view = eng.ViewState(PRESETS[0].default_center, PRESETS[0].default_span)
        self.palettes = builtin_palettes()
        self.palette_obj = self.palettes[0]
        self.cs = eng.ColorSettings()
        self.escape_formulas: list[EscapeFormula] = list(PRESETS)
        self.newton_formulas: list[NewtonFormula] = list(NEWTON_PRESETS)

        self.history: list[dict] = []
        self.hist_pos = -1
        self.zoom_click_factor = 10.0

        self.gen = 0
        self.thread: RenderThread | None = None
        self.cached: eng.RenderResult | None = None

        self.canvas = FractalView(self)
        self.setCentralWidget(self.canvas)
        self.canvas.clicked.connect(self.on_click)
        self.canvas.right_clicked.connect(self.on_right_click)
        self.canvas.wheel_zoom.connect(self.on_wheel)
        self.canvas.panned.connect(self.on_pan)

        self._build_dock()
        self._build_menus()
        self._build_statusbar()

        self.cycle_timer = QTimer(self)
        self.cycle_timer.setInterval(40)
        self.cycle_timer.timeout.connect(self._cycle_tick)
        self._last_cycle = time.time()

        self.resize_timer = QTimer(self)
        self.resize_timer.setSingleShot(True)
        self.resize_timer.setInterval(250)
        self.resize_timer.timeout.connect(self.request_render)

        self.push_history()
        QTimer.singleShot(50, self.request_render)

    # ------------------------------------------------------------------ dock UI
    def _build_dock(self):
        dock = QDockWidget("Settings", self)
        dock.setFeatures(QDockWidget.DockWidgetFeature.DockWidgetMovable)
        w = QWidget(); form = QFormLayout(w)

        self.mode_cb = QComboBox()
        self.mode_cb.addItems(["Escape-time", "Newton basins"])
        self.mode_cb.currentIndexChanged.connect(self.on_mode_changed)
        form.addRow("Mode", self.mode_cb)

        frow = QHBoxLayout()
        self.formula_cb = QComboBox()
        for f in self.escape_formulas:
            self.formula_cb.addItem(f.name)
        self.formula_cb.currentIndexChanged.connect(self.on_formula_changed)
        newf = QPushButton("New…"); newf.setFixedWidth(52)
        newf.clicked.connect(self.on_new_formula)
        frow.addWidget(self.formula_cb, 1); frow.addWidget(newf)
        form.addRow("Formula", frow)

        self.newton_cb = QComboBox()
        for f in self.newton_formulas:
            self.newton_cb.addItem(f.name)
        self.newton_cb.currentIndexChanged.connect(self.on_newton_changed)
        form.addRow("Newton p(z)", self.newton_cb)

        self.plane_cb = QComboBox()
        self.plane_cb.addItems(["Mandelbrot plane (c = pixel)", "Julia (fixed c)"])
        self.plane_cb.currentIndexChanged.connect(self.on_plane_changed)
        form.addRow("Plane", self.plane_cb)

        self.julia_edit = QLineEdit(self._fmt_c(self.settings.julia_c))
        self.julia_edit.editingFinished.connect(self.on_julia_c_edited)
        form.addRow("Julia c", self.julia_edit)

        it = QHBoxLayout()
        self.iter_spin = QSpinBox(); self.iter_spin.setRange(10, 100000)
        self.iter_spin.setValue(self.settings.max_iter)
        self.iter_spin.valueChanged.connect(self.on_iters_changed)
        self.auto_iter = QCheckBox("auto"); self.auto_iter.setChecked(True)
        self.auto_iter.toggled.connect(self.on_iters_changed)
        it.addWidget(self.iter_spin, 1); it.addWidget(self.auto_iter)
        form.addRow("Max iterations", it)

        self.zoom_spin = QDoubleSpinBox()
        self.zoom_spin.setRange(1.1, 1000.0); self.zoom_spin.setValue(10.0)
        self.zoom_spin.valueChanged.connect(lambda v: setattr(self, "zoom_click_factor", v))
        form.addRow("Click zoom ×", self.zoom_spin)

        prow = QHBoxLayout()
        self.palette_cb = QComboBox()
        for p in self.palettes:
            self.palette_cb.addItem(p.name)
        self.palette_cb.currentIndexChanged.connect(self.on_palette_changed)
        pedit = QPushButton("Edit…"); pedit.setFixedWidth(52)
        pedit.clicked.connect(self.on_edit_palette)
        prow.addWidget(self.palette_cb, 1); prow.addWidget(pedit)
        form.addRow("Palette", prow)

        self.density = QDoubleSpinBox()
        self.density.setRange(0.05, 64.0); self.density.setSingleStep(0.25)
        self.density.setValue(self.cs.density)
        self.density.valueChanged.connect(self.on_color_changed)
        form.addRow("Color density", self.density)

        self.offset_slider = QSlider(Qt.Orientation.Horizontal)
        self.offset_slider.setRange(0, 254)
        self.offset_slider.valueChanged.connect(self.on_color_changed)
        form.addRow("Color offset", self.offset_slider)

        self.log_cb = QCheckBox("Log color mapping (deep zooms)")
        self.log_cb.toggled.connect(self.on_color_changed)
        form.addRow(self.log_cb)

        cyc = QHBoxLayout()
        self.cycle_cb = QCheckBox("Cycle colors")
        self.cycle_cb.toggled.connect(self.on_cycle_toggled)
        self.cycle_speed = QDoubleSpinBox()
        self.cycle_speed.setRange(1, 500); self.cycle_speed.setValue(self.cs.cycle_speed)
        self.cycle_speed.valueChanged.connect(
            lambda v: setattr(self.cs, "cycle_speed", v))
        cyc.addWidget(self.cycle_cb); cyc.addWidget(self.cycle_speed)
        form.addRow(cyc)

        self.ss_cb = QComboBox(); self.ss_cb.addItems(["1", "2"])
        self.ss_cb.currentIndexChanged.connect(lambda _: self.request_render())
        form.addRow("Display supersample", self.ss_cb)

        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, dock)
        dock.setWidget(w)
        self._sync_dock_enabled()

    def _build_menus(self):
        m = self.menuBar()

        fm = m.addMenu("&File")
        self._act(fm, "Save high-res image…", "Ctrl+S", self.on_save_image)
        self._act(fm, "Save location…", "Ctrl+L", self.on_save_location)
        self._act(fm, "Load location…", "Ctrl+O", self.on_load_location)
        fm.addSeparator()
        self._act(fm, "Quit", "Ctrl+Q", self.close)

        vm = m.addMenu("&View")
        self._act(vm, "Home (reset view)", "H", self.on_home)
        self._act(vm, "Back", "Backspace", self.on_back)
        self._act(vm, "Forward", "Shift+Backspace", self.on_forward)
        self._act(vm, "Zoom in ×", "+", lambda: self.zoom_at(self.view.center, self.zoom_click_factor))
        self._act(vm, "Zoom out ×", "-", lambda: self.zoom_at(self.view.center, 1.0 / self.zoom_click_factor))

        bm = m.addMenu("&Bookmarks")
        self._act(bm, "Save bookmark…", "Ctrl+B", self.on_save_bookmark)
        bm.addSeparator()
        self.bookmarks_menu = bm
        self._reload_bookmarks()

        xm = m.addMenu("Vi&deo")
        self._act(xm, "Zoom-in video…", "Ctrl+Shift+Z", self.on_zoom_video)
        self._act(xm, "Julia morph video…", "Ctrl+Shift+J", self.on_morph_video)

        hm = m.addMenu("&Help")
        self._act(hm, "Mouse controls", None, self.on_help)

    def _act(self, menu, text, shortcut, fn):
        a = QAction(text, self)
        if shortcut:
            a.setShortcut(QKeySequence(shortcut))
        a.triggered.connect(fn)
        menu.addAction(a)
        return a

    def _build_statusbar(self):
        sb = self.statusBar()
        self.coord_label = QLabel("")
        self.info_label = QLabel("")
        self.pbar = QProgressBar(); self.pbar.setMaximumWidth(160)
        self.pbar.setRange(0, 1000); self.pbar.hide()
        self.cancel_btn = QPushButton("Stop"); self.cancel_btn.setFixedWidth(44)
        self.cancel_btn.clicked.connect(self.cancel_render); self.cancel_btn.hide()
        sb.addWidget(self.coord_label, 1)
        sb.addPermanentWidget(self.info_label)
        sb.addPermanentWidget(self.pbar)
        sb.addPermanentWidget(self.cancel_btn)

    # ------------------------------------------------------------------ rendering
    def request_render(self):
        self.cancel_render()
        self.gen += 1
        W = max(64, self.canvas.width())
        H = max(48, self.canvas.height())
        ss = int(self.ss_cb.currentText())
        self.thread = RenderThread(self.settings, self.view, self.palette_obj,
                                   self.cs, W, H, ss, self.gen)
        self.thread.preview_ready.connect(self._on_preview)
        self.thread.field_ready.connect(self._on_field)
        self.thread.progressed.connect(self._on_progress)
        self._t0 = time.time()
        self.pbar.setValue(0); self.pbar.show(); self.cancel_btn.show()
        self.info_label.setText(self._info_text() + "   rendering…")
        self.thread.start()

    def cancel_render(self):
        if self.thread is not None:
            self.thread.cancel.cancel()
            self.thread.wait(3000)
            self.thread = None

    def _on_preview(self, rgb, gen):
        if gen == self.gen:
            self.canvas.set_image(rgb)

    def _on_progress(self, p, gen):
        if gen == self.gen:
            self.pbar.setValue(int(p * 1000))

    def _on_field(self, res, rgb, gen):
        if gen != self.gen:
            return
        self.cached = res
        self.canvas.set_image(rgb)
        self.pbar.hide(); self.cancel_btn.hide()
        dt = time.time() - self._t0
        self.info_label.setText(self._info_text() + f"   {dt:.2f}s")

    def recolor(self):
        """Instant palette/density/offset update from the cached field."""
        if self.cached is None:
            return
        rgb = eng.downsample_rgb(eng.colorize(self.cached, self.palette_obj, self.cs),
                                 self.cached.supersample)
        self.canvas.set_image(rgb)

    def _info_text(self) -> str:
        mag = self.view.magnification(self.settings.formula.default_span)
        it = self.settings.effective_max_iter(self.view)
        return (f"center {self.view.center.real:.15g}{self.view.center.imag:+.15g}i   "
                f"span {self.view.span:.3e}   mag {mag:.3e}   iters {it}")

    # ------------------------------------------------------------------ navigation
    def show_cursor_coord(self, c: complex):
        self.coord_label.setText(f"{c.real:+.15f} {c.imag:+.15f}i")

    def zoom_at(self, point: complex, factor: float):
        self.push_history()
        v = self.view.zoomed(point, factor)
        v, clamped = v.clamped(max(64, self.canvas.width()))
        self.view = v
        if clamped:
            self.statusBar().showMessage(
                "float64 precision limit reached (~10\u00b9\u00b3 magnification)", 4000)
        self.request_render()

    def on_click(self, point: complex, mods):
        if mods & Qt.KeyboardModifier.ShiftModifier:
            self.zoom_at(point, 1.0 / self.zoom_click_factor)
        else:
            self.zoom_at(point, self.zoom_click_factor)

    def on_wheel(self, point: complex, factor: float):
        self.zoom_at(point, factor)

    def on_pan(self, dx: int, dy: int):
        self.push_history()
        px = self.view.pixel_size(max(64, self.canvas.width()))
        self.view = eng.ViewState(self.view.center - complex(dx * px, -dy * px),
                                  self.view.span)
        self.request_render()

    def on_right_click(self, point: complex):
        if self.settings.mode != "escape":
            self.statusBar().showMessage("Julia-from-point applies to escape-time mode", 3000)
            return
        self.push_history()
        self.settings.plane = "julia"
        self.settings.julia_c = point
        f = self.settings.formula
        self.view = eng.ViewState(0j, f.default_span)
        self.plane_cb.setCurrentIndex(1)
        self.julia_edit.setText(self._fmt_c(point))
        self.statusBar().showMessage(f"Julia set for c = {self._fmt_c(point)}", 4000)
        self.request_render()

    def on_home(self):
        self.push_history()
        if self.settings.mode == "newton":
            n = self.settings.newton
            self.view = eng.ViewState(n.default_center, n.default_span)
        else:
            f = self.settings.formula
            self.view = (eng.ViewState(0j, f.default_span)
                         if self.settings.plane == "julia"
                         else eng.ViewState(f.default_center, f.default_span))
        self.request_render()

    # ------------------------------------------------------------------ history
    def _snapshot(self) -> dict:
        return eng.location_to_dict(self.settings, self.view, self.palette_obj, self.cs)

    def push_history(self):
        self.history = self.history[:self.hist_pos + 1]
        self.history.append(self._snapshot())
        self.hist_pos = len(self.history) - 1

    def _apply_snapshot(self, d: dict):
        s, v, p, cs = eng.location_from_dict(d)
        self.settings, self.view, self.palette_obj, self.cs = s, v, p, cs
        self._sync_widgets_from_state()
        self.request_render()

    def on_back(self):
        if self.hist_pos > 0:
            if self.hist_pos == len(self.history) - 1:
                self.history[self.hist_pos] = self._snapshot()
            self.hist_pos -= 1
            self._apply_snapshot(self.history[self.hist_pos])

    def on_forward(self):
        if self.hist_pos < len(self.history) - 1:
            self.hist_pos += 1
            self._apply_snapshot(self.history[self.hist_pos])

    def _sync_widgets_from_state(self):
        for wdg in (self.mode_cb, self.plane_cb, self.julia_edit, self.iter_spin,
                    self.auto_iter, self.density, self.offset_slider, self.log_cb):
            wdg.blockSignals(True)
        self.mode_cb.setCurrentIndex(1 if self.settings.mode == "newton" else 0)
        self.plane_cb.setCurrentIndex(1 if self.settings.plane == "julia" else 0)
        self.julia_edit.setText(self._fmt_c(self.settings.julia_c))
        self.iter_spin.setValue(self.settings.max_iter)
        self.auto_iter.setChecked(self.settings.auto_iter)
        self.density.setValue(self.cs.density)
        self.offset_slider.setValue(int(self.cs.offset) % 255)
        self.log_cb.setChecked(self.cs.log_mode)
        for wdg in (self.mode_cb, self.plane_cb, self.julia_edit, self.iter_spin,
                    self.auto_iter, self.density, self.offset_slider, self.log_cb):
            wdg.blockSignals(False)
        if self.palette_cb.findText(self.palette_obj.name) < 0:
            self.palettes.append(self.palette_obj)
            self.palette_cb.blockSignals(True)
            self.palette_cb.addItem(self.palette_obj.name)
            self.palette_cb.blockSignals(False)
        self.palette_cb.blockSignals(True)
        self.palette_cb.setCurrentText(self.palette_obj.name)
        self.palette_cb.blockSignals(False)
        self._sync_dock_enabled()

    def _sync_dock_enabled(self):
        esc = self.settings.mode == "escape"
        self.formula_cb.setEnabled(esc)
        self.plane_cb.setEnabled(esc)
        self.julia_edit.setEnabled(esc and self.settings.plane == "julia")
        self.newton_cb.setEnabled(not esc)

    # ------------------------------------------------------------------ dock handlers
    @staticmethod
    def _fmt_c(c: complex) -> str:
        return f"{c.real:+.12f}{c.imag:+.12f}j"

    def on_mode_changed(self, idx):
        self.push_history()
        self.settings.mode = "newton" if idx == 1 else "escape"
        self._sync_dock_enabled()
        self.on_home()

    def on_formula_changed(self, idx):
        self.push_history()
        self.settings.formula = self.escape_formulas[idx]
        self.settings.julia_c = self.settings.formula.default_julia_c
        self.julia_edit.setText(self._fmt_c(self.settings.julia_c))
        self.on_home()

    def on_newton_changed(self, idx):
        self.push_history()
        self.settings.newton = self.newton_formulas[idx]
        self.on_home()

    def on_plane_changed(self, idx):
        self.push_history()
        self.settings.plane = "julia" if idx == 1 else "mandelbrot"
        self._sync_dock_enabled()
        self.on_home()

    def on_julia_c_edited(self):
        try:
            self.settings.julia_c = complex(self.julia_edit.text().replace(" ", ""))
            if self.settings.plane == "julia":
                self.request_render()
        except ValueError:
            self.statusBar().showMessage("c must look like -0.7269+0.1889j", 3000)

    def on_iters_changed(self, *_):
        self.settings.max_iter = self.iter_spin.value()
        self.settings.auto_iter = self.auto_iter.isChecked()
        self.request_render()

    def on_new_formula(self):
        dlg = FormulaDialog(self)
        if not dlg.exec():
            return
        if dlg.escape_result is not None:
            self.escape_formulas.append(dlg.escape_result)
            self.formula_cb.addItem(dlg.escape_result.name)
            self.mode_cb.setCurrentIndex(0)
            self.formula_cb.setCurrentIndex(len(self.escape_formulas) - 1)
        elif dlg.newton_result is not None:
            self.newton_formulas.append(dlg.newton_result)
            self.newton_cb.addItem(dlg.newton_result.name)
            self.mode_cb.setCurrentIndex(1)
            self.newton_cb.setCurrentIndex(len(self.newton_formulas) - 1)

    def on_palette_changed(self, idx):
        self.palette_obj = self.palettes[idx]
        self.recolor()

    def on_edit_palette(self):
        dlg = PaletteEditor(self, self.palette_obj)
        if dlg.exec() and dlg.result_palette is not None:
            self.palette_obj = dlg.result_palette
            if self.palette_cb.findText(self.palette_obj.name) < 0:
                self.palettes.append(self.palette_obj)
                self.palette_cb.addItem(self.palette_obj.name)
            self.palette_cb.setCurrentText(self.palette_obj.name)
            self.recolor()

    def on_color_changed(self, *_):
        self.cs.density = self.density.value()
        self.cs.offset = float(self.offset_slider.value())
        self.cs.log_mode = self.log_cb.isChecked()
        self.recolor()

    def on_cycle_toggled(self, on):
        if on:
            self._last_cycle = time.time()
            self.cycle_timer.start()
        else:
            self.cycle_timer.stop()

    def _cycle_tick(self):
        now = time.time()
        self.cs.offset = (self.cs.offset + self.cs.cycle_speed *
                          (now - self._last_cycle)) % 255.0
        self._last_cycle = now
        self.offset_slider.blockSignals(True)
        self.offset_slider.setValue(int(self.cs.offset))
        self.offset_slider.blockSignals(False)
        self.recolor()

    # ------------------------------------------------------------------ file / video actions
    def on_save_image(self):
        SaveImageDialog(self, copy.deepcopy(self.settings), self.view,
                        self.palette_obj, copy.deepcopy(self.cs)).exec()

    def on_save_location(self):
        path, _ = QFileDialog.getSaveFileName(self, "Save location", LOCATIONS_DIR,
                                              "Location (*.json)")
        if path:
            eng.save_location(path, self.settings, self.view, self.palette_obj,
                              self.cs, os.path.basename(path))
            self._reload_bookmarks()

    def on_load_location(self):
        path, _ = QFileDialog.getOpenFileName(self, "Load location", LOCATIONS_DIR,
                                              "Location (*.json)")
        if path:
            self._load_location_file(path)

    def _load_location_file(self, path):
        try:
            s, v, p, cs = eng.load_location(path)
        except Exception as e:
            QMessageBox.warning(self, "Load failed", str(e))
            return
        self.push_history()
        self.settings, self.view, self.palette_obj, self.cs = s, v, p, cs
        self._sync_widgets_from_state()
        self.request_render()

    def on_save_bookmark(self):
        name, ok = QInputDialog.getText(self, "Save bookmark", "Name:")
        if not ok or not name.strip():
            return
        os.makedirs(LOCATIONS_DIR, exist_ok=True)
        safe = "".join(ch if ch.isalnum() or ch in "-_ " else "_" for ch in name.strip())
        eng.save_location(os.path.join(LOCATIONS_DIR, safe + ".json"),
                          self.settings, self.view, self.palette_obj, self.cs, name)
        self._reload_bookmarks()

    def _reload_bookmarks(self):
        acts = self.bookmarks_menu.actions()
        for a in acts[2:]:
            self.bookmarks_menu.removeAction(a)
        os.makedirs(LOCATIONS_DIR, exist_ok=True)
        for path in sorted(glob.glob(os.path.join(LOCATIONS_DIR, "*.json"))):
            name = os.path.splitext(os.path.basename(path))[0]
            self._act(self.bookmarks_menu, name, None,
                      lambda _=False, p=path: self._load_location_file(p))

    def on_zoom_video(self):
        ZoomVideoDialog(self, copy.deepcopy(self.settings), self.view,
                        self.palette_obj, copy.deepcopy(self.cs),
                        self.settings.formula.default_span).exec()

    def on_morph_video(self):
        if self.settings.mode != "escape":
            QMessageBox.information(self, "Julia morph",
                                    "Julia morph videos apply to escape-time mode.")
            return
        JuliaMorphDialog(self, copy.deepcopy(self.settings), self.view,
                         self.palette_obj, copy.deepcopy(self.cs)).exec()

    def on_help(self):
        QMessageBox.information(self, "Mouse controls", self.__doc__ or __doc__)

    # ------------------------------------------------------------------ events
    def resizeEvent(self, ev):
        super().resizeEvent(ev)
        self.resize_timer.start()

    def closeEvent(self, ev):
        self.cancel_render()
        super().closeEvent(ev)
