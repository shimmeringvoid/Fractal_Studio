"""Render engine: view state, colorization, progress/cancel, high-res tiling,
and location (bookmark) files.

Pipeline:  kernel -> nu field (float32; -1 = interior)  ->  8-bit palette index
           ->  RGB via palette LUT.
The nu field is cached, so palette edits / cycling / density changes recolor
instantly without re-rendering.
"""
from __future__ import annotations

import json
import math
import threading
from dataclasses import dataclass, field, asdict
from typing import Callable, Optional, Tuple

import numpy as np

from .formulas import (EscapeFormula, NewtonFormula, PRESETS, NEWTON_PRESETS,
                       render_escape_strip, render_newton_strip)
from .palette import Palette

FLOAT64_MIN_PIXEL = 5e-17     # ~10^13 magnification limit before float64 pixelates


# ----------------------------------------------------------------------------- view + settings

@dataclass
class ViewState:
    """Where we are in the complex plane."""
    center: complex = -0.5 + 0j
    span: float = 3.5                     # width of the view in complex units

    def magnification(self, base_span: float = 3.5) -> float:
        return base_span / self.span

    def pixel_size(self, width: int) -> float:
        return self.span / width

    def complex_at(self, px: float, py: float, width: int, height: int) -> complex:
        """Complex coordinate of a pixel (py measured from the top)."""
        dx = self.span / width
        xmin = self.center.real - self.span / 2.0
        ymax = self.center.imag + dx * height / 2.0
        return complex(xmin + px * dx, ymax - py * dx)

    def zoomed(self, point: complex, factor: float) -> "ViewState":
        """Zoom by `factor` keeping `point` fixed on screen (wheel-style)."""
        new_span = self.span / factor
        new_center = point + (self.center - point) / factor
        return ViewState(new_center, new_span)

    def zoomed_centered(self, point: complex, factor: float) -> "ViewState":
        """Zoom by `factor` with `point` becoming the new center (click-style)."""
        return ViewState(point, self.span / factor)

    def clamped(self, width: int) -> Tuple["ViewState", bool]:
        """Enforce the float64 depth limit. Returns (state, was_clamped)."""
        if self.span / width < FLOAT64_MIN_PIXEL:
            return ViewState(self.center, FLOAT64_MIN_PIXEL * width), True
        return self, False


@dataclass
class ColorSettings:
    density: float = 4.0        # palette indices advanced per iteration
    offset: float = 0.0         # cycling offset (indices)
    log_mode: bool = False      # compress deep-zoom iteration ranges
    cycle_speed: float = 40.0   # indices/second when cycling animation is on


@dataclass
class RenderSettings:
    mode: str = "escape"                 # "escape" | "newton"
    plane: str = "mandelbrot"            # "mandelbrot" | "julia"  (escape mode)
    formula: EscapeFormula = field(default_factory=lambda: PRESETS[0])
    newton: NewtonFormula = field(default_factory=lambda: NEWTON_PRESETS[0])
    julia_c: complex = -0.7269 + 0.1889j
    max_iter: int = 300
    auto_iter: bool = True
    supersample: int = 1                 # on-screen SS (video/stills set their own)

    def effective_max_iter(self, view: ViewState) -> int:
        if not self.auto_iter:
            return self.max_iter
        mag = max(view.magnification(self.formula.default_span), 1.0)
        auto = 100 + 140 * math.log10(1.0 + mag) ** 1.25
        return int(min(100_000, max(self.max_iter, auto)))


# ----------------------------------------------------------------------------- colorization

def colorize(result: "RenderResult", palette: Palette, cs: ColorSettings) -> np.ndarray:
    """nu field(s) -> RGB uint8 image using the 256-entry palette."""
    colors = palette.rotated(int(round(cs.offset)))
    if result.mode == "newton":
        return _colorize_newton(result.root, result.nu, colors, cs, result.n_roots)
    return _colorize_escape(result.nu, colors, cs)


def _colorize_escape(nu: np.ndarray, colors: np.ndarray, cs: ColorSettings) -> np.ndarray:
    idx = np.zeros(nu.shape, dtype=np.uint16)
    esc = nu >= 0.0
    v = nu[esc].astype(np.float64)
    if cs.log_mode:
        v = np.log1p(v) * 32.0
    v = v * cs.density + cs.offset - int(round(cs.offset))   # fractional part of offset
    idx[esc] = 1 + np.mod(v, 255.0).astype(np.uint16)
    return colors[idx]


def _colorize_newton(root: np.ndarray, nu: np.ndarray, colors: np.ndarray,
                     cs: ColorSettings, n_roots: int) -> np.ndarray:
    idx = np.zeros(root.shape, dtype=np.uint16)
    n_roots = max(int(n_roots), 1)
    band = 255 // n_roots
    conv = root >= 0
    v = nu[conv].astype(np.float64) * cs.density
    shade = np.mod(v, band).astype(np.uint16)
    idx[conv] = 1 + root[conv].astype(np.uint16) * band + shade
    return colors[np.minimum(idx, 255)]


def downsample_rgb(rgb: np.ndarray, ss: int) -> np.ndarray:
    """Box-filter an (H*ss, W*ss, 3) image down to (H, W, 3)."""
    if ss <= 1:
        return rgb
    H, W = rgb.shape[0] // ss, rgb.shape[1] // ss
    r = rgb[:H * ss, :W * ss].reshape(H, ss, W, ss, 3).astype(np.uint32)
    return (r.mean(axis=(1, 3)) + 0.5).astype(np.uint8)


# ----------------------------------------------------------------------------- render results & jobs

@dataclass
class RenderResult:
    mode: str
    nu: np.ndarray                        # (H, W) float32
    root: Optional[np.ndarray] = None     # (H, W) int16 for newton
    n_roots: int = 0
    view: Optional[ViewState] = None
    supersample: int = 1


class CancelToken:
    def __init__(self):
        self._ev = threading.Event()

    def cancel(self):
        self._ev.set()

    @property
    def cancelled(self) -> bool:
        return self._ev.is_set()


def render_field(settings: RenderSettings, view: ViewState, width: int, height: int,
                 supersample: int = 1,
                 progress: Optional[Callable[[float], None]] = None,
                 cancel: Optional[CancelToken] = None,
                 strip_rows: int = 64) -> Optional[RenderResult]:
    """Render the nu field (and root ids for newton) at width*ss x height*ss.
    Returns None if cancelled. Strips give progress + cancellation points."""
    ss = max(1, int(supersample))
    W, H = width * ss, height * ss
    dx = view.span / W
    xmin = view.center.real - view.span / 2.0
    ymin = view.center.imag - dx * H / 2.0
    max_iter = settings.effective_max_iter(view)

    nu = np.empty((H, W), dtype=np.float32)
    root = np.empty((H, W), dtype=np.int16) if settings.mode == "newton" else None

    y = 0
    while y < H:
        if cancel is not None and cancel.cancelled:
            return None
        y1 = min(H, y + strip_rows * ss)
        if settings.mode == "newton":
            render_newton_strip(settings.newton, xmin, ymin, dx, dx, W, y, y1,
                                max_iter, root[y:y1], nu[y:y1])
        else:
            julia = settings.plane == "julia"
            render_escape_strip(settings.formula, xmin, ymin, dx, dx, W, y, y1,
                                julia, settings.julia_c, max_iter, nu[y:y1])
        y = y1
        if progress is not None:
            progress(y / H)

    # kernels index rows bottom-up in imag; flip so row 0 is the top of the image
    nu = nu[::-1].copy()
    if root is not None:
        root = root[::-1].copy()
    return RenderResult(settings.mode, nu, root,
                        settings.newton.n_roots() if settings.mode == "newton" else 0,
                        view, ss)


def render_image(settings: RenderSettings, view: ViewState, palette: Palette,
                 cs: ColorSettings, width: int, height: int, supersample: int = 1,
                 progress: Optional[Callable[[float], None]] = None,
                 cancel: Optional[CancelToken] = None) -> Optional[np.ndarray]:
    """Full pipeline to an RGB uint8 (height, width, 3) image."""
    res = render_field(settings, view, width, height, supersample, progress, cancel)
    if res is None:
        return None
    rgb = colorize(res, palette, cs)
    return downsample_rgb(rgb, res.supersample)


def render_highres_tiled(settings: RenderSettings, view: ViewState, palette: Palette,
                         cs: ColorSettings, width: int, height: int, supersample: int = 2,
                         progress: Optional[Callable[[float], None]] = None,
                         cancel: Optional[CancelToken] = None,
                         tile_rows: int = 256) -> Optional[np.ndarray]:
    """Arbitrary-resolution still, rendered in horizontal bands to bound memory."""
    ss = max(1, int(supersample))
    W, H = width * ss, height * ss
    dx = view.span / W
    xmin = view.center.real - view.span / 2.0
    ymax = view.center.imag + dx * H / 2.0
    max_iter = settings.effective_max_iter(view)
    out = np.empty((height, width, 3), dtype=np.uint8)

    done = 0
    for ty in range(0, height, tile_rows):
        if cancel is not None and cancel.cancelled:
            return None
        th = min(tile_rows, height - ty)          # output rows in this band
        Hs = th * ss
        # top of this band in complex coords -> ymin for the kernel (bottom-up)
        band_ymax = ymax - (ty * ss) * dx
        band_ymin = band_ymax - Hs * dx
        nu = np.empty((Hs, W), dtype=np.float32)
        root = np.empty((Hs, W), dtype=np.int16) if settings.mode == "newton" else None
        if settings.mode == "newton":
            render_newton_strip(settings.newton, xmin, band_ymin, dx, dx, W, 0, Hs,
                                max_iter, root, nu)
        else:
            render_escape_strip(settings.formula, xmin, band_ymin, dx, dx, W, 0, Hs,
                                settings.plane == "julia", settings.julia_c, max_iter, nu)
        res = RenderResult(settings.mode, nu[::-1].copy(),
                           root[::-1].copy() if root is not None else None,
                           settings.newton.n_roots() if settings.mode == "newton" else 0,
                           view, ss)
        rgb = downsample_rgb(colorize(res, palette, cs), ss)
        out[ty:ty + th] = rgb
        done += th
        if progress is not None:
            progress(done / height)
    return out


# ----------------------------------------------------------------------------- location files

def location_to_dict(settings: RenderSettings, view: ViewState, palette: Palette,
                     cs: ColorSettings, name: str = "") -> dict:
    d = {
        "type": "fractal-studio-location",
        "version": 1,
        "name": name,
        "mode": settings.mode,
        "plane": settings.plane,
        "center_re": view.center.real,
        "center_im": view.center.imag,
        "span": view.span,
        "max_iter": settings.max_iter,
        "auto_iter": settings.auto_iter,
        "julia_c_re": settings.julia_c.real,
        "julia_c_im": settings.julia_c.imag,
        "color": {"density": cs.density, "offset": cs.offset, "log_mode": cs.log_mode,
                  "cycle_speed": cs.cycle_speed},
        "palette": palette.to_json(),
    }
    if settings.mode == "newton":
        d["newton"] = {"name": settings.newton.name,
                       "coeffs": [[c.real, c.imag] for c in
                                  (complex(x) for x in settings.newton.coeffs)],
                       "tol": settings.newton.tol}
    else:
        f = settings.formula
        d["formula"] = {"name": f.name, "text": f.text, "degree": f.degree,
                        "bailout": f.bailout, "z0": [f.z0.real, f.z0.imag],
                        "default_span": f.default_span}
    return d


def location_from_dict(d: dict) -> Tuple[RenderSettings, ViewState, Palette, ColorSettings]:
    s = RenderSettings()
    s.mode = d.get("mode", "escape")
    s.plane = d.get("plane", "mandelbrot")
    s.max_iter = int(d.get("max_iter", 300))
    s.auto_iter = bool(d.get("auto_iter", True))
    s.julia_c = complex(d.get("julia_c_re", -0.7269), d.get("julia_c_im", 0.1889))
    if s.mode == "newton" and "newton" in d:
        n = d["newton"]
        s.newton = NewtonFormula(n.get("name", "Newton"),
                                 [complex(a, b) for a, b in n["coeffs"]],
                                 n.get("tol", 1e-9))
    elif "formula" in d:
        f = d["formula"]
        s.formula = EscapeFormula(f.get("name", f["text"]), f["text"],
                                  f.get("degree", 2.0), f.get("bailout", 1000.0),
                                  complex(*f.get("z0", [0.0, 0.0])),
                                  default_span=f.get("default_span", 3.5))
    view = ViewState(complex(d["center_re"], d["center_im"]), float(d["span"]))
    pal = Palette.from_json(d["palette"]) if "palette" in d else Palette()
    c = d.get("color", {})
    cs = ColorSettings(c.get("density", 4.0), c.get("offset", 0.0),
                       c.get("log_mode", False), c.get("cycle_speed", 40.0))
    return s, view, pal, cs


def save_location(path: str, settings: RenderSettings, view: ViewState,
                  palette: Palette, cs: ColorSettings, name: str = "") -> None:
    with open(path, "w") as f:
        json.dump(location_to_dict(settings, view, palette, cs, name), f, indent=1)


def load_location(path: str):
    with open(path) as f:
        return location_from_dict(json.load(f))
