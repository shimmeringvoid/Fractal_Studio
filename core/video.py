"""Video rendering.

Zoom videos
    You specify zoom rate (x per second) and either duration or the start/end
    magnification; the per-frame factor is derived (rate ** (1/fps)).  Frames
    are written straight into an H.264 MP4 (imageio-ffmpeg), optionally also
    kept as a PNG sequence for editing with a music track later.
    Default output: 3840x2160 @ 30 fps, 2x2 supersampling, rate 2.0x/s.

Julia morph videos
    The Julia constant c travels along a parametrized path:
        circle  -- c(t) = c0 + r * exp(2*pi*i*(t + phase)); closes exactly, so
                   the clip loops seamlessly.
        spiral  -- c(t) = c0 + r * s(t) * exp(2*pi*i*turns*t), s ramping 0 -> 1.
        line    -- c(t) between two constants with smoothstep easing
                   (optionally there-and-back so it also loops).
    A morph can be combined with a zoom so the fractal "wafts" while diving.
"""
from __future__ import annotations

import math
import os
from dataclasses import dataclass, field
from typing import Callable, Optional

import imageio.v2 as imageio
import numpy as np

from .engine import (CancelToken, ColorSettings, RenderSettings, ViewState,
                     render_frame_blended)
from .palette import Palette

RESOLUTIONS = {
    "1080p (1920x1080)": (1920, 1080),
    "1440p (2560x1440)": (2560, 1440),
    "4K (3840x2160)": (3840, 2160),
    "8K (7680x4320)": (7680, 4320),
}


# ----------------------------------------------------------------------------- specs

@dataclass
class ZoomVideoSpec:
    end_view: ViewState                    # the deep point you dove to
    start_span: float                      # usually the formula's home span
    rate_per_sec: float = 2.0              # magnification factor per second
    fps: int = 30
    width: int = 3840
    height: int = 2160
    supersample: int = 2
    hold_seconds: float = 1.0              # linger on the final frame
    cycle_colors: bool = False             # palette cycling during the zoom
    cycle_speed: float = 40.0              # indices per second
    png_dir: Optional[str] = None          # also dump frames here if set
    crf: int = 20                          # H.264 quality (lower = better/larger)
    preset: str = "slow"

    def n_zoom_frames(self) -> int:
        total_mag = max(self.start_span / self.end_view.span, 1.0 + 1e-9)
        seconds = math.log(total_mag) / math.log(self.rate_per_sec)
        return max(2, int(round(seconds * self.fps)) + 1)

    def duration_seconds(self) -> float:
        return self.n_zoom_frames() / self.fps + self.hold_seconds


@dataclass
class JuliaMorphSpec:
    c0: complex
    path: str = "circle"                   # "circle" | "spiral" | "line"
    radius: float = 0.02
    turns: float = 1.0                     # circle: revolutions; spiral: windings
    phase: float = 0.0
    c1: complex = 0j                       # line target
    there_and_back: bool = True            # line only: return for a seamless loop
    duration: float = 12.0
    fps: int = 30
    width: int = 3840
    height: int = 2160
    supersample: int = 2
    view: ViewState = field(default_factory=lambda: ViewState(0j, 3.4))
    zoom_end_span: Optional[float] = None  # set to combine morph with a zoom
    cycle_colors: bool = False
    cycle_speed: float = 40.0
    png_dir: Optional[str] = None
    crf: int = 20
    preset: str = "slow"

    def n_frames(self) -> int:
        return max(2, int(round(self.duration * self.fps)))

    def c_at(self, t: float) -> complex:
        """t in [0, 1)."""
        if self.path == "circle":
            ang = 2.0 * math.pi * (self.turns * t + self.phase)
            return self.c0 + self.radius * complex(math.cos(ang), math.sin(ang))
        if self.path == "spiral":
            s = _smoothstep(t)
            ang = 2.0 * math.pi * (self.turns * t + self.phase)
            return self.c0 + self.radius * s * complex(math.cos(ang), math.sin(ang))
        # line
        if self.there_and_back:
            u = 1.0 - abs(2.0 * t - 1.0)        # triangle 0->1->0
        else:
            u = t
        return self.c0 + (self.c1 - self.c0) * _smoothstep(u)


def _smoothstep(t: float) -> float:
    t = min(max(t, 0.0), 1.0)
    return t * t * (3.0 - 2.0 * t)


# ----------------------------------------------------------------------------- writer

class FrameCollector:
    """Writer-compatible sink that keeps frames in memory (for in-app previews,
    which deliberately avoid video files and system codecs entirely)."""

    def __init__(self):
        self.frames: list = []

    def add(self, frame):
        self.frames.append(frame)

    def close(self):
        pass


class _Writer:
    """H.264 writer.

    Fractal frames are maximum-entropy content (every pixel differs; palette
    cycling defeats temporal prediction), so a fixed quality scale produces
    absurd bitrates -- 4K fractal video encoded at imageio's `quality=8`
    lands near 600 Mbps, which is ~6x 4K Blu-ray and unplayable on most
    machines. Encode by CRF instead: it targets visual quality and lets the
    bitrate fall where it must. CRF 18 is visually transparent; 20-23 is
    smaller and still excellent.
    """

    def __init__(self, path: str, fps: int, png_dir: Optional[str],
                 crf: int = 20, preset: str = "slow"):
        os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
        self.w = imageio.get_writer(
            path, fps=fps, codec="libx264", macro_block_size=8,
            pixelformat="yuv420p",
            output_params=["-crf", str(crf), "-preset", preset])
        self.png_dir = png_dir
        if png_dir:
            os.makedirs(png_dir, exist_ok=True)
        self.i = 0

    def add(self, frame: np.ndarray):
        self.w.append_data(frame)
        if self.png_dir:
            imageio.imwrite(os.path.join(self.png_dir, f"frame_{self.i:06d}.png"), frame)
        self.i += 1

    def close(self):
        self.w.close()


# ----------------------------------------------------------------------------- renderers

def render_zoom_video(spec: ZoomVideoSpec, settings: RenderSettings, palette: Palette,
                      cs: ColorSettings, out_path: str,
                      progress: Optional[Callable[[float, str], None]] = None,
                      cancel: Optional[CancelToken] = None, writer=None) -> bool:
    """Render a zoom-in video ending at spec.end_view. Returns False if cancelled.
    Pass a FrameCollector as `writer` to render in memory instead of to a file."""
    n = spec.n_zoom_frames()
    per_frame = spec.rate_per_sec ** (1.0 / spec.fps)
    if writer is None:
        writer = _Writer(out_path, spec.fps, spec.png_dir, spec.crf, spec.preset)
    base_offset = cs.offset
    try:
        last = None
        for k in range(n):
            if cancel is not None and cancel.cancelled:
                return False
            span = spec.start_span / (per_frame ** k)
            span = max(span, spec.end_view.span)
            view = ViewState(spec.end_view.center, span)
            fcs = ColorSettings(cs.density,
                                base_offset + (spec.cycle_speed * k / spec.fps
                                               if spec.cycle_colors else 0.0),
                                cs.log_mode, cs.cycle_speed)
            frame = render_frame_blended(settings, view, palette, fcs,
                                         spec.width, spec.height, spec.supersample,
                                         cancel=cancel)
            if frame is None:
                return False
            writer.add(frame)
            last = frame
            if progress is not None:
                progress((k + 1) / n, f"frame {k + 1}/{n}  span={span:.3e}")
        for _ in range(int(round(spec.hold_seconds * spec.fps))):
            if last is not None:
                writer.add(last)
        return True
    finally:
        writer.close()


def render_julia_morph_video(spec: JuliaMorphSpec, settings: RenderSettings,
                             palette: Palette, cs: ColorSettings, out_path: str,
                             progress: Optional[Callable[[float, str], None]] = None,
                             cancel: Optional[CancelToken] = None, writer=None) -> bool:
    """Render a Julia morph (optionally combined with a zoom).
    Pass a FrameCollector as `writer` to render in memory instead of to a file."""
    n = spec.n_frames()
    if writer is None:
        writer = _Writer(out_path, spec.fps, spec.png_dir, spec.crf, spec.preset)
    base_offset = cs.offset
    span0 = spec.view.span
    span1 = spec.zoom_end_span if spec.zoom_end_span else span0
    try:
        for k in range(n):
            if cancel is not None and cancel.cancelled:
                return False
            t = k / n                      # endpoint excluded -> seamless loops
            s = RenderSettings(mode="escape", plane="julia",
                               formula=settings.formula, newton=settings.newton,
                               julia_c=spec.c_at(t), max_iter=settings.max_iter,
                               auto_iter=settings.auto_iter)
            span = span0 * (span1 / span0) ** _smoothstep(t) if span1 != span0 else span0
            view = ViewState(spec.view.center, span)
            fcs = ColorSettings(cs.density,
                                base_offset + (spec.cycle_speed * k / spec.fps
                                               if spec.cycle_colors else 0.0),
                                cs.log_mode, cs.cycle_speed)
            frame = render_frame_blended(s, view, palette, fcs,
                                         spec.width, spec.height, spec.supersample,
                                         cancel=cancel)
            if frame is None:
                return False
            writer.add(frame)
            if progress is not None:
                progress((k + 1) / n, f"frame {k + 1}/{n}  c={s.julia_c:.6f}")
        return True
    finally:
        writer.close()
