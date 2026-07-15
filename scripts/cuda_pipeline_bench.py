#!/usr/bin/env python
"""Benchmark the sharded frame pipeline: 1 worker (sequential) vs N (all GPUs).

Workstream A. render_zoom_video renders frames across `workers` threads, each
pinned to a GPU, and encodes them in order on a single consumer. This times the
same real 4K clip at workers=1 (sequential-equivalent) and at the default
(one worker per GPU), for MP4-only and MP4+PNG output.

    python scripts/cuda_pipeline_bench.py

The clip crosses the f32->f64 handoff band so the slow CPU-f64 crossfade frames
are exercised (they must not stall the ordered encoder). See
docs/workstreamA_bench.md for recorded numbers and where the time goes.
"""
import os
import sys
import glob
import shutil
import time
import threading

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from numba import cuda
from core.engine import ColorSettings, RenderSettings, ViewState, render_frame_blended
from core.formulas import cuda_available
from core.palette import Palette
from core.video import ZoomVideoSpec, render_zoom_video

OUT = "/tmp/fractal_pipeline_bench"
CENTER = -0.743643887037151 + 0.13182590420533j


def spec(png=None, w=3840, h=2160):
    return ZoomVideoSpec(end_view=ViewState(CENTER, 1e-4), start_span=3.5,
                         rate_per_sec=3.0, fps=30, width=w, height=h,
                         supersample=1, hold_seconds=0.5, png_dir=png)


def main():
    if not cuda_available():
        print("No GPU; the pipeline runs on the CPU only -- nothing to shard.")
        return
    os.makedirs(OUT, exist_ok=True)
    s = RenderSettings(); pal = Palette(); cs = ColorSettings(density=8.0)
    n = spec().n_zoom_frames()
    ngpu = len(cuda.gpus)
    print(f"clip: 4K ss=1, {n} zoom frames (~{n/30:.0f}s) + hold, crosses handoff "
          f"band; GPUs={ngpu}")

    # warm every device + CPU kernels so we time steady state, not JIT compile
    def warm(d):
        cuda.select_device(d)
        render_frame_blended(s, ViewState(CENTER, 1e-2), pal, cs, 256, 256, 1)
    ts = [threading.Thread(target=warm, args=(d,)) for d in range(ngpu)]
    [t.start() for t in ts]; [t.join() for t in ts]
    render_frame_blended(s, ViewState(CENTER, 5e-5), pal, cs, 256, 256, 1)   # cpu f64
    render_frame_blended(s, ViewState(CENTER, 3e-4), pal, cs, 256, 256, 1)   # band
    print("warmed all devices + CPU kernels\n")

    def run(png, workers, tag):
        sp = spec(os.path.join(OUT, "png") if png else None)
        if sp.png_dir:
            shutil.rmtree(sp.png_dir, ignore_errors=True)
        t0 = time.perf_counter()
        ok = render_zoom_video(sp, s, pal, cs, os.path.join(OUT, "out.mp4"),
                               workers=workers)
        dt = time.perf_counter() - t0
        npng = len(glob.glob(os.path.join(sp.png_dir, "*.png"))) if sp.png_dir else 0
        print(f"  {tag:32} {dt:7.2f}s  ({n/dt:4.1f} fps)  ok={ok} png={npng}")
        return dt

    print("=== MP4 only ===")
    b = run(False, 1, "workers=1 (sequential)")
    a = run(False, ngpu, f"workers={ngpu} (all GPUs)")
    print(f"  -> {b/a:.2f}x\n")
    print("=== MP4 + PNG sequence ===")
    bp = run(True, 1, "workers=1 (sequential)")
    ap = run(True, ngpu, f"workers={ngpu} (all GPUs)")
    print(f"  -> {bp/ap:.2f}x")


if __name__ == "__main__":
    main()
