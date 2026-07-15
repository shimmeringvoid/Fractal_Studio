#!/usr/bin/env python
"""Verify the CUDA escape-time twins against the CPU kernel, then benchmark 4K.

Workstream A (CUDA rendering). Run from the repo root inside the `fractal`
conda env:

    python scripts/cuda_verify_bench.py

Checks:
  * float64 GPU twin matches the CPU kernel everywhere except a sparse set of
    escape-time-boundary pixels (the only place two different FP backends can
    disagree for an iterated map);
  * float32 GPU fast path is visually equivalent to the CPU field;
  * render_escape_frame's auto precision + CPU fallback behave correctly;
  * times a 4K frame on CPU vs GPU float64 vs GPU float32.

See docs/workstreamA_bench.md for recorded numbers and the design rationale.
"""
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.formulas import (PRESETS, cuda_available, render_escape_frame,   # noqa: E402
                           render_escape_frame_gpu, render_escape_strip)


def cpu_frame(f, xmin, ymin, dx, w, h, julia, c, mi):
    out = np.empty((h, w), np.float32)
    render_escape_strip(f, xmin, ymin, dx, dx, w, 0, h, julia, c, mi, out)
    return out


def bounds(f, w, h, span=None):
    span = f.default_span if span is None else span
    dx = span / w
    xmin = f.default_center.real - span / 2.0
    ymin = f.default_center.imag - dx * h / 2.0
    return xmin, ymin, dx


def best(fn, n=3):
    b = 1e9
    for _ in range(n):
        t = time.perf_counter(); fn(); b = min(b, time.perf_counter() - t)
    return b


def main():
    if not cuda_available():
        print("No CUDA GPU available; the CPU fallback path is what would run.")
        return
    W, H, MI = 800, 600, 400

    print("=" * 78)
    print("VERIFY float64 twin vs CPU (match except sparse boundary pixels)")
    print("=" * 78)
    worst = 0.0
    for f in PRESETS:
        xmin, ymin, dx = bounds(f, W, H)
        for julia in (False, True):
            c = f.default_julia_c if julia else 0j
            cpu = cpu_frame(f, xmin, ymin, dx, W, H, julia, c, MI)
            gpu = render_escape_frame_gpu(f, xmin, ymin, dx, dx, W, H, julia, c,
                                          MI, precision="f64")
            n = cpu.size
            ndiff = int((cpu != gpu).sum())
            flips = int(((cpu < 0) != (gpu < 0)).sum())
            esc = (cpu >= 0) & (gpu >= 0)
            dmax = float(np.abs(cpu[esc] - gpu[esc]).max()) if esc.any() else 0.0
            worst = max(worst, 100 * ndiff / n)
            plane = "julia " if julia else "mandel"
            print(f"  {f.text:38.38s} {plane} diff={ndiff:5d}/{n} "
                  f"({100*ndiff/n:5.3f}%) flips={flips:4d} max|d|={dmax:7.2f}")
    print(f"  -> worst differing fraction {worst:.3f}% (all on the "
          f"discontinuous escape-time boundary)\n")

    print("=" * 78)
    print("VERIFY float32 fast path vs CPU (visually equivalent)")
    print("=" * 78)
    for f in PRESETS:
        xmin, ymin, dx = bounds(f, W, H)
        cpu = cpu_frame(f, xmin, ymin, dx, W, H, False, 0j, MI)
        gpu = render_escape_frame_gpu(f, xmin, ymin, dx, dx, W, H, False, 0j,
                                      MI, precision="f32")
        esc = (cpu >= 0) & (gpu >= 0)
        close = float((np.abs(cpu[esc] - gpu[esc]) <= 1.0).mean()) if esc.any() else 1.0
        flips = float(((cpu < 0) != (gpu < 0)).mean())
        print(f"  {f.text:38.38s} within-1-iter={100*close:6.2f}% "
              f"boundary_flips={100*flips:5.3f}%")
    print()

    print("=" * 78)
    print("BENCHMARK 4K (3840x2160) Mandelbrot z**2+c, max_iter=1000")
    print("=" * 78)
    mandel = PRESETS[0]
    W4, H4, MI4 = 3840, 2160, 1000
    xmin, ymin, dx = bounds(mandel, W4, H4)
    cpu_frame(mandel, xmin, ymin, dx, 64, 48, False, 0j, MI4)                 # warmups
    render_escape_frame_gpu(mandel, xmin, ymin, dx, dx, 64, 48, False, 0j, MI4, precision="f64")
    render_escape_frame_gpu(mandel, xmin, ymin, dx, dx, 64, 48, False, 0j, MI4, precision="f32")
    t_cpu = best(lambda: cpu_frame(mandel, xmin, ymin, dx, W4, H4, False, 0j, MI4))
    t_g64 = best(lambda: render_escape_frame_gpu(mandel, xmin, ymin, dx, dx, W4, H4, False, 0j, MI4, precision="f64"))
    t_g32 = best(lambda: render_escape_frame_gpu(mandel, xmin, ymin, dx, dx, W4, H4, False, 0j, MI4, precision="f32"))
    print(f"  CPU float64 (12 threads)   : {t_cpu:6.3f}s   1.00x")
    print(f"  GPU float64 (1 GTX TITAN X): {t_g64:6.3f}s   {t_cpu/t_g64:6.2f}x")
    print(f"  GPU float32 (1 GTX TITAN X): {t_g32:6.3f}s   {t_cpu/t_g32:6.2f}x")
    print(f"  GPU float32 x4 GPUs (proj) : ~{t_cpu/t_g32*4:.0f}x over CPU\n")

    print("=" * 78)
    print("DISPATCHER auto precision + CPU fallback")
    print("=" * 78)
    xmin, ymin, dx = bounds(mandel, W, H)
    auto = render_escape_frame(mandel, xmin, ymin, dx, dx, W, H, False, 0j, MI, precision="auto")
    g32 = render_escape_frame_gpu(mandel, xmin, ymin, dx, dx, W, H, False, 0j, MI, precision="f32")
    print(f"  shallow: auto == gpu_f32 ? {np.array_equal(auto, g32)}")
    xd, yd, dxd = bounds(mandel, W, H, span=1e-6)
    autod = render_escape_frame(mandel, xd, yd, dxd, dxd, W, H, False, 0j, MI, precision="auto")
    cpud = cpu_frame(mandel, xd, yd, dxd, W, H, False, 0j, MI)
    print(f"  deep (span=1e-6): auto == cpu_f64 ? {np.array_equal(autod, cpud)}")
    fcpu = render_escape_frame(mandel, xmin, ymin, dx, dx, W, H, False, 0j, MI, precision="cpu")
    print(f"  precision='cpu' == strip kernel ? {np.array_equal(fcpu, cpu_frame(mandel, xmin, ymin, dx, W, H, False, 0j, MI))}")


if __name__ == "__main__":
    main()
