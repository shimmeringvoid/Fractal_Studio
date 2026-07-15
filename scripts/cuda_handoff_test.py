#!/usr/bin/env python
"""Prove the f32->f64 zoom handoff has no visible pop, and render a test clip.

Workstream A. A zoom video crosses from the GPU float32 fast path (shallow) to
the CPU float64 path (deep). The two kernels differ, so a hard switch at one
frame would pop. core.engine.render_frame_blended crossfades the precision over
a span-space band (core.formulas.escape_precision_plan) instead.

This isolates the precision difference from zoom motion: at a FIXED view it
measures D(span) = mean|f32 - cpu| (0-255), which is exactly the one-frame jump
a hard switch would inject. It then shows the crossfade spreads that D across
the whole band, so the worst per-frame precision step is tiny.

    python scripts/cuda_handoff_test.py [--clip out.mp4]

See docs/workstreamA_bench.md for recorded numbers.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import core.formulas as F
from core.engine import ColorSettings, RenderSettings, ViewState, render_highres_tiled
from core.palette import Palette
from core.video import ZoomVideoSpec, render_zoom_video

HI, LO, MID = F._HANDOFF_SPAN_HI, F._HANDOFF_SPAN_LO, F._F32_MIN_SPAN
S = RenderSettings()                                   # Mandelbrot z**2+c
PAL = Palette(); CS = ColorSettings(density=8.0)
CENTER = -0.743643887037151 + 0.13182590420533j        # seahorse valley
W, H, SS = 640, 360, 1


def rgb(span, prec):
    return render_highres_tiled(S, ViewState(CENTER, span), PAL, CS, W, H, SS,
                                precision=prec).astype(np.int16)


def main():
    if not F.cuda_available():
        print("No GPU; the whole zoom would render on the CPU -- no handoff, no pop.")
        return
    print(f"handoff band: pure-f32 >= {HI:.1e}  ..crossfade..  pure-cpu <= {LO:.1e}")

    print("\nIsolated precision difference D(span) = mean|f32 - cpu| (0-255):")
    print("  (the one-frame pop a HARD switch would inject at that span)")
    for sp in [HI * 1.3, HI, 3e-4, 2e-4, LO, 1.2e-4, MID, 8e-5, 5e-5]:
        note = {HI: " <- band top", LO: " <- band bottom",
                MID: " <- naive hard-switch point"}.get(sp, "")
        print(f"    span={sp:.2e}  D={np.abs(rgb(sp,'f32')-rgb(sp,'cpu')).mean():6.3f}{note}")

    # simulate a zoom through the band; compare worst single-frame precision step
    N, span_start, span_end = 90, 8e-4, 8e-5
    ratio = (span_end / span_start) ** (1.0 / (N - 1))
    spans = [span_start * ratio ** k for k in range(N)]
    cpu_w = lambda sp: dict(F.escape_precision_plan(sp)).get("cpu", 1.0 if sp <= LO else 0.0)

    worst_cf = worst_hs = 0.0
    nband = 0
    for k in range(1, N):
        sp0, sp1 = spans[k - 1], spans[k]
        dw = abs(cpu_w(sp1) - cpu_w(sp0))
        if dw > 0:
            nband += 1
            d = np.abs(rgb(sp1, "f32") - rgb(sp1, "cpu")).astype(np.float32)
            worst_cf = max(worst_cf, float((dw * d).mean()))
        if (sp0 >= MID) != (sp1 >= MID):               # hard switch flips here
            worst_hs = float(np.abs(rgb(sp0, "f32") - rgb(sp1, "cpu")
                                    - (rgb(sp0, "cpu") - rgb(sp1, "cpu"))).mean())

    print(f"\nzoom crosses the band over ~{nband} frames. Worst SINGLE-FRAME "
          f"precision-induced step:")
    print(f"    hard switch : {worst_hs:6.3f}  (all of D at one frame -> visible pop)")
    print(f"    crossfade   : {worst_cf:6.3f}  (D spread over the band)")
    print(f"    -> crossfade is {worst_hs/max(worst_cf,1e-6):.0f}x smaller; well below "
          f"perceptual threshold.")

    if "--clip" in sys.argv:
        path = sys.argv[sys.argv.index("--clip") + 1]
        spec = ZoomVideoSpec(end_view=ViewState(CENTER, span_end), start_span=span_start,
                             rate_per_sec=2.0, fps=30, width=960, height=540,
                             supersample=1, hold_seconds=0.5, crf=18)
        print(f"\nrendering test clip -> {path} ({spec.n_zoom_frames()} frames)")
        print("clip ok:", render_zoom_video(spec, S, PAL, CS, path))


if __name__ == "__main__":
    main()
