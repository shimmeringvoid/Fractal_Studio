# Workstream A -- CUDA rendering: results & decisions

Status: **escape-time CUDA twin implemented, verified, and benchmarked** on
monster (4x GTX TITAN X, Maxwell sm_52). Reproduce with:

```
conda activate fractal
python scripts/cuda_verify_bench.py
```

## What was added

`core/formulas.py` now generates `numba.cuda` twins of the escape-time kernel
from the *same* transpiled formula expression as the CPU kernel:

* `render_escape_frame_gpu(..., precision='f32'|'f64')` -- renders a whole
  frame on one GPU (one thread per pixel; frames are the unit of GPU work).
* `render_escape_frame(..., precision='auto')` -- the public entry point.
  Picks a path and **falls back to the CPU kernel automatically** (on missing
  GPU, an unsupported view, or any CUDA error -- which latches the GPU off for
  the rest of the session and warns once).
* `cuda_available()` -- cached probe.

Toolchain: numba 0.65.1 compiling through **conda `cudatoolkit=11.8`** (pinned
in `environment.yml`). numba prefers the conda toolkit over the system CUDA
12.1; 11.8 gives full sm_52 (Maxwell) support with no deprecation noise.

## The precision decision (this is the whole story on Maxwell)

The GTX TITAN X is Maxwell: **FP64 throughput is 1/32 of FP32**. Measured, 4K
frame (3840x2160), Mandelbrot z**2+c, max_iter=1000, single GPU:

| path                         | time    | vs CPU  |
|------------------------------|---------|---------|
| CPU float64 (12 threads)     | 1.26 s  | 1.0x    |
| GPU **float64** (1 Titan X)  | 4.01 s  | 0.31x (slower!) |
| GPU **float32** (1 Titan X)  | 0.020 s | **62x** |

So a naive float64 GPU port is *slower than the 2014 CPU* -- the FP64 units are
the bottleneck. The float32 path is the reason to use these GPUs at all:
~62x per GPU, and frames shard trivially across the 4 GPUs (~250x projected).

Getting real float32 speed needed one non-obvious codegen step: numba lowers
complex `**` (and any mix with float64 literals) by promoting the whole
expression to complex128, so `z**2` ran in FP64 even inside a "float32" kernel.
The f32 path therefore rewrites the formula AST -- integer powers to repeated
multiplication (`z**2` -> `z*z`) and literals cast to `complex64`/`float32`
(`_F32Specializer`). With that, `z*z` stays complex64 and the kernel is genuinely
float32.

### Cost of float32: zoom depth

float32 has ~1.2e-7 relative epsilon, so it can only resolve pixels down to
roughly **span > 1e-4** (~10^4x zoom from the home view). Deeper than that the
pixel step underflows float32 and detail collapses. `render_escape_frame`'s
`auto` mode guards on this (`_F32_MIN_SPAN = 1e-4`): shallow views -> GPU
float32; deeper views -> CPU float64 (not GPU float64, which is slower here).
Deep zooms (down to the float64 limit ~1e13) stay on the CPU for now.

## Verification (`scripts/cuda_verify_bench.py`)

**float64 twin vs CPU** -- identical except a sparse set of pixels *on the
escape-time boundary*, where the map is genuinely discontinuous and x86 (no
FMA) vs Maxwell (FMA contraction) round differently. This is the theoretical
best two different FP backends can do. Differing-pixel fraction: <=0.06% for
the polynomials, up to 2.4% for Burning Ship (lots of boundary filaments);
interiors and smooth regions match to the last float32 bit.

**float32 fast path vs CPU** -- visually equivalent: 99.7-99.9% of escaped
pixels agree within 1 iteration for the polynomial/`conj`/`lambda` formulas,
97-99% for Burning Ship and `sin`. `exp(z)+c` is the outlier (~68% within 1
iter) because `exp` blows the orbit up near the 1e4 bailout where float32 is
coarse; topology is still right (boundary flips 0.12%). Prefer float64/CPU for
`exp`-type formulas if exact smooth shading matters.

## Engine / video integration (done)

The GPU path is now wired through the whole app, all via `precision='auto'`:

* `engine.render_field` (interactive window + fast preview) -- escape mode does a
  whole-frame GPU render when `auto` resolves to a GPU path (shallow), and falls
  back to the CPU strip loop (with per-strip progress + cancel) when it resolves
  to CPU (deep zoom, or no GPU). Newton stays on the CPU.
* `engine.render_highres_tiled` (stills, video frames) -- each memory-bounding
  band now goes through `render_escape_frame` at the requested precision.
* `engine.render_frame_blended` (video frames) -- see handoff below.
* `video.render_zoom_video` / `render_julia_morph_video` use `render_frame_blended`.

Verified with a spy on `render_escape_frame_gpu`: the interactive window's two
`render_field` passes and every in-app zoom-Preview frame hit GPU-f32; a deep
(span 1e-6) interactive view makes 0 GPU calls (CPU strip, as intended).

## Pop-free f32 -> f64 zoom handoff

A zoom video crosses from GPU-f32 (shallow) to CPU-f64 (deep). The two kernels
differ, so switching precision on a single frame pops. `escape_precision_plan`
defines a span-space crossfade band -- pure f32 above `_HANDOFF_SPAN_HI` (5e-4),
pure cpu-f64 below `_HANDOFF_SPAN_LO` (1.5e-4, safely above the 1e-4 f32 floor),
and a smoothstep blend of BOTH between. `render_frame_blended` renders the band
frames at both precisions and alpha-blends the colorized RGB by those weights.
Band edges are continuous (weight 0 at HI = pure f32, weight 1 at LO = pure cpu).

Proof (`scripts/cuda_handoff_test.py`), isolating precision from zoom motion by
measuring `D(span) = mean|f32 - cpu|` at a fixed view (0-255 scale):

* `D` is real and would-be-visible: 16-35 gray levels across the band, ~25 at
  the naive hard-switch span (1e-4) -- a ~10% one-frame flash.
* Over a 2x/s zoom the band spans ~47 frames. Worst single-frame precision step:
  **hard switch 25.2  vs  crossfade 0.98** (0-255) -- the crossfade is 26x
  smaller, far below perceptual threshold. `--clip out.mp4` renders a real
  test clip crossing the handoff.

Cost: band frames render twice (f32 fast + cpu-f64 ~1.3s/4K), a one-time ~1min
for a deep zoom. Outside the band and for stills there is no extra render.

## Not done yet (next steps)

* **Shard frame batches across the 4 GPUs** (one CUDA context / worker per GPU)
  -- the ~250x projection above. Frames are independent; the current path uses
  one GPU.
* **Newton kernel CUDA twin** (same pattern; lower priority).
* Optional: perturbation/double-double for deep zoom on the GPU, to lift the
  float32 depth cap (currently deep zooms render on the CPU).
