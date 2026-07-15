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

## Not done yet (next steps)

* **Wire the GPU path into `core/engine.py` / `core/video.py`.** The kernel and
  fallback exist and are tested, but `render_field` still uses the CPU strip
  kernel. The clean integration: full-frame GPU render for video frames and
  stills (respecting `auto` precision), keeping the CPU strip path for the
  interactive progressive/cancellable preview.
* **Shard frame batches across the 4 GPUs** (one CUDA context / worker per GPU)
  -- the ~250x projection above.
* **Newton kernel CUDA twin** (same pattern; lower priority).
* Optional: perturbation/double-double for deep zoom on the GPU, to lift the
  float32 depth cap.
