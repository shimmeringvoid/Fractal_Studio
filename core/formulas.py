"""Fractal formulas.

Two families:

1. Escape-time iterations  z <- f(z, c)
   * Presets (Mandelbrot, higher powers, Burning Ship, Tricorn, ...)
   * Free-text formulas in z and c ("z**3 + c*sin(z) + 0.1j"), validated with
     a whitelist AST parser and compiled into a Numba-parallel kernel.
   * Plane modes: "mandelbrot" (c = pixel, z0 = 0) or "julia" (z = pixel, c = const).

2. Newton basins  z <- z - p(z)/p'(z) for a polynomial p given by coefficients.
   Colored by which root the orbit converges to (basin of attraction), shaded
   by (smooth) convergence speed.  This is the multi-attractor mode.

Kernels write a float32 "nu" (smooth iteration value; -1 for interior /
non-converged) so recoloring never requires re-rendering.
"""
from __future__ import annotations

import ast
import math
import warnings
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
from numba import njit, prange

# ----------------------------------------------------------------------------- safe formula parsing

_ALLOWED_FUNCS = {
    "sin": "_sin", "cos": "_cos", "tan": "_tan",
    "sinh": "_sinh", "cosh": "_cosh", "tanh": "_tanh",
    "exp": "_exp", "log": "_log", "sqrt": "_sqrt",
    "asin": "_asin", "acos": "_acos", "atan": "_atan",
    "abs": "abs", "conj": "_conj", "re": "_re", "im": "_im",
}
_ALLOWED_NAMES = {"z", "c"} | set(_ALLOWED_FUNCS)
_ALLOWED_BINOPS = (ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Pow)
_ALLOWED_UNARY = (ast.USub, ast.UAdd)


class FormulaError(ValueError):
    pass


class _Validator(ast.NodeVisitor):
    def generic_visit(self, node):
        ok = (ast.Expression, ast.BinOp, ast.UnaryOp, ast.Call, ast.Name,
              ast.Constant, ast.Load) + _ALLOWED_BINOPS + _ALLOWED_UNARY
        if not isinstance(node, ok):
            raise FormulaError(f"'{type(node).__name__}' is not allowed in formulas")
        super().generic_visit(node)

    def visit_Name(self, node):
        if node.id not in _ALLOWED_NAMES:
            raise FormulaError(f"unknown name '{node.id}' (use z, c, and functions like sin, exp, conj, re, im, abs)")

    def visit_Call(self, node):
        if not isinstance(node.func, ast.Name) or node.func.id not in _ALLOWED_FUNCS:
            raise FormulaError("only these functions are allowed: " + ", ".join(sorted(_ALLOWED_FUNCS)))
        if len(node.args) != 1 or node.keywords:
            raise FormulaError(f"{node.func.id}() takes exactly one argument")
        for a in node.args:
            self.visit(a)

    def visit_Constant(self, node):
        if not isinstance(node.value, (int, float, complex)):
            raise FormulaError("only numeric constants are allowed")


class _Renamer(ast.NodeTransformer):
    def visit_Call(self, node):
        self.generic_visit(node)
        node.func = ast.Name(id=_ALLOWED_FUNCS[node.func.id], ctx=ast.Load())
        return node


def parse_formula(text: str) -> str:
    """Validate a user formula in z and c; return the transpiled expression source."""
    text = text.strip()
    if not text:
        raise FormulaError("empty formula")
    try:
        tree = ast.parse(text, mode="eval")
    except SyntaxError as e:
        raise FormulaError(f"syntax error: {e.msg}") from None
    _Validator().visit(tree)
    tree = _Renamer().visit(tree)
    ast.fix_missing_locations(tree)
    return ast.unparse(tree)


# ----------------------------------------------------------------------------- escape-time kernel codegen

_KERNEL_TEMPLATE = '''
import math
from cmath import (sin as _sin, cos as _cos, tan as _tan, sinh as _sinh,
                   cosh as _cosh, tanh as _tanh, exp as _exp, log as _log,
                   sqrt as _sqrt, asin as _asin, acos as _acos, atan as _atan)
from numba import njit, prange

@njit(inline='always', cache=False)
def _conj(z):
    return complex(z.real, -z.imag)

@njit(inline='always', cache=False)
def _re(z):
    return complex(z.real, 0.0)

@njit(inline='always', cache=False)
def _im(z):
    return complex(z.imag, 0.0)

@njit(inline='always', cache=False)
def _f(z, c):
    return {expr}

@njit(parallel=True, nogil=True, cache=False)
def kernel(xmin, ymin, dx, dy, width, y0, y1, julia, cre, cim,
           z0re, z0im, max_iter, bailout, inv_log_deg, out):
    b2 = bailout * bailout
    log_b = math.log(bailout)
    for jj in prange(y1 - y0):
        j = y0 + jj
        pim = ymin + j * dy
        for i in range(width):
            pre = xmin + i * dx
            if julia:
                z = complex(pre, pim)
                c = complex(cre, cim)
            else:
                z = complex(z0re, z0im)
                c = complex(pre, pim)
            nu = -1.0
            for k in range(max_iter):
                z = _f(z, c)
                m2 = z.real * z.real + z.imag * z.imag
                if not (m2 <= b2):          # catches > bailout, inf, and nan
                    if math.isfinite(m2) and m2 > 1.0:
                        l = 0.5 * math.log(m2)
                        t = math.log(l / log_b) * inv_log_deg
                        nu = k + 1.0 - t
                        if nu < 0.0:
                            nu = 0.0
                    else:
                        nu = k + 1.0
                    break
            out[jj, i] = nu
'''


@dataclass
class EscapeFormula:
    name: str
    text: str                      # user-facing formula text, e.g. "z**2 + c"
    degree: float = 2.0            # leading power, used for smooth coloring
    bailout: float = 1000.0
    z0: complex = 0j               # start value in mandelbrot-plane mode
    default_center: complex = -0.5 + 0j
    default_span: float = 3.5      # complex-plane width of the "home" view
    default_julia_c: complex = -0.7269 + 0.1889j

    def key(self) -> str:
        return self.text


PRESETS: List[EscapeFormula] = [
    EscapeFormula("Mandelbrot  z\u00b2 + c", "z**2 + c", 2.0),
    EscapeFormula("Cubic  z\u00b3 + c", "z**3 + c", 3.0, default_center=0j, default_span=3.0,
                  default_julia_c=0.4 + 0.25j),
    EscapeFormula("Quartic  z^4 + c", "z**4 + c", 4.0, default_center=0j, default_span=2.8,
                  default_julia_c=-0.55 + 0.4j),
    EscapeFormula("Burning Ship", "(abs(re(z)) + 1j*abs(im(z)))**2 + c", 2.0,
                  default_center=-0.45 - 0.55j, default_span=3.4, default_julia_c=-0.598 + 0.9225j),
    EscapeFormula("Tricorn", "conj(z)**2 + c", 2.0, default_center=-0.3 + 0j, default_span=4.0,
                  default_julia_c=-0.75 + 0.1j),
    EscapeFormula("Lambda  c\u00b7z(1-z)", "c*z*(1 - z)", 2.0, z0=0.5 + 0j,
                  default_center=1.0 + 0j, default_span=5.0, default_julia_c=1.0 + 0.4j),
    EscapeFormula("Sine  c\u00b7sin(z)", "c*sin(z)", 2.0, z0=1.0 + 0j, bailout=50.0,
                  default_center=0j, default_span=8.0, default_julia_c=1.0 + 0.3j),
    EscapeFormula("Exponential  exp(z) + c", "exp(z) + c", 2.0, bailout=1e4,
                  default_center=0j, default_span=8.0, default_julia_c=-0.65 + 0j),
]


class KernelCache:
    """Compiles and caches escape-time kernels per formula text."""

    def __init__(self):
        self._cache: Dict[str, object] = {}

    def get(self, formula_text: str):
        k = formula_text.strip()
        fn = self._cache.get(k)
        if fn is None:
            expr = parse_formula(k)
            ns: dict = {}
            exec(compile(_KERNEL_TEMPLATE.format(expr=expr), "<fractal-kernel>", "exec"), ns)
            fn = ns["kernel"]
            self._cache[k] = fn
        return fn


KERNELS = KernelCache()


def render_escape_strip(formula: EscapeFormula, xmin: float, ymin: float,
                        dx: float, dy: float, width: int, y0: int, y1: int,
                        julia: bool, c: complex, max_iter: int,
                        out: np.ndarray) -> None:
    """Render rows [y0, y1) of the smooth-iteration field into out[(y1-y0), width]."""
    kern = KERNELS.get(formula.text)
    deg = max(float(formula.degree), 1.0001)
    kern(xmin, ymin, dx, dy, width, y0, y1, julia,
         c.real, c.imag, formula.z0.real, formula.z0.imag,
         max_iter, formula.bailout, 1.0 / math.log(deg), out)


# ----------------------------------------------------------------------------- CUDA escape-time twin
#
# numba.cuda twins of the CPU escape-time kernel, generated from the SAME
# transpiled formula expression so their output tracks the CPU kernel. One
# thread renders one pixel of a whole frame (frames are the unit of GPU work:
# video frames are embarrassingly parallel across the 4 GPUs, and a still is
# one frame). All callers should go through render_escape_frame, which picks a
# precision and falls back to the CPU strip kernel automatically.
#
# PRECISION (this is the whole ballgame on Maxwell) --------------------------
# The GTX TITAN X is Maxwell (sm_52): its FP64 throughput is 1/32 of FP32.
# MEASURED on monster, 4K Mandelbrot, max_iter=1000 (docs/workstreamA_bench.md):
#     CPU float64 (12 threads)   1.30 s   1.0x
#     GPU float64 (1 Titan X)    4.05 s   0.32x   <- SLOWER than the CPU
#     GPU float32 (1 Titan X)    0.02 s   62x     <- the reason to use the GPU
# So we generate two kernels per formula:
#   'f64' -- complex128 throughout. Bit-faithful twin of the CPU kernel
#            (matches it everywhere except a ~0.05% sparse set of boundary
#            pixels where escape-time is genuinely discontinuous and x86-vs-
#            Maxwell rounding + FMA diverge). Correct, but slow on Maxwell.
#   'f32' -- complex64 orbit (float32). ~8.5x the CPU per GPU, but float32's
#            ~1.2e-7 relative epsilon caps usable zoom depth (~span > 1e-4);
#            deeper than that the pixel step underflows float32 and detail
#            collapses. render_escape_frame guards on this automatically.
# Pixel coordinates are always computed from float64 xmin/dx (passed in) and
# only then rounded into the orbit's precision, so f32 keeps the best possible
# float32 coordinate.
#
# Maxwell toolchain: numba compiles through the conda cudatoolkit 11.8 pinned
# in environment.yml (full sm_52 support). See CLAUDE.md, Workstream A.

# span below which float32 pixel coordinates lose too much resolution; deeper
# views fall back to the float64 path. float32 eps ~1.2e-7; keep ~25x margin.
_F32_MIN_SPAN = 1e-4


_CUDA_IMPORTS = '''
import math
from cmath import (sin as _sin, cos as _cos, tan as _tan, sinh as _sinh,
                   cosh as _cosh, tanh as _tanh, exp as _exp, log as _log,
                   sqrt as _sqrt, asin as _asin, acos as _acos, atan as _atan)
'''

# float64 twin: no casts, complex128 throughout -- identical ops to the CPU
# kernel (hence identical output bar the discontinuous-boundary pixels).
_CUDA_F64_TEMPLATE = _CUDA_IMPORTS + '''
from numba import cuda

@cuda.jit(device=True, inline=True)
def _conj(z):
    return complex(z.real, -z.imag)

@cuda.jit(device=True, inline=True)
def _re(z):
    return complex(z.real, 0.0)

@cuda.jit(device=True, inline=True)
def _im(z):
    return complex(z.imag, 0.0)

@cuda.jit(device=True, inline=True)
def _f(z, c):
    return {expr}

@cuda.jit
def kernel(xmin, ymin, dx, dy, width, height, julia, cre, cim,
           z0re, z0im, max_iter, bailout, inv_log_deg, out):
    i, j = cuda.grid(2)
    if i >= width or j >= height:
        return
    b2 = bailout * bailout
    log_b = math.log(bailout)
    pre = xmin + i * dx
    pim = ymin + j * dy
    if julia:
        z = complex(pre, pim)
        c = complex(cre, cim)
    else:
        z = complex(z0re, z0im)
        c = complex(pre, pim)
    nu = -1.0
    for k in range(max_iter):
        z = _f(z, c)
        m2 = z.real * z.real + z.imag * z.imag
        if not (m2 <= b2):              # catches > bailout, inf, and nan
            if math.isfinite(m2) and m2 > 1.0:
                l = 0.5 * math.log(m2)
                t = math.log(l / log_b) * inv_log_deg
                nu = k + 1.0 - t
                if nu < 0.0:
                    nu = 0.0
            else:
                nu = k + 1.0
            break
    out[j, i] = nu
'''

# float32 fast path: complex64 orbit. The complex64()/float32() casts pin the
# state to float32 after each step (formula literals like 1j would otherwise
# promote the expression back to complex128). Pixel coords are computed in
# float64 (xmin/dx come in as float64) then rounded to float32.
_CUDA_F32_TEMPLATE = _CUDA_IMPORTS + '''
from numba import cuda, float32, complex64

@cuda.jit(device=True, inline=True)
def _conj(z):
    return complex64(complex(z.real, -z.imag))

@cuda.jit(device=True, inline=True)
def _re(z):
    return complex64(complex(z.real, 0.0))

@cuda.jit(device=True, inline=True)
def _im(z):
    return complex64(complex(z.imag, 0.0))

@cuda.jit(device=True, inline=True)
def _f(z, c):
    return complex64({expr})

@cuda.jit
def kernel(xmin, ymin, dx, dy, width, height, julia, cre, cim,
           z0re, z0im, max_iter, bailout, inv_log_deg, out):
    i, j = cuda.grid(2)
    if i >= width or j >= height:
        return
    b2 = float32(bailout * bailout)
    log_b = float32(math.log(bailout))
    pre = float32(xmin + i * dx)         # coord computed in float64, then cast
    pim = float32(ymin + j * dy)
    if julia:
        z = complex64(complex(pre, pim))
        c = complex64(complex(float32(cre), float32(cim)))
    else:
        z = complex64(complex(float32(z0re), float32(z0im)))
        c = complex64(complex(pre, pim))
    nu = float32(-1.0)
    for k in range(max_iter):
        z = complex64(_f(z, c))
        m2 = float32(z.real * z.real + z.imag * z.imag)
        if not (m2 <= b2):              # catches > bailout, inf, and nan
            if math.isfinite(m2) and m2 > float32(1.0):
                l = float32(0.5) * float32(math.log(m2))
                t = float32(math.log(l / log_b)) * inv_log_deg
                nu = k + float32(1.0) - t
                if nu < float32(0.0):
                    nu = float32(0.0)
            else:
                nu = k + float32(1.0)
            break
    out[j, i] = nu
'''


class _F32Specializer(ast.NodeTransformer):
    """Rewrite a transpiled formula expression to stay in float32/complex64.

    numba lowers complex `**` and mixes with float64 literals by promoting the
    whole expression to complex128 (FP64) -- which on Maxwell is ~30x slower
    than FP32 and defeats the point of the float32 path. Two rewrites keep it
    float32:
      * integer powers  base**n  ->  base*base*...*base   (n in 2..8), so the
        arithmetic is complex64 multiplication instead of a complex128 pow;
      * numeric literals wrapped in complex64()/float32() so they don't drag a
        complex64 subexpression back up to complex128.
    A `**` with a non-small/non-int exponent is left alone (that subexpression
    falls back to complex128, but no preset hits this path)."""

    def visit_BinOp(self, node):
        if (isinstance(node.op, ast.Pow) and isinstance(node.right, ast.Constant)
                and isinstance(node.right.value, int)
                and 2 <= node.right.value <= 8):
            base = self.visit(node.left)
            expr = base
            for _ in range(node.right.value - 1):
                expr = ast.BinOp(left=expr, op=ast.Mult(), right=base)
            return expr
        self.generic_visit(node)
        return node

    def visit_Constant(self, node):
        if isinstance(node.value, complex):
            ctor = "complex64"
        elif isinstance(node.value, (int, float)):
            ctor = "float32"
        else:
            return node
        return ast.Call(func=ast.Name(id=ctor, ctx=ast.Load()),
                        args=[node], keywords=[])


def _specialize_expr_f32(expr: str) -> str:
    """float32-optimize an already-transpiled formula expression (see above)."""
    tree = ast.parse(expr, mode="eval")
    tree = _F32Specializer().visit(tree)
    ast.fix_missing_locations(tree)
    return ast.unparse(tree)


def _cuda_kernel_source(expr: str, precision: str) -> str:
    """Generate numba.cuda kernel source for one formula at 'f32' or 'f64'."""
    if precision == "f32":
        return _CUDA_F32_TEMPLATE.format(expr=_specialize_expr_f32(expr))
    if precision == "f64":
        return _CUDA_F64_TEMPLATE.format(expr=expr)
    raise ValueError(f"precision must be 'f32' or 'f64', got {precision!r}")


class CudaKernelCache:
    """Compiles and caches numba.cuda escape-time kernels per (formula, precision)."""

    def __init__(self):
        self._cache: Dict[Tuple[str, str], object] = {}

    def get(self, formula_text: str, precision: str = "f32"):
        key = (formula_text.strip(), precision)
        fn = self._cache.get(key)
        if fn is None:
            expr = parse_formula(key[0])
            ns: dict = {}
            src = _cuda_kernel_source(expr, precision)
            exec(compile(src, f"<fractal-cuda-kernel-{precision}>", "exec"), ns)
            fn = ns["kernel"]
            self._cache[key] = fn
        return fn


CUDA_KERNELS = CudaKernelCache()

_CUDA_OK: Optional[bool] = None      # None = not probed yet
_CUDA_DISABLED = False               # set True after a runtime GPU failure


def cuda_available() -> bool:
    """True if a usable CUDA GPU is present. Probed once, then cached.

    Returns False (never raises) so callers can treat it as a plain feature
    flag; a mid-run GPU failure latches this off via _CUDA_DISABLED so we stop
    retrying the GPU on every frame."""
    global _CUDA_OK
    if _CUDA_DISABLED:
        return False
    if _CUDA_OK is None:
        try:
            from numba import cuda
            _CUDA_OK = bool(cuda.is_available() and len(cuda.gpus) > 0)
        except Exception:
            _CUDA_OK = False
    return _CUDA_OK


def render_escape_frame_gpu(formula: EscapeFormula, xmin: float, ymin: float,
                            dx: float, dy: float, width: int, height: int,
                            julia: bool, c: complex, max_iter: int,
                            precision: str = "f32",
                            out: Optional[np.ndarray] = None,
                            stream=None) -> np.ndarray:
    """Render a full width x height smooth-iteration frame on the GPU.

    precision is 'f32' (fast, Maxwell-friendly) or 'f64' (bit-faithful twin of
    the CPU kernel). Row j corresponds to imag = ymin + j*dy (bottom-up, same
    as the CPU strip kernel). Raises on any CUDA error -- callers wanting
    fallback should use render_escape_frame."""
    from numba import cuda
    kern = CUDA_KERNELS.get(formula.text, precision)
    deg = max(float(formula.degree), 1.0001)
    if out is None:
        out = np.empty((height, width), dtype=np.float32)
    d_out = cuda.device_array((height, width), dtype=np.float32, stream=stream)
    tpb = (16, 16)
    bpg = ((width + tpb[0] - 1) // tpb[0], (height + tpb[1] - 1) // tpb[1])
    kern[bpg, tpb, stream](xmin, ymin, dx, dy, width, height, julia,
                           c.real, c.imag, formula.z0.real, formula.z0.imag,
                           max_iter, formula.bailout, 1.0 / math.log(deg), d_out)
    d_out.copy_to_host(out, stream=stream)
    if stream is not None:
        stream.synchronize()
    return out


# f32<->f64 handoff for zoom VIDEO. A hard switch at one frame would pop (the
# two kernels differ slightly), so a zoom crossing the handoff crossfades over a
# span-space band: pure GPU-f32 above HI, pure CPU-f64 below LO, a blend of both
# in between. The band sits safely ABOVE _F32_MIN_SPAN so f32 is still valid
# everywhere it's rendered here. escape_precision_plan() returns the mix; the
# blend of the two colorized frames happens in engine.render_frame_blended.
_HANDOFF_SPAN_HI = 5e-4     # >= this: pure f32
_HANDOFF_SPAN_LO = 1.5e-4   # <= this: pure cpu-f64; band is (LO, HI)


def resolve_escape_precision(span: float, precision: str = "auto") -> str:
    """Concrete path ('f32' | 'f64' | 'cpu') that `precision` resolves to.

    'auto' -> GPU 'f32' when the view is shallow enough for float32 to resolve
    pixels (span >= _F32_MIN_SPAN), else 'cpu' (on Maxwell the GPU float64 path
    is slower than the CPU, so deep frames go to the CPU, not GPU-f64). Forced
    'f32'/'f64' degrade to 'cpu' when no GPU is present."""
    if precision == "cpu":
        return "cpu"
    if precision in ("f32", "f64"):
        return precision if cuda_available() else "cpu"
    if precision != "auto":
        raise ValueError(f"unknown precision {precision!r}")
    if not cuda_available():
        return "cpu"
    return "f32" if span >= _F32_MIN_SPAN else "cpu"


def escape_precision_plan(span: float) -> List[Tuple[str, float]]:
    """Render plan for one zoom frame at `span`: a list of (precision, weight)
    with weights summing to 1. One entry outside the handoff band; two (f32 and
    cpu) inside it, so the video pipeline can crossfade and avoid a switch pop."""
    if not cuda_available() or span <= _HANDOFF_SPAN_LO:
        return [("cpu", 1.0)]
    if span >= _HANDOFF_SPAN_HI:
        return [("f32", 1.0)]
    t = (_HANDOFF_SPAN_HI - span) / (_HANDOFF_SPAN_HI - _HANDOFF_SPAN_LO)
    w = t * t * (3.0 - 2.0 * t)          # smoothstep: eased crossfade weight
    return [("f32", 1.0 - w), ("cpu", w)]


def render_escape_frame(formula: EscapeFormula, xmin: float, ymin: float,
                        dx: float, dy: float, width: int, height: int,
                        julia: bool, c: complex, max_iter: int,
                        precision: str = "auto",
                        prefer_gpu: bool = True,
                        out: Optional[np.ndarray] = None) -> np.ndarray:
    """Render a full frame, preferring the GPU with automatic CPU fallback.

    precision:
      'auto' -- GPU float32 when shallow enough (span >= _F32_MIN_SPAN), else
                the CPU float64 kernel. See resolve_escape_precision.
      'f32' / 'f64' -- force GPU at that precision (still CPU-fallback on error).
      'cpu' -- force the CPU kernel.

    On the first GPU failure this warns once and latches the GPU off for the
    rest of the process (see cuda_available), so a bad driver/toolkit degrades
    to CPU rendering instead of failing every frame."""
    global _CUDA_DISABLED
    if out is None:
        out = np.empty((height, width), dtype=np.float32)

    resolved = resolve_escape_precision(dx * width, precision)
    if resolved in ("f32", "f64") and prefer_gpu and cuda_available():
        try:
            return render_escape_frame_gpu(formula, xmin, ymin, dx, dy,
                                           width, height, julia, c, max_iter,
                                           precision=resolved, out=out)
        except Exception as e:                       # pragma: no cover
            _CUDA_DISABLED = True
            warnings.warn(f"CUDA render failed ({e!r}); falling back to CPU "
                          f"for the rest of this session.", RuntimeWarning)

    render_escape_strip(formula, xmin, ymin, dx, dy, width, 0, height,
                        julia, c, max_iter, out)
    return out


# ----------------------------------------------------------------------------- Newton basins

@dataclass
class NewtonFormula:
    name: str
    coeffs: List[complex]          # highest degree first, e.g. z^3 - 1 -> [1, 0, 0, -1]
    tol: float = 1e-9
    default_center: complex = 0j
    default_span: float = 4.0

    def roots(self) -> np.ndarray:
        return np.roots(np.array(self.coeffs, dtype=np.complex128))

    def n_roots(self) -> int:
        return len(self.coeffs) - 1

    @staticmethod
    def parse_coeffs(text: str) -> List[complex]:
        """Parse '1, 0, 0, -1' (highest degree first) into complex coefficients."""
        parts = [p.strip() for p in text.replace(";", ",").split(",") if p.strip()]
        if len(parts) < 3:
            raise FormulaError("need at least 3 coefficients (degree >= 2)")
        out = []
        for p in parts:
            try:
                out.append(complex(p.replace(" ", "")))
            except ValueError:
                raise FormulaError(f"bad coefficient: '{p}'") from None
        if out[0] == 0:
            raise FormulaError("leading coefficient must be nonzero")
        return out


NEWTON_PRESETS: List[NewtonFormula] = [
    NewtonFormula("Newton  z\u00b3 - 1", [1, 0, 0, -1]),
    NewtonFormula("Newton  z^4 - 1", [1, 0, 0, 0, -1]),
    NewtonFormula("Newton  z\u00b3 - 2z + 2", [1, 0, -2, 2]),
    NewtonFormula("Newton  z^5 - 1", [1, 0, 0, 0, 0, -1]),
    NewtonFormula("Newton  z^6 + z^3 - 1", [1, 0, 0, 1, 0, 0, -1]),
    NewtonFormula("Newton  z^8 + 15z^4 - 16", [1, 0, 0, 0, 15, 0, 0, 0, -16]),
]


@njit(parallel=True, nogil=True, cache=False)
def _newton_kernel(xmin, ymin, dx, dy, width, y0, y1, coeffs, dcoeffs, roots,
                   max_iter, tol, out_root, out_nu):
    tol2 = tol * tol
    log_tol = math.log(tol)
    nc = coeffs.shape[0]
    nd = dcoeffs.shape[0]
    nr = roots.shape[0]
    for jj in prange(y1 - y0):
        j = y0 + jj
        pim = ymin + j * dy
        for i in range(width):
            z = complex(xmin + i * dx, pim)
            prev2 = 1e300
            root_id = -1
            nu = -1.0
            for k in range(max_iter):
                p = coeffs[0]
                for a in range(1, nc):
                    p = p * z + coeffs[a]
                dp = dcoeffs[0]
                for a in range(1, nd):
                    dp = dp * z + dcoeffs[a]
                d2 = dp.real * dp.real + dp.imag * dp.imag
                if d2 == 0.0 or not math.isfinite(d2):
                    break
                step = p / dp
                z = z - step
                s2 = step.real * step.real + step.imag * step.imag
                if not math.isfinite(s2):
                    break
                if s2 < tol2:
                    best = 0
                    bd = 1e300
                    for r in range(nr):
                        dr = z.real - roots[r].real
                        di = z.imag - roots[r].imag
                        d = dr * dr + di * di
                        if d < bd:
                            bd = d
                            best = r
                    root_id = best
                    # smooth fraction: where |step| crossed tol (in log space)
                    frac = 0.0
                    if prev2 > 0.0 and s2 > 0.0 and prev2 < 1e299:
                        lp = 0.5 * math.log(prev2)
                        lc = 0.5 * math.log(s2)
                        denom = lc - lp
                        if denom != 0.0:
                            frac = (log_tol - lp) / denom
                            if frac < 0.0:
                                frac = 0.0
                            elif frac > 1.0:
                                frac = 1.0
                    nu = k + frac
                    break
                prev2 = s2
            out_root[jj, i] = root_id
            out_nu[jj, i] = nu


def render_newton_strip(formula: NewtonFormula, xmin: float, ymin: float,
                        dx: float, dy: float, width: int, y0: int, y1: int,
                        max_iter: int, out_root: np.ndarray, out_nu: np.ndarray) -> None:
    coeffs = np.array(formula.coeffs, dtype=np.complex128)
    dcoeffs = (coeffs[:-1] * np.arange(len(coeffs) - 1, 0, -1)).astype(np.complex128)
    roots = formula.roots().astype(np.complex128)
    _newton_kernel(xmin, ymin, dx, dy, width, y0, y1, coeffs, dcoeffs, roots,
                   max_iter, formula.tol, out_root, out_nu)
