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
    EscapeFormula("Quartic  z\u2074 + c", "z**4 + c", 4.0, default_center=0j, default_span=2.8,
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
    NewtonFormula("Newton  z\u2074 - 1", [1, 0, 0, 0, -1]),
    NewtonFormula("Newton  z\u00b3 - 2z + 2", [1, 0, -2, 2]),
    NewtonFormula("Newton  z\u2075 - 1", [1, 0, 0, 0, 0, -1]),
    NewtonFormula("Newton  z\u2076 + z\u00b3 - 1", [1, 0, 0, 1, 0, 0, -1]),
    NewtonFormula("Newton  z\u2078 + 15z\u2074 - 16", [1, 0, 0, 0, 15, 0, 0, 0, -16]),
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
