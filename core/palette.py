"""Palette system: 256-entry (index 0 reserved for black interior/non-convergence).

Palette sources:
  * rainbow          -- default 256-color HSV rainbow
  * sphere paths     -- paths on the sphere inscribed in the RGB cube
                        (center (127.5,)*3, radius 127.5), sampled to 255 colors.
                        Great circles and pole-to-pole spirals give vibrant,
                        smoothly blending colors.
  * gradient stops   -- user-defined color stops, linearly interpolated
  * JSON files       -- save/load any of the above (generator params preserved)

Cycling is done at display time by rotating indices 1..255 (index 0 fixed).
"""
from __future__ import annotations

import colorsys
import json
import os
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np

PALETTE_SIZE = 256


# ----------------------------------------------------------------------------- generators

def rainbow_palette() -> np.ndarray:
    """Default palette: index 0 black, 1..255 a full HSV rainbow sweep."""
    pal = np.zeros((PALETTE_SIZE, 3), dtype=np.uint8)
    for i in range(1, PALETTE_SIZE):
        h = (i - 1) / 255.0
        r, g, b = colorsys.hsv_to_rgb(h, 1.0, 1.0)
        pal[i] = (int(r * 255), int(g * 255), int(b * 255))
    return pal


def _sphere_points_to_palette(pts: np.ndarray) -> np.ndarray:
    """pts: (255, 3) points on/near the RGB sphere -> palette with index 0 black."""
    pal = np.zeros((PALETTE_SIZE, 3), dtype=np.uint8)
    pal[1:] = np.clip(np.round(pts), 0, 255).astype(np.uint8)
    return pal


def sphere_great_circle_palette(tilt_deg: float = 55.0, azimuth_deg: float = 30.0,
                                phase: float = 0.0) -> np.ndarray:
    """Great circle on the sphere inscribed in the RGB cube.

    The circle's plane normal is given by (tilt, azimuth) in spherical coords
    relative to the gray axis (1,1,1)/sqrt(3).  Closes exactly, so the palette
    wraps seamlessly -- ideal for cycling and mod-255 coloring.
    """
    center = np.array([127.5, 127.5, 127.5])
    radius = 127.5
    tilt = np.radians(tilt_deg)
    az = np.radians(azimuth_deg)
    # Build an orthonormal frame around the chosen normal.
    n = np.array([np.sin(tilt) * np.cos(az), np.sin(tilt) * np.sin(az), np.cos(tilt)])
    a = np.array([0.0, 0.0, 1.0]) if abs(n[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
    u = np.cross(n, a); u /= np.linalg.norm(u)
    v = np.cross(n, u)
    t = (np.arange(255) / 255.0 + phase) * 2.0 * np.pi   # endpoint excluded -> seamless
    pts = center + radius * (np.cos(t)[:, None] * u + np.sin(t)[:, None] * v)
    return _sphere_points_to_palette(pts)


def sphere_spiral_palette(turns: float = 3.0, phase: float = 0.0,
                          pole_axis: str = "gray") -> np.ndarray:
    """Pole-to-pole spiral (loxodrome-like) on the RGB sphere.

    pole_axis: 'gray' spirals from near-black to near-white through vivid hues;
               'r', 'g', 'b' use that channel's axis instead.
    """
    center = np.array([127.5, 127.5, 127.5])
    radius = 127.5
    axes = {
        "gray": np.array([1.0, 1.0, 1.0]) / np.sqrt(3.0),
        "r": np.array([1.0, 0.0, 0.0]),
        "g": np.array([0.0, 1.0, 0.0]),
        "b": np.array([0.0, 0.0, 1.0]),
    }
    k = axes.get(pole_axis, axes["gray"])
    a = np.array([0.0, 0.0, 1.0]) if abs(k[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
    u = np.cross(k, a); u /= np.linalg.norm(u)
    v = np.cross(k, u)
    t = np.arange(255) / 254.0                       # 0..1 inclusive: pole to pole
    theta = t * np.pi                                # polar angle
    phi = (turns * t + phase) * 2.0 * np.pi          # winding
    pts = center + radius * (np.sin(theta)[:, None] * (np.cos(phi)[:, None] * u +
                                                       np.sin(phi)[:, None] * v)
                             + np.cos(theta)[:, None] * k[None, :])
    return _sphere_points_to_palette(pts)


def gradient_palette(stops: List[Tuple[float, Tuple[int, int, int]]]) -> np.ndarray:
    """Linear interpolation between color stops [(pos in 0..1, (r,g,b)), ...].
    Index 0 stays black; stops map onto indices 1..255."""
    stops = sorted(stops, key=lambda s: s[0])
    if not stops:
        return rainbow_palette()
    pal = np.zeros((PALETTE_SIZE, 3), dtype=np.uint8)
    pos = np.array([s[0] for s in stops], dtype=np.float64)
    cols = np.array([s[1] for s in stops], dtype=np.float64)
    t = np.arange(255) / 254.0
    for ch in range(3):
        pal[1:, ch] = np.clip(np.round(np.interp(t, pos, cols[:, ch])), 0, 255)
    return pal


# ----------------------------------------------------------------------------- Palette object

@dataclass
class Palette:
    """A named palette plus (optionally) the generator recipe that made it,
    so saved files can be re-edited parametrically."""
    name: str = "Rainbow"
    colors: np.ndarray = field(default_factory=rainbow_palette)   # (256,3) uint8
    recipe: Optional[dict] = None    # e.g. {"kind":"sphere_spiral","turns":3.0,...}

    def rotated(self, offset: int) -> np.ndarray:
        """Return colors with indices 1..255 rotated by `offset` (index 0 fixed)."""
        off = int(offset) % 255
        if off == 0:
            return self.colors
        out = self.colors.copy()
        out[1:] = np.roll(self.colors[1:], off, axis=0)
        return out

    # -- JSON I/O ------------------------------------------------------------
    def to_json(self) -> dict:
        return {
            "name": self.name,
            "recipe": self.recipe,
            "colors": self.colors.tolist(),
        }

    @staticmethod
    def from_json(d: dict) -> "Palette":
        recipe = d.get("recipe")
        if recipe and "colors" not in d:
            colors = palette_from_recipe(recipe)
        else:
            colors = np.array(d["colors"], dtype=np.uint8)
            if colors.shape != (PALETTE_SIZE, 3):
                raise ValueError(f"palette must be {PALETTE_SIZE}x3, got {colors.shape}")
        return Palette(name=d.get("name", "Unnamed"), colors=colors, recipe=recipe)

    def save(self, path: str) -> None:
        with open(path, "w") as f:
            json.dump(self.to_json(), f)

    @staticmethod
    def load(path: str) -> "Palette":
        with open(path) as f:
            return Palette.from_json(json.load(f))


def palette_from_recipe(recipe: dict) -> np.ndarray:
    kind = recipe.get("kind")
    if kind == "rainbow":
        return rainbow_palette()
    if kind == "sphere_great_circle":
        return sphere_great_circle_palette(recipe.get("tilt_deg", 55.0),
                                           recipe.get("azimuth_deg", 30.0),
                                           recipe.get("phase", 0.0))
    if kind == "sphere_spiral":
        return sphere_spiral_palette(recipe.get("turns", 3.0),
                                     recipe.get("phase", 0.0),
                                     recipe.get("pole_axis", "gray"))
    if kind == "gradient":
        stops = [(p, tuple(c)) for p, c in recipe.get("stops", [])]
        return gradient_palette(stops)
    raise ValueError(f"unknown palette recipe kind: {kind}")


def builtin_palettes() -> List[Palette]:
    """Palettes available from the menu on first run."""
    fire = [(0.0, (0, 0, 0)), (0.2, (120, 0, 0)), (0.45, (255, 80, 0)),
            (0.7, (255, 200, 40)), (1.0, (255, 255, 220))]
    ocean = [(0.0, (0, 5, 40)), (0.35, (0, 80, 160)), (0.65, (60, 190, 210)),
             (0.85, (200, 250, 240)), (1.0, (255, 255, 255))]
    return [
        Palette("Rainbow", rainbow_palette(), {"kind": "rainbow"}),
        Palette("Sphere: Great Circle",
                sphere_great_circle_palette(),
                {"kind": "sphere_great_circle", "tilt_deg": 55.0, "azimuth_deg": 30.0, "phase": 0.0}),
        Palette("Sphere: Spiral (3 turns)",
                sphere_spiral_palette(3.0),
                {"kind": "sphere_spiral", "turns": 3.0, "phase": 0.0, "pole_axis": "gray"}),
        Palette("Sphere: Spiral (6 turns)",
                sphere_spiral_palette(6.0),
                {"kind": "sphere_spiral", "turns": 6.0, "phase": 0.0, "pole_axis": "gray"}),
        Palette("Fire", gradient_palette(fire), {"kind": "gradient", "stops": fire}),
        Palette("Ocean", gradient_palette(ocean), {"kind": "gradient", "stops": ocean}),
    ]


def palette_strip_image(colors: np.ndarray, height: int = 24) -> np.ndarray:
    """(height, 256, 3) preview strip for the GUI."""
    return np.repeat(colors[None, :, :], height, axis=0)
