# Fractal Studio

Interactive fractal explorer and video renderer. Python + PySide6 + Numba.

Escape-time fractals (Mandelbrot, Julia, Burning Ship, Tricorn, and any custom
formula f(z, c) you type), Newton multi-attractor basin fractals, 256-entry
palettes (including sphere-in-RGB-cube path palettes), deep-dive navigation
with bookmarks, high-resolution stills, 4K zoom-in videos, and Julia morph
videos.

## Install (Ubuntu)

Recommended: conda (required on Ubuntu 18.04 -- see below).

```bash
conda env create -f environment.yml
conda activate fractal
python main.py
```

Or, on Ubuntu 20.04+ only, plain pip works too:

```bash
sudo apt install libegl1 libxkbcommon0        # Qt runtime libs (usually present)
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
python3 main.py
```

ffmpeg is bundled via `imageio-ffmpeg`; nothing else to install.

**Ubuntu 18.04 note.** The pip wheels of PySide6 >= 6.3 require glibc >= 2.28;
18.04 has 2.27, so pip cannot install a working PySide6 there. conda-forge
builds Qt6/PySide6 against a glibc 2.17 sysroot, so the conda environment
above works on 18.04 (and anything newer). Just never `pip install pyside6`
inside the env.

## Controls

| Action | Effect |
|---|---|
| left-click | zoom in by the click factor (default ×10), centered on the point |
| shift + left-click | zoom out by the same factor |
| left-drag | pan |
| scroll wheel | zoom ×1.5 per notch, anchored at the cursor |
| right-click | open the Julia set with c = clicked point |
| `H` | home view · `Backspace` back · `Shift+Backspace` forward |
| `Ctrl+B` | save bookmark · `Ctrl+S` high-res image · `Ctrl+L`/`Ctrl+O` save/load location |
| `Ctrl+Shift+Z` | zoom-in video · `Ctrl+Shift+J` Julia morph video |

The first render of any formula pauses ~1–3 s while Numba JIT-compiles its
kernel; after that it's cached for the session.

## Concepts

**Smooth-iteration cache.** Kernels output a float32 fractional-iteration field.
Palette choice, color density/offset, log mapping, and color cycling are applied
after the fact, so all recolor instantly without re-rendering.

**Palettes.** 256 entries, index 0 reserved for the interior (black by default).
Smooth coloring is quantized onto the palette, so no banding. Sources: rainbow,
gradient stops, JSON files, and *sphere paths* — great circles or pole-to-pole
spirals on the sphere inscribed in the RGB cube (center (127.5)³, radius 127.5),
sampled to 255 colors. Great circles close exactly, so they wrap seamlessly
under mod-255 coloring and cycling. Edit via Palette → Edit…

**Custom formulas.** Any expression in `z` and `c` with `+ - * / **` and
`sin cos tan sinh cosh tanh exp log sqrt asin acos atan abs conj re im`.
Validated with a whitelist AST parser, then compiled to a parallel Numba kernel.
Example — Burning Ship is just `(abs(re(z)) + 1j*abs(im(z)))**2 + c`.

**Newton basins.** Enter polynomial coefficients (highest degree first,
complex allowed). Pixels are colored by which root they converge to — one
palette band per attractor — shaded by smooth convergence speed.
Non-converging points get index 0.

**Locations.** A location JSON stores everything: center (full float64
precision), span, formula, mode, iterations, palette, color settings.
Bookmarks are location files in `locations/`. High-res stills also write a
`.location.json` sidecar, so every image is reproducible.

**Zoom videos.** Dive to a spot, then Video → Zoom-in video. You specify the
zoom rate in ×/second (default 2.0 — a per-frame factor of 2^(1/30) ≈ 1.023
at 30 fps, which reads as a smooth glide; 1.2×/frame would be ≈240×/s) and the
duration is derived from total magnification. Frames are written directly into
an H.264 MP4; check "keep PNG frames" to also get the sequence for your music
edit. Default 4K, 2×2 supersampled.

**Julia morph videos.** c travels a parametrized path: a circle around c₀
(closes exactly → seamless loop), an outward spiral, or a line to a second
constant (optionally there-and-back). "Combine with zoom" makes the set waft
while diving.

## Limits and roadmap (good Claude Code sessions)

* **Depth**: float64 pixelates near ~10¹³ magnification; the app clamps there
  and tells you. Going deeper means perturbation rendering — planned v2.
* **CUDA**: rendering is Numba-parallel on all CPU cores. The kernel codegen in
  `core/formulas.py` is isolated so a `numba.cuda` twin (and splitting frame
  rows across your 4 Titan X's) is a clean drop-in — the top priority once the
  app is on the target machine, since it will cut 4K frame times by an order
  of magnitude. Written CPU-only for now because this build environment has no
  GPU to test against.
* **DeepDream frame filter** (phase 2): PyTorch + a pretrained CNN, gradient
  ascent on each frame toward a chosen channel ("flower detector"), with
  previous-frame seeding for temporal coherence.
* Smaller ideas: pick c₁ for line morphs by clicking; per-preset default
  palettes; palette rotation baked into videos at arbitrary phase; z₀ control
  in the UI; arbitrary aspect-ratio video presets.

## Layout

```
main.py               entry point
core/formulas.py      presets, formula parser/validator, Numba kernel codegen, Newton kernels
core/engine.py        view math, strip/tile rendering, colorization, cancel/progress, locations
core/palette.py       rainbow / sphere-path / gradient palettes, JSON I/O, cycling
core/video.py         zoom + Julia-morph video pipelines (imageio-ffmpeg)
gui/main_window.py    canvas, mouse navigation, threaded progressive rendering, dock, history
gui/dialogs.py        palette editor, custom formula, high-res save, video export dialogs
palettes/  locations/ bundled palettes and example bookmarks
```
