# Fractal Studio — project context for Claude

Read this first. It encodes the project state and several hard-won constraints
that are expensive to re-learn.

## What this is

Interactive fractal explorer and video renderer. Python 3.12, PySide6 GUI,
Numba-parallel CPU kernels. Escape-time fractals (Mandelbrot/Julia/Burning
Ship/custom f(z,c) via a whitelist-AST formula compiler), Newton
multi-attractor basins, 256-entry palettes (incl. sphere-in-RGB-cube paths),
click-to-zoom deep dives to ~10^13 magnification (float64 limit, clamped),
bookmarks ("location" JSONs storing everything needed to reproduce a view),
high-res tiled stills, 4K zoom-in videos (rate specified in x/second), and
Julia morph videos (circle/spiral/line paths in c-space, seamless loops,
combinable with zoom).

Repo: https://github.com/shimmeringvoid/Fractal_Studio.git
Owner: rafa. Goal: explore fractals and produce zoom-in videos (music added
later in an editor), eventually with a DeepDream frame filter (below).

## Machines

* Laptop (working daily driver): System76, Ubuntu 18.04.6, glibc 2.27,
  i7-9750H (6c/12t), 62.5 GiB RAM, GTX 1650 (Turing, sm_75), GNOME 3.28, X11.
* Workstation (render farm, not yet set up): 4x NVIDIA Titan X, ~16 CPU cores,
  large RAM. Ubuntu version unknown — run `lsb_release -a`, `ldd --version`,
  and `nvidia-smi` first. NOTE: "Titan X" is ambiguous — Maxwell (sm_52) vs
  Titan X Pascal (sm_61). This matters for PyTorch binary support; check
  before choosing versions.

## Hard-won landmines (do not re-learn these)

1. **glibc / PySide6.** Pip wheels of PySide6 >= 6.3 require glibc >= 2.28;
   Ubuntu 18.04 has 2.27. PySide6 MUST come from conda-forge (their Qt6 is
   built against a glibc 2.17 sysroot). Never `pip install pyside6` in the
   env. Env is `environment.yml` (conda-forge + pip:imageio-ffmpeg only).
2. **Font-fallback segfault.** On this desktop, Qt 6.11 segfaults the moment
   text shaping needs a glyph missing from the default UI font (Ubuntu font) —
   the font-fallback walk crashes (suspect: 18.04's ancient Noto Color Emoji
   during enumeration). Two crashes were caused by this (a ✕ button glyph;
   an ≈ in a dialog label). Rule: **UI strings stay ASCII**, plus these
   proven-safe chars only: ² ³ ¹ × · — … . No arrows, no ✓/✕, no sub/
   superscripts beyond ³, no emoji. An audit approach that works: walk all
   ast.Constant strings and flag chars outside the safe set. Root-cause fix
   (optional side quest): identify/neutralize the crashing system font, then
   this rule can be relaxed.
3. **SESSION_MANAGER.** main.py pops it from the environment before
   QApplication (the app has no session-restore state); Qt otherwise prints
   warnings, and conda env vars can't express "unset". Leave that line alone.
4. Env has `QT_LOGGING_RULES="qt.qpa.theme.gnome=false"` set via
   `conda env config vars` to mute a harmless GNOME-3.28 dbus complaint.

## Architecture map

    main.py               entry point (drops SESSION_MANAGER, starts Qt)
    core/formulas.py      presets; formula parser/validator; **kernel codegen**
                          (escape-time, exec'd Numba source); Newton kernels
    core/engine.py        ViewState math; strip/tile rendering with progress +
                          cancel; smooth-nu -> 8-bit palette colorization;
                          location (bookmark) JSON I/O
    core/palette.py       rainbow / sphere-path / gradient palettes, cycling
    core/video.py         zoom + Julia-morph video pipelines (imageio-ffmpeg,
                          H.264 MP4, optional PNG frame sequence)
    gui/main_window.py    canvas, mouse nav (click recenters+zooms x10, wheel
                          anchors under cursor, drag pans, right-click Julia),
                          threaded progressive rendering, dock, history,
                          bookmarks
    gui/dialogs.py        palette editor, custom formula, high-res save,
                          video export dialogs (background threads + cancel)

Key design facts: kernels emit a float32 smooth-iteration field ("nu");
recoloring/cycling never re-renders. Formula kernels are generated as source
and exec'd, cached per formula text — this isolation is what makes a CUDA
twin a clean drop-in. Videos render frames via the same tiled path as stills.

## Status (2026-07-10)

v1 fully working on the laptop: exploration, palettes/cycling, Julia
from point, bookmarks, and zoom-video export all exercised. Recent commits:
recentering click-zoom; glyph sweep (landmine 2); SESSION_MANAGER fix.
Workstation not yet cloned/set up.

## Workstream A (top priority): CUDA rendering

Numba CPU kernels are the current bottleneck for 4K supersampled video.
Plan: generate `numba.cuda` twins of the escape-time kernel in
core/formulas.py (same codegen pattern), auto-fallback to CPU, then shard
frame batches across the 4 GPUs (frames are embarrassingly parallel).
Newton kernel can follow. Verify Numba CUDA support for the Titans'
compute capability; Maxwell needs CUDA <= 11.x toolchains — pin accordingly.
The GTX 1650 laptop (sm_75) can be the dev/test GPU.

## Workstream B: DeepDream frame filter (phase 2)

Background: long ago rafa independently discovered this effect on an early
CNN — feeding a "flower detector" node's output back amplified flower
hallucinations over ~5 iterations. That is DeepDream / "Inceptionism"
(Mordvintsev, Olah & Tyka, Google 2015): iterated gradient ascent on the
*input image* to maximize a chosen unit's activation. Goal here: apply it
per-frame to fractal videos ("flower fractals").

Technical plan:
* PyTorch + torchvision pretrained net. GoogLeNet/Inception-v1 is the
  classic dreamer (mid layers like inception4c/4d); VGG16 also works well.
  Objective: L2 norm of the chosen channel's activation (hook the layer).
* Per image: 3–5 octaves (scale ~1.4), re-adding lost detail between
  octaves; per octave 10–20 steps of: random jitter roll (<=32 px), forward,
  backward, `img += lr * grad / (mean|grad| + eps)`, unroll, clamp.
* Channel discovery: batch-dream noise/gray images per channel into a
  contact sheet; pick "flower"-like channels by eye. (ImageNet nets have
  plenty of flora/texture detectors in mid layers.)
* **Temporal coherence (the interesting part):** naive per-frame dreaming
  flickers. Standard fixes: seed frame t with the dreamed frame t-1 blended
  with the fresh frame t; fixed jitter seeds. Our advantage: for zoom videos
  the exact inter-frame transform is known (pure scale about center by the
  per-frame factor), so warp dream(t-1) by that zoom before blending —
  init = a*zoomwarp(dream[t-1]) + (1-a)*frame[t], a≈0.5–0.8. Should give
  far better coherence than optical flow guessing.
* Shape: standalone CLI first — `dream/dream_frames.py --in DIR --out DIR
  --model googlenet --layer inception4c --channel N --octaves 4 --steps 12
  --blend 0.6 --coherent-zoom RATE` operating on the PNG sequences the video
  dialogs already emit; re-encode with ffmpeg. GUI checkbox later.
* Performance: dream at 1080p–2K and upscale for 4K if needed (or tile);
  shard frame ranges across the 4 workstation GPUs.
* PyTorch install caution: recent PyTorch binaries may have dropped Maxwell
  (sm_52). Check `torch.cuda.get_arch_list()` against `nvidia-smi`; pin an
  older torch (e.g. 1.13/2.0-era cu11x) if the Titans are Maxwell. Laptop's
  sm_75 works on any recent build — prototype there.

Milestones: (1) single-image flower-dream on the laptop GPU; (2) coherent
5-second dreamed zoom clip; (3) multi-GPU batch on the workstation.

Open questions for rafa: which network aesthetic (GoogLeNet's ornate look
vs AlexNet-era chunkier features, closer to the original memory)? dream
strength constant vs ramping during the zoom? per-video channel, or blend
several channels?

## Working conventions that have served well

Small changes delivered as `git apply` heredoc patches or sed one-liners;
commit locally often, push when the state is clone-worthy; bookmarks/location
JSONs are the currency for "render this spot later/elsewhere"; every long
render has progress + cancel; verify claims empirically (wheel tags, feedstock
configs, minimal crash probes) before acting on them.
