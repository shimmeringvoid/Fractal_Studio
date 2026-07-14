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

* "monster" -- the render workstation: i7-5930K (6c/12t @ 3.5 GHz, 2014-era),
  64 GiB RAM, 8 TB disk, 4x NVIDIA GTX TITAN X (GM200 = MAXWELL, sm_52,
  12 GB each), Ubuntu 22.04.5, GNOME 42.9, X11. Modern glibc; conda env
  solves unpinned. IMPORTANT: until the CUDA render path exists, monster's
  CPU rendering is NO faster than the laptops (same 12 threads, older
  cores) -- the 4 GPUs are the entire point of this machine. Maxwell
  confirmed => numba-CUDA must target a toolkit the installed driver
  supports; recent PyTorch binaries may lack sm_52. MEASURED 2026-07-14: driver 535.183.01, CUDA ceiling 12.2, all 4 GPUs healthy and idle. DECISION: use CUDA toolkit 11.8 for numba (full sm_52 support; CUDA 12 deprecates Maxwell, CUDA 13 removes it).
* System76 laptop: Ubuntu 18.04.6, glibc 2.27 (the reason for landmine 1),
  i7-9750H (6c/12t), 62.5 GiB RAM, GTX 1650 (Turing, sm_75), GNOME 3.28.
* Newer laptop: Ubuntu 22.04.4, same conda env; its default GNOME Videos
  mis-renders high-bitrate H.264 (gray + squeezed) -- mpv is the fix.

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

## Workstream B: DeepDream video filter — SEPARATE REPO

The DeepDream feature lives in its own project, `deepdream-video` (separate
repo, separate PyTorch/CUDA env), NOT in this codebase. Rationale: it's a
general video-to-video filter, useful on any footage, and its PyTorch/CUDA
dependency stack must not co-resolve with this repo's fragile
conda-forge/PySide6/glibc-2.27 env. The two projects meet only through a
folder of numbered PNG frames — exactly what this app's "keep PNG frames"
video option emits. Fractal Studio is just one upstream frame source.

The one fractal-specific hook worth remembering: for zoom videos the exact
inter-frame transform is a pure scale about center by the per-frame factor.
Exposing that (it's already derivable from the ZoomVideoSpec) lets the dreamer
warp-and-blend dream[t-1] into the seed for frame t, giving strong temporal
coherence. Consider writing the per-frame zoom factor into a sidecar
(e.g. frames_meta.json) alongside the PNG sequence so the dreamer can consume
it without knowing anything else about fractals. See the deepdream-video repo's
own CLAUDE.md for that project's full spec.

## Working conventions that have served well

Small changes delivered as `git apply` heredoc patches or sed one-liners;
commit locally often, push when the state is clone-worthy; bookmarks/location
JSONs are the currency for "render this spot later/elsewhere"; every long
render has progress + cancel; verify claims empirically (wheel tags, feedstock
configs, minimal crash probes) before acting on them.
