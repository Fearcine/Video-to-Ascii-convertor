# VIDEO TO ASCII

### Luminance Compositing Studio & Layered Glyph VFX Engine

[Platform](https://www.google.com/search?q=https://github.com/Fearcine/Video-to-Ascii-convertor/releases)
[](https://www.python.org/)
[](https://opencv.org/)
[MIT License](https://www.google.com/search?q=https://github.com/Fearcine/Video-to-Ascii-convertor/blob/main/LICENSE)

*A real-time desktop studio and headless optical rendering engine that transforms video footage and still imagery into high-luminance, full-color typographic artworks, emissive glyph fields, and bloom-graded motion sequences.*



*Layered Glyph VFX Mode ("Scattered Lumina") — Poisson-disk glyph instances aligned to image gradient tangents, selective subject masking, multi-scale emissive stroke bloom, and temporal wind sway over graded night footage. [View Full-Palette GIF](https://www.google.com/search?q=assets/demo-scatter.gif) · [View Native 30 FPS MP4](https://www.google.com/search?q=assets/demo-scatter.mp4)*

## I. The Optical Problem & The Studio Upgrade

Conventional ASCII converters render every cell of a rigid character grid over a flat black canvas. Because typographic glyphs occupy only 15% to 35% of their rectangular cell area (and sparse punctuation marks cover less than 10%), up to 85% of display luminance is lost to dead space. Dark background regions collapse into a noisy wall of characters, while bright highlights appear dim and washed out.

This release introduces a decoupled, five-stage **ASCII Magic Studio & Layered Glyph VFX Engine** (`vfx/`) alongside the original high-speed **Classic Grid** renderer:

1. **Full-Resolution Graded Backdrops (`vfx/grading.py`)**: Composites the original video or image frame underneath the typographic layer as a `Blurred Image`, `Original Image`, `Frosted Glass` diffuser, or `Solid Color` canvas with independent controls for Gaussian blur radius, opacity, exposure, contrast, saturation (`0%` for monochrome backing), and luminance-preserving color tinting.
2. **Salient Subject & Contour Isolation (`vfx/fields.py`)**: Eliminates background character clutter using a configurable `[Dark Cutoff, Light Cutoff]` luminance window with smooth Hermite (`smoothstep`) edge feathering, Sobel gradient magnitude emphasis, automatic `Salient Object` foreground detection (combining border color distance, Difference-of-Gaussians local contrast, and edge density), and 8×8 Bayer ordered dither coverage (`coverage`).
3. **Incandescent Highlight Cores & Gradient LUTs (`vfx/lighting.py`)**: Amplifies character RGB luminance (`50%` to `300%`), drives peak highlights toward a `255` hot-white core while preserving midtone chroma, and maps tonal values across 9 built-in and custom 256-entry color gradient look-up tables.
4. **Per-Character Emissive Bloom & Optical Blending (`vfx/bloom.py`, `vfx/compositor.py`)**: Computes multi-scale Gaussian stroke glow (`charBloom`) directly from isolated character masks before compositing via `Screen`, `Add (Linear Dodge)`, `Color Dodge`, `Overlay`, `Soft Light`, or `Normal` blend modes, paired with a 1px micro-contrast shadow so glowing glyphs remain legible over bright clouds and specular highlights.
5. **Spatial Particle Scatter & Stackable Post-FX (`vfx/distribution.py`, `vfx/glyph_bank.py`)**: Introduces **Layered Glyph VFX** mode, scattering up to 6,000 rotated and scaled glyph sprites in resolution-independent `[0, 1]²` coordinates (`Poisson Disk` or `Jittered Grid`) aligned to image gradient tangents or normals with deterministic wind animation, finished with Chromatic Aberration, CRT Scanlines, Film Grain, and Radial Vignette.

## II. Motion Exhibition: Source Footage vs. Layered Glyph VFX

| Original Source Sequence (`assets/demo.mp4`) | Layered Glyph VFX Output (`assets/demo-scatter.mp4`) |
| :---: | :---: |
|  |  |
| *Raw 30 FPS multi-scene night portrait sequence ([View MP4](https://www.google.com/search?q=assets/demo.mp4))* | *Poisson-disk scattered glyphs, gradient-tangent rotation, smoothstep density thinning, and emissive bloom ([View MP4](https://www.google.com/search?q=assets/demo-scatter.mp4))* |

## III. Visual Studies: Source vs. Studio Output

Each study below pairs the unedited source image from `assets/` directly against the real-time PyQt6 studio workspace capture and active optical pipeline parameters.

### Study I — Chiaroscuro & Salient Subject Isolation

By setting the detection filter to `Salient Object` and desaturating both the backdrop and character layer (`0%` saturation), the engine isolates the foreground painter and skeletal figure from Arnold Böcklin's *Self-Portrait with Death Playing the Fiddle* (1872). The upper canvas void remains completely free of character noise while facial structure and bone contours are sculpted in high-contrast monochrome glyphs (`268%` Luminosity, `63%` Hot-White Core, `Screen` blend).

| Source Artwork (`demo-painting.png`) | ASCII Magic Studio Output & Parameters (`demo-painting-exp.png`) |
| :---: | :---: |
|  |  |

* **Recreation Recipe**: Mode: `ASCII Magic Studio` · Backdrop: `Original Image` (Blur `14`, Opacity `90%`, Exposure `75%`, Contrast `135%`, Saturation `0%`) · Detection: `Salient Object` (Dark Cutoff `27`, Edge Feather `50`, Edge Emphasis `45%`) · Illumination: `Screen` Blend, Luminosity `268%`, Hot-White Core `63%`, Char Saturation `0%`, Grid `190×228`

### Study II — Incandescent Highlights & Per-Character Bloom

Night endurance racing photography presents extreme dynamic range: dark asphalt, specular carbon-fiber reflections, tungsten pit lighting, and aerial fireworks. Using `Color Dodge` blending over a softly blurred backdrop with `99%` Per-Character Bloom (`18px` glow radius) and `40%` Scene Bloom transforms the fireworks, headlights, and red-and-white livery into self-luminous typographic structures.

| Source Photograph (`demo-nascar.png`) | ASCII Magic Studio Output & Parameters (`demo-nascar-exp.png`) |
| :---: | :---: |
|  |  |

* **Recreation Recipe**: Mode: `ASCII Magic Studio` · Detection: Dark Cutoff `21`, Edge Feather `50`, Edge Emphasis `33%` · Illumination: `Color Dodge` Blend, Luminosity `106%`, Hot-White Core `45%`, Char Saturation `120%` · Bloom: Char Bloom `99%` (Radius `18px`), Scene Bloom `40%`, Vignette `15%`, Grid `200×72` (`ASCII Magic` ramp)

### Study III — Chromatic Gradient Mapping & Solar Core

Applied to a neon-lit portrait (`demo-ana.png`), the `Solar Flare Gold` recipe blends a `75%` `Solar Gold` gradient map with the source magenta-violet lighting via `Color Dodge`. The `Luminance + Edges` detector suppresses characters in deep hair and background shadows (`Dark Cutoff: 30`) while driving skin and background highlights toward a warm, incandescent gold core.

| Source Portrait (`demo-ana.png`) | ASCII Magic Studio Output & Parameters (`demo-ana-ex.png`) |
| :---: | :---: |
|  |  |

* **Recreation Recipe**: Mode: `ASCII Magic Studio` · Recipe: `Solar Flare Gold` · Backdrop: `Original Image` (Blur `8`, Exposure `85%`, Contrast `110%`, Saturation `115%`) · Detection: `Luminance + Edges` (Dark Cutoff `30`, Edge Feather `22`, Edge Emphasis `30%`) · Illumination: `Color Dodge` Blend, Luminosity `160%`, Hot-White Core `65%`, Gradient Map: `Solar Gold` (Mix `75%`), Grid `200×116`

## IV. Why Local Open-Source Over Paid Web Generators

Browser-based ASCII and shader platforms frequently lock advanced effect libraries, studio dithering, and high-resolution video exports behind monthly Pro subscriptions ($48 to $96/year) or metered render credits, while struggling with browser memory caps on long video files. **Video to ASCII** runs locally on your hardware with zero restrictions.

| Capability | Video to ASCII (Local Open-Source Studio) | Paid Online & Freemium Browser Tools |
| :--- | :--- | :--- |
| **Cost & Licensing** | **100% Free & Open Source (MIT)** — every shader, blend mode, gradient LUT, and recipe unlocked permanently | Free tiers restrict studio effects or resolution; full shader libraries and 4K/pro exports require **$4–$8/mo subscriptions** or paid credit packs |
| **Video Export Limits** | **Unlimited duration, frame rate, and resolution** — export full-length MP4 videos, high-DPI PNGs, TXT streams, and HTML | Browser WebGL texture ceilings, short duration caps, render queues, or metered export credits |
| **Media Privacy** | **100% Offline Local Execution** — unreleased edits, personal portraits, and client footage never leave your disk | Requires loading media into browser sandboxes or uploading clips to external cloud processing servers |
| **Real-Time Scrubbing** | **Multithreaded C++/OpenCV SIMD & NumPy** — instant paused-frame re-rendering and deadline-scheduled playback | Heavy DOM/Canvas overhead that drops frames or crashes browser tabs on high-density grids |
| **Temporal Stability** | **Seed-Deterministic `[0, 1]²` Coordinates** — monotonic `smoothstep` thinning and `t = frame_index / fps` wind math prevent glyph popping | Frame-independent random noise that shimmers or boils unpredictably during video export |
| **Programmability** | **Qt-Free Python SDK (`vfx/`)** — script batch renders, custom LUTs, or automated pipelines directly in Python | Closed proprietary web UI with no local CLI or programmatic batch API |

## V. Optical Architecture & What Changed

The rendering core (`vfx/`) is architected as a stateless, Qt-free image processing pipeline that accepts any BGR `numpy.ndarray` frame and returns a composited RGB `uint8` frame alongside the underlying 2D character and color grids:

1. **Stage 1 — Backdrop Grading (`vfx/grading.py`)**: Renders the underlying frame as `Blurred Image`, `Original Image`, `Frosted Glass`, or `Solid Color`, applying exposure, contrast, saturation, and luminance-preserving RGB tinting.
2. **Stage 2 — Spatial & Saliency Analysis (`vfx/fields.py`)**: Evaluates `Luminance + Edges`, `Luminance Threshold`, `Salient Object` (border color distance + Difference-of-Gaussians + Sobel edge density), `Edges Only`, or `Full Frame` maps, applying a `[Dark Cutoff, Light Cutoff]` Hermite `smoothstep` window and 8×8 Bayer ordered dither coverage.
3. **Stage 3 — Illumination & Gradient Mapping (`vfx/lighting.py`)**: Maps gamma-corrected luminance to character indices and blends source RGB colors with cached 256-entry gradient LUTs, applying `luminosity_boost` (`50–300%`), edge lift, and `highlight_power` hot-white core compression.
4. **Stage 4 — Glyph Compositing (`vfx/compositor.py`, `glyph_atlas.py`, `vfx/glyph_bank.py`)**:
   - In **ASCII Magic Studio** mode: Rasterizes grid-aligned glyphs with a 1px micro-contrast shadow, computes isolated per-character stroke bloom, and blends over the graded backdrop using 1 of 6 optical blend modes.
   - In **Layered Glyph VFX** mode: Samples candidate coordinates from the cached `[0, 1]²` `Poisson Disk` or `Jittered Grid` distribution via `cv2.remap`, selects quantized size and rotation bins (`12` angle steps across `[-90°, +75°]`) from `GlyphBank`, applies harmonic wind sway, and alpha-composites instances with emissive accumulation.
5. **Stage 5 — Scene Highlight Bloom & Post-FX (`vfx/bloom.py`)**: Applies multi-scale pyramid highlight bloom with soft-shoulder tone roll-off, followed by Chromatic Aberration, CRT Scanlines, Film Grain, and Radial Vignette.

### Module-by-Module Upgrade Summary

| Module | Role in the Architecture |
| :--- | :--- |
| `vfx/engine.py` | Orchestrates `render_ascii_magic_studio()` and `render_scattered_vfx()` with zero GUI dependencies. |
| `vfx/settings.py` | Defines the immutable, slotted `VfxSettings` dataclass with automatic range clamping (`validated()`) and forward-compatible JSON serialization. |
| `vfx/grading.py` | Renders full-resolution graded backdrops (`Blurred Image`, `Original Image`, `Frosted Glass`, `Solid Color`) with exposure, contrast, saturation, and tint controls. |
| `vfx/fields.py` | Computes luminance, Sobel edge magnitude, and gradient orientation maps; implements `Salient Object` isolation, `smoothstep` feathering, and 8×8 Bayer ordered dither coverage. |
| `vfx/lighting.py` | Caches 256-entry RGB gradient LUTs and computes per-cell illuminated colors with `luminosity_boost`, edge lift, and `highlight_power` white-hot cores. |
| `vfx/compositor.py` | Performs SIMD-accelerated `uint16`/float compositing for both grid-aligned studio glyphs (with 1px micro-shadow) and rotated scattered sprite instances. |
| `vfx/bloom.py` | Computes isolated 1/2 and 1/4 resolution per-character stroke bloom (`compute_character_bloom()`), pyramid scene highlight bloom, and stackable retro/cinematic Post-FX. |
| `vfx/distribution.py` | Generates seed-cached normalized `[0, 1]²` candidate points (`Poisson Disk` and `Jittered Grid`) with monotonic threshold values `u` for flicker-free density changes. |
| `vfx/glyph_bank.py` | Pre-rasterizes and caches rotated (`12` angle bins) and scaled (`3` size tiers) glyph sprite masks with optional 1px outline rings and CJK font fallback. |
| `vfx/presets.py` | Provides 9 curated studio and VFX recipes for instant one-click styling. |

## VI. Curated One-Click Recipes

Select any recipe from the **One-Click Recipe** dropdown in the sidebar to configure all five pipeline stages automatically, then fine-tune any slider in real time (even while video playback is paused):

| Recipe Name | Engine Mode | Optical Signature |
| :--- | :--- | :--- |
| **Cloud Glow (ASCII Magic)** | `ASCII Magic Studio` | Softly blurred full-brightness backdrop (`bg_blur=10`), `Luminance + Edges` mask (`coverage=76%`), `Sky & Cloud Glow` gradient mix, hot-white core, and `Screen` character bloom. |
| **Cyberpunk Neon** | `ASCII Magic Studio` | Deep violet-tinted blurred backdrop, high edge emphasis (`65%`), `Cyberpunk Neon` cyan-magenta gradient, `Add (Linear Dodge)` blend, strong character bloom, chromatic aberration, and CRT scanlines. |
| **Matrix Phosphor** | `ASCII Magic Studio` | Japanese CJK character ramp over a dark emerald-tinted backdrop with `Matrix Phosphor` gradient, CRT scanlines (`35%`), and green phosphor bloom (`(80, 255, 140)`). |
| **Solar Flare Gold** | `ASCII Magic Studio` | Crisp dimmed original backdrop with `Solar Gold` gradient (`75%` mix), `Color Dodge` blending, incandescent white-gold highlight cores (`65%`), and warm scene bloom. |
| **Salient Subject Spotlight** | `ASCII Magic Studio` | `Frosted Glass` backdrop paired with automatic `Salient Object` foreground detection so only the primary subject receives glowing source-colored ASCII glyphs. |
| **Edge Contour Neon** | `ASCII Magic Studio` | Crisp original backdrop with `Edges Only` detection (`edge_emphasis=90%`) tracing structural outlines in luminous `Ice Crystal` neon characters. |
| **Vaporwave Sunset** | `ASCII Magic Studio` | Dense block glyphs (`░▒▓█`) over a violet-tinted blurred backdrop with `Vaporwave` magenta-aqua gradient, scanlines, and chromatic shift. |
| **Film Noir Grain** | `ASCII Magic Studio` | High-contrast monochrome `Detailed` ramp over a desaturated backdrop (`0%` saturation) with organic film grain (`28%`) and deep radial vignette (`45%`). |
| **Scattered Lumina (VFX)** | `Layered Glyph VFX` | Up to 2,800 Poisson-disk scattered glyph instances aligned to `Image Gradient Tangent` with animated harmonic wind sway and emissive bloom. |

## VII. Studio Controls & Multi-Script Typography

### Sidebar Control Modules

| Sidebar Module | Parameters & Ranges | Optical Effect |
| :--- | :--- | :--- |
| **1. Background Image & Blur** | `Backdrop` (`Blurred Image`, `Original Image`, `Frosted Glass`, `Solid Color`), `Bg Blur` (`0–100`), `Bg Opacity` (`0–100%`), `Bg Exposure` (`0–200%`), `Bg Contrast` (`20–200%`), `Bg Saturation` (`0–200%`), `Bg Tint` | Grades, blurs, or tints the underlying full-resolution video/image frame before ASCII compositing. |
| **2. Object Detection & Mask** | `Detect Filter` (`Luminance + Edges`, `Luminance Threshold`, `Salient Object`, `Edges Only`, `Full Frame`), `Dark Cutoff` (`0–100`), `Light Cutoff` (`0–100`), `Edge Feather` (`0–50`), `Edge Emphasis` (`0–100%`), `Coverage` (`0–100%`), `Invert Mask` | Isolates illuminated subjects or structural contours and thins glyph density via 8×8 Bayer ordered dithering. |
| **3. Luminosity & Gradient** | `Blend Mode` (`Screen`, `Add`, `Color Dodge`, `Overlay`, `Soft Light`, `Normal`), `ASCII Opacity` (`0–100%`), `Luminosity` (`50–300%`), `Hot-White Core` (`0–100%`), `Tone Contrast` (`20–200%`), `Char Saturation` (`0–200%`), `Gradient Map`, `Gradient Mix` (`0–100%`), `Custom Stops` | Controls glyph radiance, incandescent highlight centers, 256-entry gradient LUT mixing, and backdrop blend math. |
| **4. Character Bloom & Post-FX** | `Char Bloom` (`0–100%`), `Char Glow Rad` (`1–50px`), `Scene Bloom` (`0–100%`), `Bloom Cutoff` (`0–100`), `Chromatic Abr` (`0–20px`), `CRT Scanlines` (`0–100%`), `Film Grain` (`0–100%`), `Vignette` (`0–100%`) | Generates multi-scale emissive halos around individual character strokes and applies final optical post-processing. |
| **5. Scattered Glyph Instances** | `Distribution` (`Poisson Disk`, `Jittered Grid`), `Density` (`0–100%`), `Max Glyphs` (`50–6000`), `Glyph Scale` (`0.25–4.0x`), `Size Jitter` (`0–100%`), `Orientation` (`Fixed`, `Image Gradient Tangent`, `Image Gradient Normal`, `Random`), `Wind Sway` (`0–100%`), `Outline` | Controls spatial particle distribution, contour-following rotation, and harmonic wind sway in `Layered Glyph VFX` mode. |
| **6. Grid & Character Sets** | `Width` (`20–800` cols), `Height`, `Aspect Ratio` (`Source`, `1:1`, `4:3`, `3:2`, `16:9`, `16:10`, `21:9`, `Custom`), `Preset` (10 built-in ramps or Custom string), `Export Font Size` (`4–32px`) | Configures character grid resolution, aspect ratio locking, Latin/CJK glyph ramps, and export rasterization scale. |

### Built-In Character Ramps (Sorted Sparse to Dense by Measured Pixel Coverage)

| Ramp Name | Character Sequence | Script & Visual Character |
| :--- | :--- | :--- |
| **ASCII Magic** | `" .,:-+=xX80S#@"` | Balanced 14-step studio ramp optimized for emissive bloom and contour legibility. |
| **Standard** | `" .:,*;?+S%#@"` | Classic 12-step typographic ramp for general video and portrait rendering. |
| **Detailed** | `" .':,;+=-*xX08S#$&%@"` | Extended 20-step fine tonal ramp ideal for monochrome and chiaroscuro studies. |
| **Minimal** | `" .:-=+*#%@"` | Clean 10-step geometric ramp with high negative space. |
| **Dense** | `" ░▒▓█"` | Unicode shade blocks for smooth halftone and vaporwave gradients. |
| **Simple** | `" █"` | High-contrast binary solid block mask. |
| **Binary** | `" 01"` | Digital binary stream ramp. |
| **Japanese** | `" 一いアんカの木水あ月日足黎達機鬱響"` | 17-step Hiragana, Katakana, and Kanji ramp with automatic CJK font routing. |
| **Chinese** | `" 二上川土女口王字金正右草海織龍鬱響"` | 17-step Hanzi stroke-density ramp from sparse (`二`) to ultra-dense (`響`). |
| **Best Mix** | `" :,*しア;+小のき大水口あ#海愛@響"` | Hybrid Latin + CJK ramp combining fine punctuation with complex ideographs. |

## VIII. Export Formats Matrix

All visual exports (`MP4` and `PNG`) render through the exact active studio or VFX pipeline at your chosen **Export Font Size** (`4px` to `32px`), independent of the preview window size.

| Export Action | Output Format | Supported Modes | Details |
| :--- | :--- | :--- | :--- |
| **Export as ASCII MP4** | `.mp4` (H.264 / MPEG-4) | `ASCII Magic Studio`, `Layered Glyph VFX`, `Classic Grid` | Renders every frame of the source video with full backdrop grading, object masking, bloom, and post-FX. |
| **Export as ASCII PNG** | `.png` (24-bit RGB) | `ASCII Magic Studio`, `Layered Glyph VFX`, `Classic Grid` | Saves the active video frame or still image at full export font resolution with all optical effects applied. |
| **Save ASCII Text** | `.txt` (Frame Sequence) | `ASCII Magic Studio`, `Classic Grid` | Writes all video frames as plain text (`FRAME_START` / `FRAME_END`), respecting active object detection masking. |
| **Save Current Frame** | `.txt` / `.html` | `ASCII Magic Studio`, `Classic Grid` | Exports the current frame as plain text or self-contained per-character colored HTML. |
| **Export Frame as HTML** | `.html` (Styled DOM) | `ASCII Magic Studio`, `Classic Grid` | Generates a standalone dark-canvas HTML document with inline RGB `<span>` styling at the selected font size. |

## IX. Installation & Quickstart

### Option 1 — Standalone Windows Installer

Download `VideoToASCII_Setup.exe` from the [Releases](https://www.google.com/search?q=https://github.com/Fearcine/Video-to-Ascii-convertor/releases) page and launch the executable directly. No Python environment setup is required.

### Option 2 — Run from Source (Windows, macOS, Linux)

Requires **Python 3.11+** and a standard desktop environment.

```bash
git clone https://github.com/Fearcine/Video-to-Ascii-convertor.git
cd Video-to-Ascii-convertor
pip install -r requirements.txt
python main.py
```

#### Studio Workflow

1. Launch `python main.py` and click **Continue** on the splash screen (click **Open Video...** or **Open Image...** to load media in `.mp4`, `.avi`, `.mov`, `.mkv`, `.webm`, `.png`, `.jpg`, `.bmp`, `.webp`, or `.gif` format).
2. In **Engine Mode & ASCII Magic Recipes**, select `ASCII Magic Studio` or `Layered Glyph VFX` and pick a recipe from **One-Click Recipe** (such as `Cloud Glow (ASCII Magic)`, `Solar Flare Gold`, or `Scattered Lumina (VFX)`).
3. Adjust any backdrop, detection cutoff, luminosity, gradient, or bloom slider — when paused or in image mode, the canvas re-renders immediately on every slider adjustment.
4. Set your desired **Export Font (px)** and click **Export as ASCII MP4** or **Export as ASCII PNG**.

## X. Headless Python API

Because `vfx/` has zero Qt dependencies, you can drive both `ASCII Magic Studio` and `Layered Glyph VFX` directly from Python scripts, notebooks, or automated video processing pipelines:

```python
import cv2
from ascii_renderer import CHAR_SETS
from vfx import get_vfx_engine, get_preset_settings

# Load any BGR image or video frame
frame_bgr = cv2.imread("assets/demo-nascar.png")

# Load a curated preset (or construct VfxSettings(...) directly)
vfx = get_preset_settings("Solar Flare Gold")
engine = get_vfx_engine()

# Render ASCII Magic Studio frame (returns RGB uint8 ndarray, 2D chars, 2D RGB colors)
rgb_out, chars_2d, colors_rgb = engine.render_ascii_magic_studio(
    frame_bgr=frame_bgr,
    width=200,
    height=72,
    char_set=CHAR_SETS["ASCII Magic"],
    font_size=12,
    charset_hint="",
    color_mode="Colored",
    intensity=100,
    brightness=100,
    invert_ascii=False,
    mono_color=(255, 255, 255),
    bg_color=(14, 14, 14),
    vfx=vfx,
    frame_index=0,
)

cv2.imwrite("output_studio.png", cv2.cvtColor(rgb_out, cv2.COLOR_RGB2BGR))
```

To render scattered particle instances programmatically, call `engine.render_scattered_vfx(...)` with `get_preset_settings("Scattered Lumina (VFX)")` and pass `frame_index` and `fps` for deterministic wind animation across video frames.

## XI. Performance Benchmarks & Verification Suite

The repository includes a characterization and unit test suite (`50+` automated tests covering classic rendering determinism, `GlyphAtlas` compositing, `VfxSettings` clamping/serialization, backdrop grading, salient object masking, luminosity curves, per-character bloom, and scattered instance identity stability) along with headless frame-time benchmarks:

| Pipeline Mode | Grid / Instance Resolution | Mean Frame Time (CPU) | Peak Allocation |
| :--- | :--- | :--- | :--- |
| **Classic Grid** | `200 × 100` chars | `57.5 ms / frame` | `< 15 MB` (reused buffer) |
| **Classic Grid** | `400 × 200` chars | `78.5 ms / frame` | `< 25 MB` (reused buffer) |
| **ASCII Magic Studio** | `200 × 75` chars (Full Backdrop + Saliency + Bloom) | `226.3 ms / frame` | `62.6 MB` |
| **Layered Glyph VFX** | `200 × 75` field (`2,800` rotated sprites + Bloom) | `263.8 ms / frame` | `47.4 MB` |

```bash
# Run the complete unit and characterization test suite
pytest -v

# Benchmark Classic Grid vs. ASCII Magic Studio & Layered Glyph VFX pipelines
python tools/bench_classic.py
python tools/bench_vfx.py
```

## XII. System Requirements & License

- **Operating System**: Windows 10 / 11 (64-bit), macOS 12+, or Linux (X11 / Wayland)
- **Runtime**: Python `3.11+` (`PyQt6 >= 6.6.0`, `opencv-python >= 4.9.0`, `numpy >= 1.26.0`, `Pillow >= 10.2.0`)
- **Memory**: 8 GB RAM recommended (16 GB recommended for ultra-wide grids above 400 columns)
- **License**: Released under the [MIT License](https://www.google.com/search?q=LICENSE). Free for personal, academic, and commercial creative production.