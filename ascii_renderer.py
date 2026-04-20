"""
ASCII Rendering Engine — NumPy vectorized, zero Python pixel loops.
Returns compact numpy arrays. Renders to PIL images for MP4 export.
"""

import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont

CHAR_SETS = {
    "Standard": " .,:;+*?%S#@",
    "Dense": " ░▒▓█",
    "Simple": " █",
}

_font_cache: dict[int, ImageFont.FreeTypeFont] = {}


def _get_font(size: int) -> ImageFont.FreeTypeFont:
    """Get a monospace PIL font at the given point size, cached."""
    if size in _font_cache:
        return _font_cache[size]
    font = None
    for name in ("consola.ttf", "cour.ttf", "lucon.ttf"):
        try:
            font = ImageFont.truetype(name, size)
            break
        except (OSError, IOError):
            continue
    if font is None:
        font = ImageFont.load_default()
    _font_cache[size] = font
    return font


def frame_to_ascii(
    frame_bgr: np.ndarray,
    width: int,
    height: int,
    char_set: str,
    color_mode: str,
    intensity: int,
    mono_color: tuple[int, int, int] = (255, 255, 255),
) -> tuple[np.ndarray, np.ndarray]:
    """
    Convert a BGR frame to ASCII representation.

    Returns:
        chars_2d:  np.ndarray dtype='<U1' shape (height, width)
        colors_rgb: np.ndarray dtype=uint8  shape (height, width, 3)
    """
    if frame_bgr is None or len(char_set) == 0:
        empty_c = np.full((1, 1), " ", dtype="<U1")
        empty_rgb = np.zeros((1, 1, 3), dtype=np.uint8)
        return empty_c, empty_rgb

    resized = cv2.resize(frame_bgr, (width, height), interpolation=cv2.INTER_AREA)
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32)

    # Vectorized luminance
    luminance = rgb[:, :, 0] * 0.299 + rgb[:, :, 1] * 0.587 + rgb[:, :, 2] * 0.114

    num_chars = len(char_set)
    indices = np.clip(
        (luminance / 255.0 * (num_chars - 1)).astype(np.int32), 0, num_chars - 1
    )

    char_array = np.array(list(char_set), dtype="<U1")
    chars_2d = char_array[indices]

    scale = intensity / 100.0

    if color_mode == "Colored":
        colors = (rgb * scale).clip(0, 255).astype(np.uint8)
    elif color_mode == "Grayscale":
        lum = (luminance * scale).clip(0, 255).astype(np.uint8)
        colors = np.stack([lum, lum, lum], axis=-1)
    else:  # Monochrome
        base = np.array(mono_color, dtype=np.float32) * scale
        base = np.clip(base, 0, 255).astype(np.uint8)
        colors = np.broadcast_to(base, (height, width, 3)).copy()

    return chars_2d, colors


def render_to_pil(
    chars_2d: np.ndarray,
    colors_rgb: np.ndarray,
    font_size: int = 12,
    bg_color: tuple[int, int, int] = (17, 17, 17),
) -> Image.Image:
    """
    Render colored ASCII art onto a PIL Image.
    Batches consecutive same-color characters per row for speed.
    """
    h, w = chars_2d.shape
    font = _get_font(font_size)

    # Cell dimensions from reference character
    ref_bbox = font.getbbox("M")
    cell_w = max(1, ref_bbox[2] - ref_bbox[0])
    ascent, descent = font.getmetrics()
    cell_h = max(1, ascent + descent)

    img = Image.new("RGB", (cell_w * w, cell_h * h), bg_color)
    draw = ImageDraw.Draw(img)

    for y in range(h):
        row_colors = colors_rgb[y]  # (w, 3)
        row_chars = chars_2d[y]  # (w,)
        py = y * cell_h

        # Detect color-run boundaries with numpy
        if w > 1:
            diffs = np.any(row_colors[1:] != row_colors[:-1], axis=1)
            breaks = np.where(diffs)[0] + 1
            boundaries = np.empty(len(breaks) + 2, dtype=np.intp)
            boundaries[0] = 0
            boundaries[1:-1] = breaks
            boundaries[-1] = w
        else:
            boundaries = np.array([0, w], dtype=np.intp)

        for i in range(len(boundaries) - 1):
            s = int(boundaries[i])
            e = int(boundaries[i + 1])
            color = (int(row_colors[s, 0]), int(row_colors[s, 1]), int(row_colors[s, 2]))
            text = "".join(row_chars[s:e].tolist())
            if text.strip():
                draw.text((s * cell_w, py), text, fill=color, font=font)

    return img


def render_to_cv2(
    chars_2d: np.ndarray,
    colors_rgb: np.ndarray,
    font_size: int = 12,
    bg_color: tuple[int, int, int] = (17, 17, 17),
) -> np.ndarray:
    """Render ASCII art and return a BGR numpy array for cv2.VideoWriter."""
    pil_img = render_to_pil(chars_2d, colors_rgb, font_size, bg_color)
    return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)


def frame_to_plain_text(chars_2d: np.ndarray) -> str:
    """Convert character array to plain text string."""
    lines = []
    for y in range(chars_2d.shape[0]):
        lines.append("".join(chars_2d[y].tolist()))
    return "\n".join(lines)


def frame_to_html(
    chars_2d: np.ndarray,
    colors_rgb: np.ndarray,
    font_size: int = 8,
) -> str:
    """Generate a standalone HTML document with inline-colored characters."""
    h, w = chars_2d.shape
    parts = [
        '<!DOCTYPE html>\n<html>\n<head><meta charset="utf-8"><title>ASCII Art</title></head>\n',
        f'<body style="background:#111;margin:0;padding:16px">\n',
        f'<pre style="font-family:\'Courier New\',monospace;font-size:{font_size}px;line-height:1.1">',
    ]
    esc = {"<": "&lt;", ">": "&gt;", "&": "&amp;"}
    for y in range(h):
        row_colors = colors_rgb[y]
        row_chars = chars_2d[y]

        # Batch by color runs
        if w > 1:
            diffs = np.any(row_colors[1:] != row_colors[:-1], axis=1)
            breaks = np.where(diffs)[0] + 1
            boundaries = np.empty(len(breaks) + 2, dtype=np.intp)
            boundaries[0] = 0
            boundaries[1:-1] = breaks
            boundaries[-1] = w
        else:
            boundaries = np.array([0, w], dtype=np.intp)

        for i in range(len(boundaries) - 1):
            s = int(boundaries[i])
            e = int(boundaries[i + 1])
            r, g, b = int(row_colors[s, 0]), int(row_colors[s, 1]), int(row_colors[s, 2])
            raw = "".join(row_chars[s:e].tolist())
            safe = "".join(esc.get(c, c) for c in raw)
            parts.append(f'<span style="color:rgb({r},{g},{b})">{safe}</span>')
        parts.append("\n")

    parts.append("</pre>\n</body>\n</html>")
    return "".join(parts)
