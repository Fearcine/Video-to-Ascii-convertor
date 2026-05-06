import numpy as np
import cv2
from PIL import Image
from glyph_atlas import get_atlas

CHAR_SETS = {
    "Standard": " .,:;+*?%S#@",
    "Dense": " ░▒▓█",
    "Simple": " █",
}


def frame_to_ascii(
    frame_bgr: np.ndarray,
    width: int,
    height: int,
    char_set: str,
    color_mode: str,
    intensity: int,
    mono_color: tuple[int, int, int] = (255, 255, 255),
) -> tuple[np.ndarray, np.ndarray]:
    if frame_bgr is None or len(char_set) == 0:
        empty_c = np.full((1, 1), " ", dtype="<U1")
        empty_rgb = np.zeros((1, 1, 3), dtype=np.uint8)
        return empty_c, empty_rgb

    resized = cv2.resize(frame_bgr, (width, height), interpolation=cv2.INTER_AREA)
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32)

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


def image_to_ascii(
    image_path: str,
    width: int,
    char_set: str,
    color_mode: str,
    intensity: int,
    mono_color: tuple[int, int, int] = (255, 255, 255),
    aspect_ratio: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    frame_bgr = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if frame_bgr is None:
        raise IOError(f"Cannot load image: {image_path}")

    if aspect_ratio is None:
        h_img, w_img = frame_bgr.shape[:2]
        aspect_ratio = w_img / h_img if h_img > 0 else 1.77

    height = max(1, int(width / aspect_ratio * 0.5))

    return frame_to_ascii(frame_bgr, width, height, char_set, color_mode, intensity, mono_color)


def render_to_rgb(
    chars_2d: np.ndarray,
    colors_rgb: np.ndarray,
    font_size: int = 12,
    bg_color: tuple[int, int, int] = (17, 17, 17),
    char_set: str = " .,:;+*?%S#@",
    out_buf: np.ndarray | None = None,
) -> np.ndarray:
    
    # Collect unique chars for atlas
    atlas = get_atlas(char_set, font_size)
    return atlas.compose_frame(chars_2d, colors_rgb, bg_color, out_buf)


def render_to_pil(
    chars_2d: np.ndarray,
    colors_rgb: np.ndarray,
    font_size: int = 12,
    bg_color: tuple[int, int, int] = (17, 17, 17),
    char_set: str = " .,:;+*?%S#@",
) -> Image.Image:
    
    rgb_array = render_to_rgb(chars_2d, colors_rgb, font_size, bg_color, char_set)
    return Image.fromarray(rgb_array, "RGB")


def render_to_cv2(
    chars_2d: np.ndarray,
    colors_rgb: np.ndarray,
    font_size: int = 12,
    bg_color: tuple[int, int, int] = (17, 17, 17),
    char_set: str = " .,:;+*?%S#@",
    out_buf: np.ndarray | None = None,
) -> np.ndarray:
   
    rgb = render_to_rgb(chars_2d, colors_rgb, font_size, bg_color, char_set, out_buf)
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def frame_to_plain_text(chars_2d: np.ndarray) -> str:
    
    lines = []
    for y in range(chars_2d.shape[0]):
        lines.append("".join(chars_2d[y].tolist()))
    return "\n".join(lines)


def frame_to_html(
    chars_2d: np.ndarray,
    colors_rgb: np.ndarray,
    font_size: int = 8,
) -> str:
    
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
