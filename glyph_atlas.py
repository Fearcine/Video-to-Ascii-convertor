"""Pre-rendered glyph atlas for fast ASCII art compositing.

Rasterizes each character in a character set into alpha masks at a given
font size, then composites colored text onto a frame buffer using vectorized
integer arithmetic.  Supports per-character font routing so CJK glyphs
render with appropriate fonts while Latin characters use monospace.
"""

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from typing import Optional


# ---------------------------------------------------------------------------
# Font management: Latin monospace + CJK font selection per character
# ---------------------------------------------------------------------------

_font_cache: dict[tuple[str, int], ImageFont.FreeTypeFont] = {}

# Latin monospace fonts (tried in order)
_LATIN_FONTS = ("consola.ttf", "cour.ttf", "lucon.ttf")

# Japanese-specific fonts (Gothic family — angular strokes, distinct kana)
_JP_FONTS = ("YuGothR.ttc", "msgothic.ttc", "meiryo.ttc")

# Chinese-specific fonts (Song / sans-serif — rounder, heavier strokes)
_CN_FONTS = ("msyh.ttc", "simsun.ttc", "simhei.ttf")

# Generic CJK fallback (tried after specific JP/CN fonts fail)
_CJK_FALLBACK = ("msyh.ttc", "msgothic.ttc", "YuGothR.ttc", "simsun.ttc")


def _load_font(names: tuple[str, ...], size: int) -> ImageFont.FreeTypeFont:
    """Try loading fonts from a list, return first success or default."""
    key = (names[0], size)
    if key in _font_cache:
        return _font_cache[key]
    font = None
    for name in names:
        try:
            font = ImageFont.truetype(name, size)
            break
        except (OSError, IOError):
            continue
    if font is None:
        try:
            font = ImageFont.load_default(size=size)
        except TypeError:
            font = ImageFont.load_default()
    _font_cache[key] = font
    return font


def _get_latin_font(size: int) -> ImageFont.FreeTypeFont:
    """Get the best available Latin monospace font at given size."""
    return _load_font(_LATIN_FONTS, size)


def _get_jp_font(size: int) -> ImageFont.FreeTypeFont:
    """Get the best available Japanese font at given size."""
    return _load_font(_JP_FONTS, size)


def _get_cn_font(size: int) -> ImageFont.FreeTypeFont:
    """Get the best available Chinese font at given size."""
    return _load_font(_CN_FONTS, size)


def _get_cjk_font(size: int) -> ImageFont.FreeTypeFont:
    """Get a generic CJK fallback font at given size."""
    return _load_font(_CJK_FALLBACK, size)


def _is_cjk(ch: str) -> bool:
    """Check if a character needs a CJK font to render properly."""
    cp = ord(ch)
    return (
        0x3040 <= cp <= 0x309F   # Hiragana
        or 0x30A0 <= cp <= 0x30FF  # Katakana
        or 0x31F0 <= cp <= 0x31FF  # Katakana Phonetic Extensions
        or 0x3400 <= cp <= 0x4DBF  # CJK Extension A
        or 0x4E00 <= cp <= 0x9FFF  # CJK Unified Ideographs
        or 0xF900 <= cp <= 0xFAFF  # CJK Compatibility Ideographs
        or 0xFF00 <= cp <= 0xFFEF  # Halfwidth and Fullwidth Forms
        or 0x20000 <= cp <= 0x2A6DF  # CJK Extension B
    )


def _is_japanese_kana(ch: str) -> bool:
    """Check if a character is Hiragana or Katakana (distinctly Japanese)."""
    cp = ord(ch)
    return (
        0x3040 <= cp <= 0x309F   # Hiragana
        or 0x30A0 <= cp <= 0x30FF  # Katakana
        or 0x31F0 <= cp <= 0x31FF  # Katakana Phonetic Extensions
    )


def _get_font_for_char(ch: str, size: int, charset_hint: str = "") -> ImageFont.FreeTypeFont:
    """Select the appropriate font for a character.

    Uses charset_hint (the name of the active character set) to disambiguate
    shared CJK ideographs between Japanese and Chinese font families.
    Kana characters always use the Japanese font regardless of hint.
    """
    if not _is_cjk(ch):
        return _get_latin_font(size)

    # Kana → always Japanese font
    if _is_japanese_kana(ch):
        return _get_jp_font(size)

    # Use charset hint to pick the right font family for shared ideographs
    hint_lower = charset_hint.lower()
    if "japanese" in hint_lower or "jp" in hint_lower:
        return _get_jp_font(size)
    elif "chinese" in hint_lower or "cn" in hint_lower:
        return _get_cn_font(size)

    # Generic CJK fallback (e.g. "Best Mix")
    return _get_cjk_font(size)


# ---------------------------------------------------------------------------
# Glyph coverage measurement (used to build sorted character ramps)
# ---------------------------------------------------------------------------

def measure_glyph_coverage(chars: str, font_size: int = 10) -> dict[str, float]:
    """Render each character and measure its alpha-mask pixel coverage.

    Returns a dict mapping each character to its coverage (0.0 = empty, 1.0 = solid).
    """
    latin_font = _get_latin_font(font_size)
    ref_bbox = latin_font.getbbox("M")
    cell_w = max(1, ref_bbox[2] - ref_bbox[0])
    ascent, descent = latin_font.getmetrics()
    cell_h = max(1, ascent + descent)

    result: dict[str, float] = {}
    for ch in chars:
        if ch == " ":
            result[ch] = 0.0
            continue

        font = _get_font_for_char(ch, font_size)

        if _is_cjk(ch):
            cjk_ascent, cjk_descent = font.getmetrics()
            cjk_h = max(1, cjk_ascent + cjk_descent)
            ch_bbox = font.getbbox(ch)
            cjk_w = max(1, ch_bbox[2] - ch_bbox[0])
            render_w = max(cjk_w, cell_w)
            render_h = max(cjk_h, cell_h)

            img = Image.new("L", (render_w, render_h), 0)
            draw = ImageDraw.Draw(img)
            x_off = max(0, (render_w - cjk_w) // 2)
            draw.text((x_off, 0), ch, fill=255, font=font)
            img = img.resize((cell_w, cell_h), Image.Resampling.LANCZOS)
        else:
            img = Image.new("L", (cell_w, cell_h), 0)
            draw = ImageDraw.Draw(img)
            ch_bbox = font.getbbox(ch)
            ch_w = ch_bbox[2] - ch_bbox[0]
            x_offset = max(0, (cell_w - ch_w) // 2)
            draw.text((x_offset, 0), ch, fill=255, font=font)

        arr = np.array(img, dtype=np.float32) / 255.0
        result[ch] = float(arr.mean())

    return result


def sort_chars_by_coverage(chars: str, font_size: int = 10) -> str:
    """Sort characters from lowest to highest pixel coverage (sparse → dense)."""
    coverage = measure_glyph_coverage(chars, font_size)
    sorted_chars = sorted(coverage.keys(), key=lambda c: coverage[c])
    return "".join(sorted_chars)


# ---------------------------------------------------------------------------
# GlyphAtlas: rasterizes characters, composites colored text onto frame buffer
# ---------------------------------------------------------------------------

# CJK render scale: render at 3× then downscale for better stroke preservation
_CJK_SUPERSAMPLE = 3


class GlyphAtlas:
    """Pre-renders character glyphs as alpha masks for fast compositing.

    CJK glyphs are rendered at a higher resolution (_CJK_SUPERSAMPLE×) and
    downscaled with LANCZOS to preserve stroke detail at small font sizes.
    Japanese and Chinese characters use distinct font families for visual
    differentiation.
    """

    def __init__(self, char_set: str, font_size: int, charset_hint: str = ""):
        self.char_set = char_set
        self.font_size = font_size
        self._charset_hint = charset_hint

        latin_font = _get_latin_font(font_size)

        # Cell dimensions from Latin font's 'M' — all chars fit this cell
        ref_bbox = latin_font.getbbox("M")
        self.cell_w = max(1, ref_bbox[2] - ref_bbox[0])
        ascent, descent = latin_font.getmetrics()
        self.cell_h = max(1, ascent + descent)

        # Build per-character alpha masks
        unique_chars = sorted(set(char_set))
        self._char_to_idx: dict[str, int] = {}
        for i, ch in enumerate(unique_chars):
            self._char_to_idx[ch] = i

        n = len(unique_chars)
        self._alpha_masks = np.zeros((n, self.cell_h, self.cell_w), dtype=np.float32)

        for i, ch in enumerate(unique_chars):
            if ch == " ":
                continue

            font = _get_font_for_char(ch, font_size, charset_hint)

            if _is_cjk(ch):
                # Render CJK at supersample resolution for stroke clarity
                ss = _CJK_SUPERSAMPLE
                hi_size = font_size * ss
                hi_font = _get_font_for_char(ch, hi_size, charset_hint)

                cjk_ascent, cjk_descent = hi_font.getmetrics()
                cjk_h = max(1, cjk_ascent + cjk_descent)
                ch_bbox = hi_font.getbbox(ch)
                cjk_w = max(1, ch_bbox[2] - ch_bbox[0])
                render_w = max(cjk_w, self.cell_w * ss)
                render_h = max(cjk_h, self.cell_h * ss)

                img = Image.new("L", (render_w, render_h), 0)
                draw = ImageDraw.Draw(img)
                x_off = max(0, (render_w - cjk_w) // 2)
                draw.text((x_off, 0), ch, fill=255, font=hi_font)
                # Downscale to cell size with high-quality resampling
                img = img.resize((self.cell_w, self.cell_h), Image.Resampling.LANCZOS)
            else:
                img = Image.new("L", (self.cell_w, self.cell_h), 0)
                draw = ImageDraw.Draw(img)
                ch_bbox = font.getbbox(ch)
                ch_w = ch_bbox[2] - ch_bbox[0]
                x_offset = max(0, (self.cell_w - ch_w) // 2)
                draw.text((x_offset, 0), ch, fill=255, font=font)

            self._alpha_masks[i] = np.array(img, dtype=np.float32) / 255.0

        self._space_idx = self._char_to_idx.get(" ", 0)

        # Pre-compute uint8 alpha masks (0-255) for integer compositing
        self._alpha_masks_u8 = (self._alpha_masks * 255).astype(np.uint8)

    def _chars_to_indices(self, chars_2d: np.ndarray) -> np.ndarray:
        h, w = chars_2d.shape
        indices = np.full((h, w), self._space_idx, dtype=np.int32)
        for ch, idx in self._char_to_idx.items():
            mask = chars_2d == ch
            indices[mask] = idx
        return indices

    def compose_frame(
        self,
        chars_2d: np.ndarray,
        colors_rgb: np.ndarray,
        bg_color: tuple[int, int, int] = (14, 14, 14),
        out_buf: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Composite colored ASCII glyphs onto a frame buffer.

        Uses proper integer alpha compositing: (fg*a + bg*(255-a) + 127) / 255
        The +127 provides rounding instead of truncation, preventing systematic
        dimming that the old >> 8 approach caused.
        """
        rows, cols = chars_2d.shape
        ch = self.cell_h
        cw = self.cell_w
        img_h = rows * ch
        img_w = cols * cw

        if out_buf is not None and out_buf.shape == (img_h, img_w, 3):
            out = out_buf
            out[:] = bg_color
        else:
            out = np.full((img_h, img_w, 3), bg_color, dtype=np.uint8)

        indices = self._chars_to_indices(chars_2d)
        alpha_all = self._alpha_masks_u8[indices]

        # Reshape alpha to full image grid
        alpha_img = alpha_all.transpose(0, 2, 1, 3).reshape(img_h, img_w)

        # Build foreground color image
        fg_colors = colors_rgb[:, np.newaxis, :, np.newaxis, :]
        fg_colors = np.broadcast_to(fg_colors, (rows, ch, cols, cw, 3))
        fg_img = fg_colors.reshape(img_h, img_w, 3)

        # Proper integer alpha compositing with rounding
        # result = (fg * alpha + bg * (255 - alpha) + 127) / 255
        alpha_3 = alpha_img[:, :, np.newaxis]  # (img_h, img_w, 1)
        a16 = alpha_3.astype(np.uint16)
        inv_a16 = np.uint16(255) - a16

        out[:] = (
            (fg_img.astype(np.uint16) * a16
             + out.astype(np.uint16) * inv_a16
             + np.uint16(127))  # Rounding term
            // np.uint16(255)
        ).astype(np.uint8)

        return out


# ---------------------------------------------------------------------------
# Atlas cache — keyed by (char_set, font_size, charset_hint)
# ---------------------------------------------------------------------------

_atlas_cache: dict[tuple[str, int, str], GlyphAtlas] = {}


def get_atlas(char_set: str, font_size: int, charset_hint: str = "") -> GlyphAtlas:
    """Get or create a cached GlyphAtlas for the given parameters."""
    key = (char_set, font_size, charset_hint)
    if key not in _atlas_cache:
        _atlas_cache[key] = GlyphAtlas(char_set, font_size, charset_hint)
    return _atlas_cache[key]


def clear_atlas_cache():
    _atlas_cache.clear()
