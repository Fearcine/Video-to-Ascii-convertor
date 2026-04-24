import numpy as np
from PIL import Image, ImageDraw, ImageFont
from typing import Optional


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


class GlyphAtlas:
    

    def __init__(self, char_set: str, font_size: int, font_name: str = "consola.ttf"):
        self.char_set = char_set
        self.font_size = font_size

        font = _get_font(font_size)

        # Determine cell dimensions from reference character
        ref_bbox = font.getbbox("M")
        self.cell_w = max(1, ref_bbox[2] - ref_bbox[0])
        ascent, descent = font.getmetrics()
        self.cell_h = max(1, ascent + descent)

        # Build character-to-index mapping
        unique_chars = sorted(set(char_set))
        self._char_to_idx: dict[str, int] = {}
        for i, ch in enumerate(unique_chars):
            self._char_to_idx[ch] = i

        # Pre-render each character as an alpha mask: shape (n_chars, cell_h, cell_w)
        n = len(unique_chars)
        self._alpha_masks = np.zeros((n, self.cell_h, self.cell_w), dtype=np.float32)

        for i, ch in enumerate(unique_chars):
            if ch == " ":
                continue  # Space stays all zeros — nothing to render
            img = Image.new("L", (self.cell_w, self.cell_h), 0)
            draw = ImageDraw.Draw(img)
            # Center the character horizontally
            ch_bbox = font.getbbox(ch)
            ch_w = ch_bbox[2] - ch_bbox[0]
            x_offset = max(0, (self.cell_w - ch_w) // 2)
            draw.text((x_offset, 0), ch, fill=255, font=font)
            self._alpha_masks[i] = np.array(img, dtype=np.float32) / 255.0

        # Fallback index for unknown characters (space)
        self._space_idx = self._char_to_idx.get(" ", 0)

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
    
        rows, cols = chars_2d.shape
        img_h = rows * self.cell_h
        img_w = cols * self.cell_w

        # Allocate or reuse output buffer
        if out_buf is not None and out_buf.shape == (img_h, img_w, 3):
            out = out_buf
            out[:] = bg_color
        else:
            out = np.full((img_h, img_w, 3), bg_color, dtype=np.uint8)

        # Map characters to atlas indices
        indices = self._chars_to_indices(chars_2d)

        # Gather all alpha masks: shape (rows, cols, cell_h, cell_w)
        alpha_all = self._alpha_masks[indices]  # fancy indexing

        # Process in row-strips to keep memory bounded
        colors_f = colors_rgb.astype(np.float32)

        for y in range(rows):
            y_px = y * self.cell_h
            row_alpha = alpha_all[y]  # (cols, cell_h, cell_w)
            row_colors = colors_f[y]  # (cols, 3)

            # Expand colors to match glyph shape: (cols, cell_h, cell_w, 3)
            colored_glyphs = row_alpha[:, :, :, np.newaxis] * row_colors[:, np.newaxis, np.newaxis, :]

            # Reshape row of glyphs into a horizontal strip
            # row_alpha shape: (cols, cell_h, cell_w) → strip: (cell_h, cols*cell_w)
            strip = colored_glyphs.transpose(1, 0, 2, 3).reshape(self.cell_h, cols * self.cell_w, 3)

            # Alpha-blend onto background
            bg_strip = out[y_px : y_px + self.cell_h, :, :].astype(np.float32)
            row_alpha_strip = row_alpha.transpose(1, 0, 2).reshape(self.cell_h, cols * self.cell_w)
            alpha_3 = row_alpha_strip[:, :, np.newaxis]

            out[y_px : y_px + self.cell_h, :, :] = (
                strip + bg_strip * (1.0 - alpha_3)
            ).clip(0, 255).astype(np.uint8)

        return out


# ── Atlas cache ───────────────────────────────────────────────────────

_atlas_cache: dict[tuple[str, int], GlyphAtlas] = {}


def get_atlas(char_set: str, font_size: int) -> GlyphAtlas:
    
    key = (char_set, font_size)
    if key not in _atlas_cache:
        _atlas_cache[key] = GlyphAtlas(char_set, font_size)
    return _atlas_cache[key]


def clear_atlas_cache():
   
    _atlas_cache.clear()
