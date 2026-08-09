"""Shared utility functions used across the Video-to-ASCII application."""


def get_preview_font_px(ascii_width: int) -> int:
    """Determine the preview font size (pixels) for a given ASCII grid width.

    Returns a smaller font for wider grids so the preview stays readable
    without the composited image becoming excessively large.

    This was previously duplicated in render_thread.py and main.py.
    """
    if ascii_width <= 60:
        return 32
    elif ascii_width <= 100:
        return 24
    elif ascii_width <= 150:
        return 16
    elif ascii_width <= 200:
        return 12
    elif ascii_width <= 300:
        return 8
    elif ascii_width <= 500:
        return 6
    else:
        return 4
