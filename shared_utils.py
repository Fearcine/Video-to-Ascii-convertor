"""Shared utility functions used across the Video-to-ASCII application."""


def get_preview_font_px(ascii_width: int) -> int:
    """Determine the preview font size (pixels) for a given ASCII grid width.

    Returns a smaller font for wider grids so the preview stays readable
    without the composited image becoming excessively large.

    This was previously duplicated in render_thread.py and main.py.
    """
    if ascii_width <= 150:
        return 10
    elif ascii_width <= 300:
        return 7
    elif ascii_width <= 500:
        return 5
    else:
        return 4
