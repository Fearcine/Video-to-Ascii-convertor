"""Shared settings containers used across render, preview, and export pipelines."""

from dataclasses import dataclass, field


@dataclass(frozen=True, slots=True)
class RenderSettings:
    """Immutable snapshot of all settings that affect ASCII rendering.

    Used in frame_to_ascii, export threads, export_full_html, and the
    GUI ↔ RenderThread settings pipeline.  Being frozen + slotted keeps
    it lightweight and safe to pass between threads.
    """

    width: int = 200
    height: int = 100
    char_set: str = " .,:;+*?%S#@"
    color_mode: str = "Colored"  # "Colored" | "Grayscale" | "Monochrome"
    intensity: int = 100
    brightness: int = 100
    mono_color: tuple[int, int, int] = (255, 255, 255)
    bg_color: tuple[int, int, int] = (14, 14, 14)
    speed: float = 1.0
    aspect_lock: bool = True
    aspect_preset: str = "Source"  # "Source"|"1:1"|"4:3"|"3:2"|"16:9"|"16:10"|"21:9"|"Custom"
    loop: bool = True
    font_size: int = 8
