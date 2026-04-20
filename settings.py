"""
Settings persistence — saves/loads all UI state to settings.json.
File is stored next to the executable (or script).
"""

import json
import os
import sys

_DEFAULTS = {
    "width": 200,
    "height": 100,
    "aspect_lock": True,
    "char_set_name": "Standard",
    "custom_chars": "",
    "color_mode": "Colored",
    "mono_color": [255, 255, 255],
    "intensity": 80,
    "speed": 1.0,
    "font_size": 8,
    "last_video": "",
    "window_width": 1400,
    "window_height": 900,
}


def _settings_path() -> str:
    if getattr(sys, "frozen", False):
        base = os.path.dirname(sys.executable)
    else:
        base = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(base, "settings.json")


def load_settings() -> dict:
    """Load settings from JSON file, falling back to defaults for missing keys."""
    path = _settings_path()
    settings = dict(_DEFAULTS)
    if os.path.isfile(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, dict):
                for k, v in data.items():
                    if k in settings:
                        settings[k] = v
        except (json.JSONDecodeError, OSError, ValueError):
            pass  # Corrupted file — use defaults
    return settings


def save_settings(settings: dict):
    """Persist current settings to disk. Silently ignores write errors."""
    path = _settings_path()
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(settings, f, indent=2, ensure_ascii=False)
    except OSError:
        pass
