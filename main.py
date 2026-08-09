"""VideoToASCII — Video & Image to Colored ASCII Art."""

import sys
import os
import traceback
from PyQt6.QtWidgets import (
    QApplication,
    QMainWindow,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QPushButton,
    QSlider,
    QLabel,
    QComboBox,
    QRadioButton,
    QButtonGroup,
    QLineEdit,
    QFileDialog,
    QColorDialog,
    QProgressDialog,
    QMessageBox,
    QGroupBox,
    QStatusBar,
    QSizePolicy,
    QCheckBox,
    QFrame,
    QScrollArea,
)
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QFont, QIcon, QColor, QImage

from preview_widget import PreviewWidget
from render_thread import RenderThread
from settings import load_settings, save_settings
from export import (
    ExportVideoThread,
    ExportMP4Thread,
    save_current_frame_txt,
    save_current_frame_html,
    export_full_html,
)
from ascii_renderer import CHAR_SETS, image_to_ascii
from glyph_atlas import get_atlas
from shared_utils import get_preview_font_px
from render_settings import RenderSettings
from PyQt6.QtWidgets import QStackedWidget
from loading_screen import MatrixLoadingScreen
import numpy as np
import cv2


# ── Defaults ──────────────────────────────────────────────────────────────
DEMO_VIDEO = "demo_bad_apple.mp4"

ASPECT_PRESETS = {
    "Source":  None,
    "1:1":    1.0,
    "4:3":    4 / 3,
    "3:2":    3 / 2,
    "16:9":   16 / 9,
    "16:10":  16 / 10,
    "21:9":   21 / 9,
    "Custom": None,
}

# Default values for the Reset button
_RESET_DEFAULTS = {
    "width": 200,
    "height": 100,
    "aspect_preset": "Source",
    "char_set_name": "Standard",
    "custom_chars": "",
    "color_mode": "Colored",
    "intensity": 100,
    "brightness": 100,
    "invert_ascii": False,
    "bg_color": (14, 14, 14),
    "mono_color": (255, 255, 255),
    "speed": 1.0,
    "loop": True,
    "font_size": 8,
}


# ── Stylesheet ────────────────────────────────────────────────────────────
# Flat system-gray palette inspired by Win95/2000 tools + waifu2x.net
_BG = "#f0f0f0"
_BG_DARK = "#d4d0c8"
_BG_FIELD = "#ffffff"
_BORDER = "#808080"
_BORDER_LT = "#c0c0c0"
_TEXT = "#000000"
_TEXT_DIM = "#444444"
_LINK = "#0000cc"
_HIGHLIGHT = "#000080"
_HIGHLIGHT_TEXT = "#ffffff"
_BTN_FACE = "#e0e0e0"
_BTN_HOVER = "#d0d0d0"
_BTN_PRESSED = "#c0c0c0"
_SUNKEN = "border: 2px inset #a0a0a0;"
_RAISED = "border: 2px outset #d4d0c8;"

_GLOBAL_STYLE = f"""
QMainWindow {{
    background: {_BG};
}}
"""

_PANEL_STYLE = f"""
QWidget {{
    background: {_BG};
    color: {_TEXT};
    font-family: 'Tahoma', 'MS Sans Serif', 'Segoe UI', sans-serif;
    font-size: 11px;
}}
QGroupBox {{
    border: 2px groove {_BORDER_LT};
    border-radius: 0px;
    margin-top: 12px;
    padding: 12px 8px 8px 8px;
    font-weight: bold;
    font-size: 11px;
    color: {_TEXT};
}}
QGroupBox::title {{
    subcontrol-origin: margin;
    left: 8px;
    padding: 0 4px;
    background: {_BG};
}}
QPushButton {{
    background: {_BTN_FACE};
    {_RAISED}
    border-radius: 0px;
    padding: 4px 12px;
    color: {_TEXT};
    font-size: 11px;
    min-height: 18px;
}}
QPushButton:hover {{
    background: {_BTN_HOVER};
}}
QPushButton:pressed {{
    background: {_BTN_PRESSED};
    {_SUNKEN}
}}
QPushButton:checked {{
    background: {_BTN_PRESSED};
    {_SUNKEN}
}}
QSlider::groove:horizontal {{
    height: 4px;
    background: #a0a0a0;
    border: 1px inset #808080;
}}
QSlider::handle:horizontal {{
    width: 11px;
    height: 20px;
    margin: -8px 0;
    background: {_BTN_FACE};
    border: 1px outset #d4d0c8;
}}
QSlider::sub-page:horizontal {{
    background: {_HIGHLIGHT};
}}
QComboBox {{
    background: {_BG_FIELD};
    border: 1px inset #808080;
    border-radius: 0px;
    padding: 2px 6px;
    color: {_TEXT};
    font-size: 11px;
    min-height: 18px;
}}
QComboBox QAbstractItemView {{
    background: {_BG_FIELD};
    color: {_TEXT};
    selection-background-color: {_HIGHLIGHT};
    selection-color: {_HIGHLIGHT_TEXT};
    border: 1px solid {_BORDER};
}}
QComboBox::drop-down {{
    border: none;
    width: 18px;
}}
QComboBox::down-arrow {{
    image: none;
    border-left: 4px solid transparent;
    border-right: 4px solid transparent;
    border-top: 5px solid {_TEXT};
    margin-right: 4px;
}}
QRadioButton {{
    spacing: 6px;
    color: {_TEXT};
    font-size: 11px;
}}
QRadioButton::indicator {{
    width: 13px;
    height: 13px;
}}
QCheckBox {{
    spacing: 6px;
    color: {_TEXT};
    font-size: 11px;
}}
QCheckBox::indicator {{
    width: 13px;
    height: 13px;
}}
QLineEdit {{
    background: {_BG_FIELD};
    border: 1px inset #808080;
    border-radius: 0px;
    padding: 2px 4px;
    color: {_TEXT};
    font-size: 11px;
}}
QLabel {{
    color: {_TEXT};
    font-size: 11px;
}}
QScrollArea {{
    border: none;
    background: {_BG};
}}
QScrollBar:vertical {{
    background: {_BG};
    width: 14px;
    border: 1px inset #a0a0a0;
}}
QScrollBar::handle:vertical {{
    background: {_BTN_FACE};
    border: 1px outset #d4d0c8;
    min-height: 20px;
}}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{
    height: 14px;
    background: {_BTN_FACE};
    border: 1px outset #d4d0c8;
}}
QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical {{
    background: {_BG};
}}
"""

_STATUS_STYLE = f"""
QStatusBar {{
    background: {_BG_DARK};
    color: {_TEXT_DIM};
    font-size: 11px;
    padding: 2px 6px;
    border-top: 2px groove #c0c0c0;
    font-family: 'Tahoma', 'MS Sans Serif', monospace;
}}
"""


def _demo_video_path() -> str:
    """Return the absolute path to the bundled demo video."""
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), DEMO_VIDEO)


# ── Main window ───────────────────────────────────────────────────────────

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("VideoToASCII")
        self.setStyleSheet(_GLOBAL_STYLE)
        self._settings = load_settings()
        self.resize(
            self._settings.get("window_width", 1400),
            self._settings.get("window_height", 900),
        )
        self.setMinimumSize(900, 600)

        self._video_path = self._settings.get("last_video", "")
        self._image_path = ""
        self._mode = "video"
        self._current_chars: np.ndarray | None = None
        self._current_colors: np.ndarray | None = None
        self._current_frame_no = 0
        self._total_frames = 0
        self._video_fps = 24.0
        self._mono_color = tuple(self._settings.get("mono_color", [255, 255, 255]))
        self._bg_color = tuple(self._settings.get("bg_color", [14, 14, 14]))
        self._export_thread: ExportVideoThread | ExportMP4Thread | None = None

        # Debounce timer
        self._settings_timer = QTimer(self)
        self._settings_timer.setSingleShot(True)
        self._settings_timer.setInterval(100)
        self._settings_timer.timeout.connect(self._apply_settings)

        # Render thread
        self._render = RenderThread(self)
        self._render.frame_rendered.connect(self._on_frame_rendered)
        self._render.playback_finished.connect(self._on_playback_finished)
        self._render.error_occurred.connect(self._on_error)
        self._render.start()

        self._build_ui()
        self._restore_settings()
        self._push_settings_to_thread()

        # Load last video or demo
        if self._video_path and os.path.isfile(self._video_path):
            self._load_video(self._video_path)
        elif os.path.isfile(_demo_video_path()):
            self._load_video(_demo_video_path())
            # Demo auto-plays in _on_continue

    def _on_continue(self):
        """Switch to main UI and play demo if loaded."""
        self.stacked_widget.setCurrentIndex(1)
        if self._video_path and self._mode == "video":
            QTimer.singleShot(200, self._auto_play_demo)

    def _auto_play_demo(self):
        """Start playing the demo video automatically."""
        if self._video_path and self._mode == "video":
            self.btn_play.setChecked(True)
            self.btn_play.setText("Pause")
            self._render.play()

    # ── UI construction ───────────────────────────────────────────────────

    def _build_ui(self):
        self.stacked_widget = QStackedWidget()
        self.setCentralWidget(self.stacked_widget)
        
        self.loading_screen = MatrixLoadingScreen(self)
        self.loading_screen.btn_continue.clicked.connect(self._on_continue)
        self.stacked_widget.addWidget(self.loading_screen)
        
        central = QWidget()
        self.stacked_widget.addWidget(central)
        main_layout = QHBoxLayout(central)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        # ─── Left sidebar ─────────────────────────────────────────────────
        left_scroll = QScrollArea()
        left_scroll.setFixedWidth(290)
        left_scroll.setWidgetResizable(True)
        left_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)

        left_panel = QWidget()
        left_panel.setStyleSheet(_PANEL_STYLE)
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(8, 8, 8, 8)
        left_layout.setSpacing(4)

        # Title
        title = QLabel("VideoToASCII")
        title.setStyleSheet(
            "font-size: 14px; font-weight: bold; color: #000080; "
            "padding: 4px 0 2px 0;"
        )
        left_layout.addWidget(title)

        subtitle = QLabel("Convert video and images to colored ASCII art")
        subtitle.setWordWrap(True)
        subtitle.setStyleSheet("font-size: 10px; color: #444; padding: 0 0 4px 0;")
        left_layout.addWidget(subtitle)

        # Separator
        sep = QFrame()
        sep.setFrameShape(QFrame.Shape.HLine)
        sep.setStyleSheet("QFrame { color: #a0a0a0; }")
        left_layout.addWidget(sep)

        # ─── Input ────────────────────────────────────────────────────────
        grp_input = QGroupBox("Input")
        input_layout = QVBoxLayout(grp_input)
        input_layout.setSpacing(4)

        upload_row = QHBoxLayout()
        upload_row.setSpacing(4)
        self.btn_upload = QPushButton("Open Video...")
        self.btn_upload.clicked.connect(self._on_upload)
        upload_row.addWidget(self.btn_upload)

        self.btn_upload_image = QPushButton("Open Image...")
        self.btn_upload_image.clicked.connect(self._on_upload_image)
        upload_row.addWidget(self.btn_upload_image)
        input_layout.addLayout(upload_row)

        self.lbl_filename = QLabel("No file loaded")
        self.lbl_filename.setWordWrap(True)
        self.lbl_filename.setStyleSheet("color: #444; font-size: 10px; padding: 1px;")
        input_layout.addWidget(self.lbl_filename)

        left_layout.addWidget(grp_input)

        # ─── Resolution ──────────────────────────────────────────────────
        grp_res = QGroupBox("Resolution")
        res_layout = QVBoxLayout(grp_res)
        res_layout.setSpacing(3)

        ar_row = QHBoxLayout()
        ar_row.addWidget(QLabel("Aspect ratio:"))
        self.cmb_aspect = QComboBox()
        self.cmb_aspect.addItems(list(ASPECT_PRESETS.keys()))
        self.cmb_aspect.setCurrentText("Source")
        self.cmb_aspect.currentTextChanged.connect(self._on_aspect_preset_changed)
        ar_row.addWidget(self.cmb_aspect, 1)
        res_layout.addLayout(ar_row)

        w_row = QHBoxLayout()
        w_row.addWidget(QLabel("Width (chars):"))
        self.slider_width = QSlider(Qt.Orientation.Horizontal)
        self.slider_width.setRange(40, 1000)
        self.slider_width.setSingleStep(10)
        self.slider_width.setPageStep(50)
        self.slider_width.setValue(200)
        self.lbl_width = QLabel("200")
        self.lbl_width.setFixedWidth(32)
        self.lbl_width.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        w_row.addWidget(self.slider_width, 1)
        w_row.addWidget(self.lbl_width)
        res_layout.addLayout(w_row)

        h_row = QHBoxLayout()
        h_row.addWidget(QLabel("Height (chars):"))
        self.slider_height = QSlider(Qt.Orientation.Horizontal)
        self.slider_height.setRange(10, 500)
        self.slider_height.setSingleStep(5)
        self.slider_height.setPageStep(25)
        self.slider_height.setValue(100)
        self.lbl_height = QLabel("100")
        self.lbl_height.setFixedWidth(32)
        self.lbl_height.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        h_row.addWidget(self.slider_height, 1)
        h_row.addWidget(self.lbl_height)
        res_layout.addLayout(h_row)

        self.slider_width.valueChanged.connect(self._on_width_changed)
        self.slider_height.valueChanged.connect(self._on_height_changed)

        left_layout.addWidget(grp_res)

        # ─── Character Set ────────────────────────────────────────────────
        grp_chars = QGroupBox("Character Set")
        chars_layout = QVBoxLayout(grp_chars)
        chars_layout.setSpacing(3)

        cs_row = QHBoxLayout()
        cs_row.addWidget(QLabel("Preset:"))
        self.cmb_charset = QComboBox()
        self.cmb_charset.addItems([
            "Standard", "Dense", "Simple",
            "Japanese", "Chinese", "Best Mix",
            "Custom",
        ])
        cs_row.addWidget(self.cmb_charset, 1)
        chars_layout.addLayout(cs_row)

        self.txt_custom_chars = QLineEdit()
        self.txt_custom_chars.setPlaceholderText("Enter custom characters...")
        self.txt_custom_chars.setEnabled(False)
        chars_layout.addWidget(self.txt_custom_chars)

        self.chk_invert_ascii = QCheckBox("Inverted ASCII")
        chars_layout.addWidget(self.chk_invert_ascii)

        self.cmb_charset.currentTextChanged.connect(self._on_charset_changed)
        self.txt_custom_chars.textChanged.connect(self._on_setting_changed)
        self.chk_invert_ascii.toggled.connect(self._on_setting_changed)

        left_layout.addWidget(grp_chars)

        # ─── Color & Brightness ───────────────────────────────────────────
        grp_color = QGroupBox("Color Mode")
        color_layout = QVBoxLayout(grp_color)
        color_layout.setSpacing(3)

        self.radio_colored = QRadioButton("Colored")
        self.radio_gray = QRadioButton("Grayscale")
        self.radio_mono = QRadioButton("Monochrome")
        self.radio_colored.setChecked(True)

        self.color_group = QButtonGroup(self)
        self.color_group.addButton(self.radio_colored)
        self.color_group.addButton(self.radio_gray)
        self.color_group.addButton(self.radio_mono)

        mode_row = QHBoxLayout()
        mode_row.addWidget(self.radio_colored)
        mode_row.addWidget(self.radio_gray)
        color_layout.addLayout(mode_row)

        mono_row = QHBoxLayout()
        mono_row.addWidget(self.radio_mono)
        self.btn_mono_color = QPushButton()
        self.btn_mono_color.setFixedSize(22, 22)
        self.btn_mono_color.setToolTip("Pick monochrome color")
        self._update_mono_button_color()
        self.btn_mono_color.clicked.connect(self._on_pick_mono_color)
        mono_row.addWidget(self.btn_mono_color)
        mono_row.addStretch()
        color_layout.addLayout(mono_row)

        self.color_group.buttonToggled.connect(self._on_setting_changed)

        # Brightness
        br_row = QHBoxLayout()
        br_row.addWidget(QLabel("Brightness:"))
        self.slider_brightness = QSlider(Qt.Orientation.Horizontal)
        self.slider_brightness.setRange(20, 200)
        self.slider_brightness.setValue(100)
        self.lbl_brightness = QLabel("100%")
        self.lbl_brightness.setFixedWidth(32)
        self.lbl_brightness.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        br_row.addWidget(self.slider_brightness, 1)
        br_row.addWidget(self.lbl_brightness)
        color_layout.addLayout(br_row)
        self.slider_brightness.valueChanged.connect(self._on_brightness_changed)

        # Intensity
        int_row = QHBoxLayout()
        int_row.addWidget(QLabel("Intensity:"))
        self.slider_intensity = QSlider(Qt.Orientation.Horizontal)
        self.slider_intensity.setRange(0, 100)
        self.slider_intensity.setValue(100)
        self.lbl_intensity = QLabel("100%")
        self.lbl_intensity.setFixedWidth(32)
        self.lbl_intensity.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        int_row.addWidget(self.slider_intensity, 1)
        int_row.addWidget(self.lbl_intensity)
        color_layout.addLayout(int_row)
        self.slider_intensity.valueChanged.connect(self._on_intensity_changed)

        # Background color
        bg_row = QHBoxLayout()
        bg_row.addWidget(QLabel("Background:"))
        self.btn_bg_color = QPushButton()
        self.btn_bg_color.setFixedSize(22, 22)
        self.btn_bg_color.setToolTip("Pick background color")
        self._update_bg_button_color()
        self.btn_bg_color.clicked.connect(self._on_pick_bg_color)
        bg_row.addWidget(self.btn_bg_color)
        bg_row.addStretch()
        color_layout.addLayout(bg_row)

        left_layout.addWidget(grp_color)

        # ─── Playback ────────────────────────────────────────────────────
        grp_play = QGroupBox("Playback")
        play_layout = QVBoxLayout(grp_play)
        play_layout.setSpacing(3)

        btn_row = QHBoxLayout()
        btn_row.setSpacing(4)
        self.btn_play = QPushButton("Play")
        self.btn_play.setCheckable(True)
        self.btn_play.clicked.connect(self._on_play_toggle)
        btn_row.addWidget(self.btn_play)

        self.btn_stop = QPushButton("Stop")
        self.btn_stop.clicked.connect(self._on_stop)
        btn_row.addWidget(self.btn_stop)
        play_layout.addLayout(btn_row)

        opts_row = QHBoxLayout()
        opts_row.addWidget(QLabel("Speed:"))
        self.cmb_speed = QComboBox()
        self.cmb_speed.addItems(["0.25x", "0.5x", "1x", "2x", "4x"])
        self.cmb_speed.setCurrentText("1x")
        self.cmb_speed.currentTextChanged.connect(self._on_setting_changed)
        opts_row.addWidget(self.cmb_speed)
        opts_row.addSpacing(8)
        self.chk_loop = QCheckBox("Loop")
        self.chk_loop.setChecked(True)
        self.chk_loop.toggled.connect(self._on_setting_changed)
        opts_row.addWidget(self.chk_loop)
        play_layout.addLayout(opts_row)

        play_layout.addWidget(QLabel("Seek:"))
        self.slider_seek = QSlider(Qt.Orientation.Horizontal)
        self.slider_seek.setRange(0, 1)
        self.slider_seek.setValue(0)
        self.slider_seek.sliderPressed.connect(self._on_seek_pressed)
        self.slider_seek.sliderReleased.connect(self._on_seek_released)
        self.slider_seek.valueChanged.connect(self._on_seek_changed)
        self._seeking = False
        play_layout.addWidget(self.slider_seek)

        left_layout.addWidget(grp_play)

        # ─── Export ──────────────────────────────────────────────────────
        grp_out = QGroupBox("Export")
        out_layout = QVBoxLayout(grp_out)
        out_layout.setSpacing(3)

        fs_row = QHBoxLayout()
        fs_row.addWidget(QLabel("Font size:"))
        self.slider_fontsize = QSlider(Qt.Orientation.Horizontal)
        self.slider_fontsize.setRange(4, 16)
        self.slider_fontsize.setValue(8)
        self.slider_fontsize.setToolTip(
            "Font size for PNG/HTML/MP4 exports.\n"
            "Preview auto-sizes based on grid width."
        )
        self.lbl_fontsize = QLabel("8px")
        self.lbl_fontsize.setFixedWidth(28)
        self.lbl_fontsize.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        self.slider_fontsize.valueChanged.connect(
            lambda v: self.lbl_fontsize.setText(f"{v}px")
        )
        fs_row.addWidget(self.slider_fontsize, 1)
        fs_row.addWidget(self.lbl_fontsize)
        out_layout.addLayout(fs_row)

        self.btn_export_mp4 = QPushButton("Export as ASCII MP4")
        self.btn_export_mp4.clicked.connect(self._on_export_mp4)
        out_layout.addWidget(self.btn_export_mp4)

        self.btn_export_png = QPushButton("Export as ASCII PNG")
        self.btn_export_png.clicked.connect(self._on_export_png)
        out_layout.addWidget(self.btn_export_png)

        self.btn_save_video = QPushButton("Save ASCII Text (.txt)")
        self.btn_save_video.clicked.connect(self._on_save_video)
        out_layout.addWidget(self.btn_save_video)

        self.btn_save_frame = QPushButton("Save Current Frame")
        self.btn_save_frame.clicked.connect(self._on_save_frame)
        out_layout.addWidget(self.btn_save_frame)

        self.btn_export_html = QPushButton("Export Frame as HTML")
        self.btn_export_html.clicked.connect(self._on_export_html)
        out_layout.addWidget(self.btn_export_html)

        left_layout.addWidget(grp_out)

        # ─── Reset ───────────────────────────────────────────────────────
        left_layout.addSpacing(4)
        self.btn_reset = QPushButton("Reset to Defaults")
        self.btn_reset.setToolTip("Reset all settings and load the demo video")
        self.btn_reset.clicked.connect(self._on_reset)
        left_layout.addWidget(self.btn_reset)

        left_layout.addStretch()

        left_scroll.setWidget(left_panel)

        # ─── Right panel (preview) ────────────────────────────────────────
        right_panel = QWidget()
        right_panel.setStyleSheet("background: #0e0e0e;")
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(0)

        self.preview = PreviewWidget()
        self.preview.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        right_layout.addWidget(self.preview)

        self.status_bar = QStatusBar()
        self.status_bar.setStyleSheet(_STATUS_STYLE)
        self.status_bar.showMessage("Ready")
        right_layout.addWidget(self.status_bar)

        # Assemble
        main_layout.addWidget(left_scroll)

        divider = QFrame()
        divider.setFrameShape(QFrame.Shape.VLine)
        divider.setStyleSheet("QFrame { color: #808080; }")
        main_layout.addWidget(divider)

        main_layout.addWidget(right_panel, 1)

    # ── Settings helpers ──────────────────────────────────────────────────

    def _restore_settings(self):
        s = self._settings
        self.slider_width.setValue(s.get("width", 200))
        self.slider_height.setValue(s.get("height", 100))

        preset = s.get("aspect_preset", "Source")
        idx = self.cmb_aspect.findText(preset)
        if idx >= 0:
            self.cmb_aspect.setCurrentIndex(idx)

        idx = self.cmb_charset.findText(s.get("char_set_name", "Standard"))
        if idx >= 0:
            self.cmb_charset.setCurrentIndex(idx)
        self.txt_custom_chars.setText(s.get("custom_chars", ""))
        self.chk_invert_ascii.setChecked(s.get("invert_ascii", False))

        cmode = s.get("color_mode", "Colored")
        {"Colored": self.radio_colored, "Grayscale": self.radio_gray}.get(
            cmode, self.radio_mono
        ).setChecked(True)

        self._mono_color = tuple(s.get("mono_color", [255, 255, 255]))
        self._update_mono_button_color()
        self._bg_color = tuple(s.get("bg_color", [14, 14, 14]))
        self._update_bg_button_color()
        self._update_preview_bg()

        self.slider_intensity.setValue(s.get("intensity", 100))
        self.slider_brightness.setValue(s.get("brightness", 100))
        self.chk_loop.setChecked(s.get("loop", True))

        speed_str = f"{s.get('speed', 1.0)}x"
        idx = self.cmb_speed.findText(speed_str)
        if idx >= 0:
            self.cmb_speed.setCurrentIndex(idx)

        self.slider_fontsize.setValue(s.get("font_size", 8))

    def _persist_settings(self):
        self._settings.update({
            "width": self.slider_width.value(),
            "height": self.slider_height.value(),
            "aspect_lock": self.cmb_aspect.currentText() != "Custom",
            "aspect_preset": self.cmb_aspect.currentText(),
            "char_set_name": self.cmb_charset.currentText(),
            "custom_chars": self.txt_custom_chars.text(),
            "color_mode": self._get_color_mode(),
            "mono_color": list(self._mono_color),
            "bg_color": list(self._bg_color),
            "intensity": self.slider_intensity.value(),
            "brightness": self.slider_brightness.value(),
            "invert_ascii": self.chk_invert_ascii.isChecked(),
            "loop": self.chk_loop.isChecked(),
            "speed": self._get_speed(),
            "font_size": self.slider_fontsize.value(),
            "last_video": self._video_path,
            "window_width": self.width(),
            "window_height": self.height(),
        })
        save_settings(self._settings)

    def _get_char_set(self) -> str:
        name = self.cmb_charset.currentText()
        if name == "Custom":
            custom = self.txt_custom_chars.text().strip()
            return custom if custom else CHAR_SETS["Standard"]
        return CHAR_SETS.get(name, CHAR_SETS["Standard"])

    def _get_charset_hint(self) -> str:
        return self.cmb_charset.currentText()

    def _get_color_mode(self) -> str:
        if self.radio_colored.isChecked():
            return "Colored"
        if self.radio_gray.isChecked():
            return "Grayscale"
        return "Monochrome"

    def _get_speed(self) -> float:
        try:
            return float(self.cmb_speed.currentText().replace("x", ""))
        except ValueError:
            return 1.0

    def _update_mono_button_color(self):
        r, g, b = self._mono_color
        self.btn_mono_color.setStyleSheet(
            f"background-color: rgb({r},{g},{b}); "
            f"border: 1px inset #808080; min-width: 20px; min-height: 20px;"
        )

    def _update_bg_button_color(self):
        r, g, b = self._bg_color
        self.btn_bg_color.setStyleSheet(
            f"background-color: rgb({r},{g},{b}); "
            f"border: 1px inset #808080; min-width: 20px; min-height: 20px;"
        )

    def _update_preview_bg(self):
        self.preview.set_bg_color(*self._bg_color)

    def _get_render_settings(self) -> RenderSettings:
        preset = self.cmb_aspect.currentText()
        aspect_lock = preset != "Custom"
        return RenderSettings(
            width=self.slider_width.value(),
            height=self.slider_height.value(),
            char_set=self._get_char_set(),
            color_mode=self._get_color_mode(),
            intensity=self.slider_intensity.value(),
            brightness=self.slider_brightness.value(),
            invert_ascii=self.chk_invert_ascii.isChecked(),
            mono_color=self._mono_color,
            bg_color=self._bg_color,
            speed=self._get_speed(),
            aspect_lock=aspect_lock,
            aspect_preset=preset,
            loop=self.chk_loop.isChecked(),
            font_size=self.slider_fontsize.value(),
        )

    def _push_settings_to_thread(self):
        self._render.apply_settings(self._get_render_settings())

    # ── Event handlers ────────────────────────────────────────────────────

    def _on_setting_changed(self, *_args):
        self._settings_timer.start()

    def _apply_settings(self):
        self._push_settings_to_thread()
        self._persist_settings()
        if self._mode == "image" and self._image_path:
            self._render_image()

    def _on_reset(self):
        """Reset all settings to defaults and load the demo video."""
        d = _RESET_DEFAULTS

        self.slider_width.setValue(d["width"])
        self.slider_height.setValue(d["height"])
        self.cmb_aspect.setCurrentText(d["aspect_preset"])
        self.cmb_charset.setCurrentText(d["char_set_name"])
        self.txt_custom_chars.setText(d["custom_chars"])
        self.chk_invert_ascii.setChecked(d["invert_ascii"])

        {"Colored": self.radio_colored, "Grayscale": self.radio_gray}.get(
            d["color_mode"], self.radio_mono
        ).setChecked(True)

        self.slider_intensity.setValue(d["intensity"])
        self.slider_brightness.setValue(d["brightness"])
        self._mono_color = d["mono_color"]
        self._update_mono_button_color()
        self._bg_color = d["bg_color"]
        self._update_bg_button_color()
        self._update_preview_bg()
        self.cmb_speed.setCurrentText(f"{d['speed']}x")
        self.chk_loop.setChecked(d["loop"])
        self.slider_fontsize.setValue(d["font_size"])

        self._push_settings_to_thread()
        self._persist_settings()

        # Load demo video
        demo = _demo_video_path()
        if os.path.isfile(demo):
            self._load_video(demo)
            QTimer.singleShot(600, self._auto_play_demo)
        else:
            QMessageBox.information(
                self, "Demo Not Found",
                f"Demo video not found:\n{demo}\n\nSettings have been reset.",
            )

    def _on_upload(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Open Video File", "",
            "Video Files (*.mp4 *.avi *.mov *.mkv *.webm);;All Files (*)",
        )
        if path:
            self._load_video(path)

    def _on_upload_image(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Open Image File", "",
            "Image Files (*.png *.jpg *.jpeg *.bmp *.webp *.tiff);;All Files (*)",
        )
        if path:
            self._load_image(path)

    def _load_image(self, path: str):
        self._image_path = path
        self._video_path = ""
        self._mode = "image"
        self.lbl_filename.setText(os.path.basename(path))

        self.btn_play.setEnabled(False)
        self.btn_stop.setEnabled(False)
        self.slider_seek.setEnabled(False)
        self.cmb_speed.setEnabled(False)
        self.chk_loop.setEnabled(False)
        self.btn_export_mp4.setEnabled(False)
        self.btn_save_video.setEnabled(False)
        self.btn_export_png.setEnabled(True)

        self._render.stop()
        self._render_image()
        self._persist_settings()

    def _render_image(self):
        if not self._image_path or not os.path.isfile(self._image_path):
            return

        try:
            img_bgr = cv2.imread(self._image_path, cv2.IMREAD_COLOR)
            if img_bgr is None:
                QMessageBox.warning(self, "Error", f"Cannot load image: {self._image_path}")
                return

            h_img, w_img = img_bgr.shape[:2]
            source_aspect = w_img / h_img if h_img > 0 else 1.77

            preset = self.cmb_aspect.currentText()
            if preset == "Custom":
                aspect = None
            elif preset == "Source":
                aspect = source_aspect
            else:
                aspect = ASPECT_PRESETS.get(preset, source_aspect)

            w = self.slider_width.value()

            chars_2d, colors_rgb = image_to_ascii(
                self._image_path, w,
                self._get_char_set(), self._get_color_mode(),
                self.slider_intensity.value(), self._mono_color,
                aspect_ratio=aspect,
                brightness=self.slider_brightness.value(),
                invert_ascii=self.chk_invert_ascii.isChecked(),
            )
            h = chars_2d.shape[0]

            self._current_chars = chars_2d
            self._current_colors = colors_rgb

            font_px = get_preview_font_px(w)
            atlas = get_atlas(self._get_char_set(), font_px, self._get_charset_hint())
            rgb_array = atlas.compose_frame(chars_2d, colors_rgb, self._bg_color)

            qimg = QImage(
                rgb_array.data, rgb_array.shape[1], rgb_array.shape[0],
                rgb_array.strides[0], QImage.Format.Format_RGB888,
            ).copy()

            self.preview.update_image(qimg)
            self.status_bar.showMessage(
                f"{os.path.basename(self._image_path)} | "
                f"{w}x{h} chars | Image mode"
            )
        except Exception as e:
            traceback.print_exc(file=sys.stderr)
            QMessageBox.warning(self, "Error", f"Image render error: {e}")

    def _load_video(self, path: str):
        self._video_path = path
        self._image_path = ""
        self._mode = "video"
        self.lbl_filename.setText(os.path.basename(path))

        self.btn_play.setEnabled(True)
        self.btn_stop.setEnabled(True)
        self.slider_seek.setEnabled(True)
        self.cmb_speed.setEnabled(True)
        self.chk_loop.setEnabled(True)
        self.btn_export_mp4.setEnabled(True)
        self.btn_save_video.setEnabled(True)
        self.btn_export_png.setEnabled(True)

        self._render.load_video(path)
        QTimer.singleShot(500, self._update_video_info)
        self._persist_settings()

    def _update_video_info(self):
        info = self._render.get_video_info()
        self._total_frames = info["total_frames"]
        self._video_fps = info["fps"]
        self.slider_seek.setRange(0, max(1, self._total_frames - 1))
        self.slider_seek.setValue(0)
        self._update_status(0, self._total_frames, 0.0)

    def _on_aspect_preset_changed(self, text: str):
        is_custom = (text == "Custom")
        self.slider_height.setEnabled(is_custom)

        if not is_custom and text != "Source":
            ratio = ASPECT_PRESETS.get(text)
            if ratio:
                w = self.slider_width.value()
                h = max(10, int(w / ratio * 0.5))
                self.slider_height.blockSignals(True)
                self.slider_height.setValue(h)
                self.lbl_height.setText(str(h))
                self.slider_height.blockSignals(False)

        self._on_setting_changed()

    def _on_width_changed(self, val: int):
        snapped = round(val / 10) * 10
        if snapped != val:
            self.slider_width.blockSignals(True)
            self.slider_width.setValue(snapped)
            self.slider_width.blockSignals(False)
        self.lbl_width.setText(str(snapped))

        preset = self.cmb_aspect.currentText()
        if preset != "Custom" and preset != "Source":
            ratio = ASPECT_PRESETS.get(preset)
            if ratio:
                h = max(10, int(snapped / ratio * 0.5))
                self.slider_height.blockSignals(True)
                self.slider_height.setValue(h)
                self.lbl_height.setText(str(h))
                self.slider_height.blockSignals(False)

        self._on_setting_changed()

    def _on_height_changed(self, val: int):
        snapped = round(val / 5) * 5
        if snapped != val:
            self.slider_height.blockSignals(True)
            self.slider_height.setValue(snapped)
            self.slider_height.blockSignals(False)
        self.lbl_height.setText(str(snapped))
        self._on_setting_changed()

    def _on_charset_changed(self, text: str):
        self.txt_custom_chars.setEnabled(text == "Custom")
        self._on_setting_changed()

    def _on_intensity_changed(self, val: int):
        self.lbl_intensity.setText(f"{val}%")
        self._on_setting_changed()

    def _on_brightness_changed(self, val: int):
        self.lbl_brightness.setText(f"{val}%")
        self._on_setting_changed()

    def _on_pick_mono_color(self):
        color = QColorDialog.getColor(QColor(*self._mono_color), self, "Pick Monochrome Color")
        if color.isValid():
            self._mono_color = (color.red(), color.green(), color.blue())
            self._update_mono_button_color()
            self._on_setting_changed()

    def _on_pick_bg_color(self):
        color = QColorDialog.getColor(QColor(*self._bg_color), self, "Pick Background Color")
        if color.isValid():
            self._bg_color = (color.red(), color.green(), color.blue())
            self._update_bg_button_color()
            self._update_preview_bg()
            self._on_setting_changed()

    def _on_play_toggle(self, checked: bool):
        if not self._video_path:
            self.btn_play.setChecked(False)
            QMessageBox.information(self, "No Video", "Please open a video first.")
            return
        if checked:
            self.btn_play.setText("Pause")
            self._render.play()
        else:
            self.btn_play.setText("Play")
            self._render.pause()

    def _on_stop(self):
        self.btn_play.setChecked(False)
        self.btn_play.setText("Play")
        self._render.stop()
        self.slider_seek.setValue(0)

    def _on_seek_pressed(self):
        self._seeking = True

    def _on_seek_released(self):
        self._seeking = False
        self._render.seek(self.slider_seek.value())

    def _on_seek_changed(self, val: int):
        if self._seeking:
            self._render.seek(val)

    def _on_frame_rendered(
        self,
        qimage: QImage,
        chars_2d: np.ndarray,
        colors_rgb: np.ndarray,
        frame_no: int,
        total: int,
        render_ms: float,
    ):
        self._render.mark_frame_consumed()

        self._current_chars = chars_2d
        self._current_colors = colors_rgb
        self._current_frame_no = frame_no
        self._total_frames = total

        self.preview.update_image(qimage)

        if not self._seeking:
            self.slider_seek.blockSignals(True)
            self.slider_seek.setRange(0, max(1, total - 1))
            self.slider_seek.setValue(frame_no)
            self.slider_seek.blockSignals(False)

        self._update_status(frame_no, total, render_ms)

    def _update_status(self, frame_no: int, total: int, render_ms: float):
        name = os.path.basename(self._video_path) if self._video_path else "-"
        w = self.slider_width.value()
        h = self.slider_height.value()
        loop_str = "Loop" if self.chk_loop.isChecked() else ""
        self.status_bar.showMessage(
            f"{name} | Frame {frame_no}/{total} | "
            f"FPS: {self._video_fps:.1f} | {w}x{h} chars | "
            f"Render: {render_ms:.1f}ms"
            + (f" | {loop_str}" if loop_str else "")
        )

    def _on_playback_finished(self):
        self.btn_play.setChecked(False)
        self.btn_play.setText("Play")

    def _on_error(self, msg: str):
        QMessageBox.warning(self, "Error", msg)

    # ── Export handlers ───────────────────────────────────────────────────

    def _on_export_mp4(self):
        if not self._video_path:
            QMessageBox.information(self, "No Video", "Please open a video first.")
            return

        path, _ = QFileDialog.getSaveFileName(
            self, "Export ASCII MP4", "", "MP4 Video (*.mp4)"
        )
        if not path:
            return

        progress = QProgressDialog("Rendering ASCII video...", "Cancel", 0, 100, self)
        progress.setWindowTitle("Exporting MP4")
        progress.setWindowModality(Qt.WindowModality.WindowModal)
        progress.setMinimumDuration(0)
        progress.setMinimumWidth(350)
        progress.setValue(0)

        preset = self.cmb_aspect.currentText()
        self._export_thread = ExportMP4Thread(
            video_path=self._video_path,
            output_path=path,
            width=self.slider_width.value(),
            height=self.slider_height.value(),
            char_set=self._get_char_set(),
            color_mode=self._get_color_mode(),
            intensity=self.slider_intensity.value(),
            mono_color=self._mono_color,
            font_size=self.slider_fontsize.value(),
            aspect_lock=preset != "Custom",
            bg_color=self._bg_color,
            brightness=self.slider_brightness.value(),
            charset_hint=self._get_charset_hint(),
            parent=self,
            invert_ascii=self.chk_invert_ascii.isChecked(),
        )

        self._export_thread.progress.connect(progress.setValue)
        self._export_thread.finished_ok.connect(
            lambda p: QMessageBox.information(self, "Done", f"ASCII MP4 saved to:\n{p}")
        )
        self._export_thread.error_occurred.connect(
            lambda e: QMessageBox.warning(self, "Export Error", e)
        )
        progress.canceled.connect(self._export_thread.cancel)
        self._export_thread.finished.connect(progress.close)
        self._export_thread.start()

    def _on_export_png(self):
        if self._current_chars is None:
            QMessageBox.information(self, "No Frame", "No frame to export.")
            return

        path, _ = QFileDialog.getSaveFileName(
            self, "Export ASCII PNG", "", "PNG Image (*.png)"
        )
        if not path:
            return

        try:
            font_size = self.slider_fontsize.value()
            atlas = get_atlas(self._get_char_set(), font_size, self._get_charset_hint())
            rgb_frame = atlas.compose_frame(
                self._current_chars, self._current_colors, self._bg_color,
            )
            bgr_frame = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR)
            cv2.imwrite(path, bgr_frame)
            QMessageBox.information(self, "Saved", f"ASCII PNG saved to:\n{path}")
        except Exception as e:
            traceback.print_exc(file=sys.stderr)
            QMessageBox.warning(self, "Export Error", str(e))

    def _on_save_video(self):
        if not self._video_path:
            QMessageBox.information(self, "No Video", "Please open a video first.")
            return

        path, _ = QFileDialog.getSaveFileName(
            self, "Save ASCII Text Video", "", "Text Files (*.txt)"
        )
        if not path:
            return

        progress = QProgressDialog("Exporting ASCII text...", "Cancel", 0, 100, self)
        progress.setWindowTitle("Exporting")
        progress.setWindowModality(Qt.WindowModality.WindowModal)
        progress.setMinimumDuration(0)
        progress.setValue(0)

        preset = self.cmb_aspect.currentText()
        self._export_thread = ExportVideoThread(
            video_path=self._video_path,
            output_path=path,
            width=self.slider_width.value(),
            height=self.slider_height.value(),
            char_set=self._get_char_set(),
            color_mode=self._get_color_mode(),
            intensity=self.slider_intensity.value(),
            mono_color=self._mono_color,
            aspect_lock=preset != "Custom",
            brightness=self.slider_brightness.value(),
            parent=self,
            invert_ascii=self.chk_invert_ascii.isChecked(),
        )

        self._export_thread.progress.connect(progress.setValue)
        self._export_thread.finished_ok.connect(
            lambda p: QMessageBox.information(self, "Done", f"Saved to:\n{p}")
        )
        self._export_thread.error_occurred.connect(
            lambda e: QMessageBox.warning(self, "Export Error", e)
        )
        progress.canceled.connect(self._export_thread.cancel)
        self._export_thread.finished.connect(progress.close)
        self._export_thread.start()

    def _on_save_frame(self):
        if self._current_chars is None:
            QMessageBox.information(self, "No Frame", "No frame to save.")
            return
        path, filt = QFileDialog.getSaveFileName(
            self, "Save Current Frame", "",
            "Text File (*.txt);;HTML File (*.html)",
        )
        if not path:
            return
        try:
            if path.lower().endswith(".html") or "HTML" in filt:
                save_current_frame_html(
                    self._current_chars, self._current_colors,
                    path, self.slider_fontsize.value(), self._bg_color,
                )
            else:
                save_current_frame_txt(self._current_chars, path)
            QMessageBox.information(self, "Saved", f"Frame saved to:\n{path}")
        except Exception as e:
            traceback.print_exc(file=sys.stderr)
            QMessageBox.warning(self, "Save Error", str(e))

    def _on_export_html(self):
        if not self._video_path and not self._image_path:
            QMessageBox.information(self, "No Content", "Please load a video or image first.")
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Export HTML", "", "HTML Files (*.html)"
        )
        if not path:
            return
        try:
            preset = self.cmb_aspect.currentText()
            if self._mode == "image" or not self._video_path:
                if self._current_chars is None:
                    QMessageBox.information(self, "No Frame", "No frame to export.")
                    return
                save_current_frame_html(
                    self._current_chars, self._current_colors,
                    path, self.slider_fontsize.value(), self._bg_color,
                )
            else:
                export_full_html(
                    video_path=self._video_path,
                    output_path=path,
                    frame_no=self._current_frame_no,
                    width=self.slider_width.value(),
                    height=self.slider_height.value(),
                    char_set=self._get_char_set(),
                    color_mode=self._get_color_mode(),
                    intensity=self.slider_intensity.value(),
                    mono_color=self._mono_color,
                    font_size=self.slider_fontsize.value(),
                    aspect_lock=preset != "Custom",
                    bg_color=self._bg_color,
                    brightness=self.slider_brightness.value(),
                    invert_ascii=self.chk_invert_ascii.isChecked(),
                )
            QMessageBox.information(self, "Exported", f"HTML exported to:\n{path}")
        except Exception as e:
            traceback.print_exc(file=sys.stderr)
            QMessageBox.warning(self, "Export Error", str(e))

    # ── Cleanup ───────────────────────────────────────────────────────────

    def closeEvent(self, event):
        try:
            self._persist_settings()
        except Exception as e:
            print(f"Warning: Failed to persist settings: {e}", file=sys.stderr)
            traceback.print_exc(file=sys.stderr)
        self._render.shutdown()
        if self._export_thread and self._export_thread.isRunning():
            self._export_thread.cancel()
            self._export_thread.wait(3000)
        event.accept()


# ── Entry point ───────────────────────────────────────────────────────────

def main():
    app = QApplication(sys.argv)
    app.setStyle("Fusion")

    from PyQt6.QtGui import QPalette
    palette = QPalette()
    palette.setColor(QPalette.ColorRole.Window, QColor(240, 240, 240))
    palette.setColor(QPalette.ColorRole.WindowText, QColor(0, 0, 0))
    palette.setColor(QPalette.ColorRole.Base, QColor(255, 255, 255))
    palette.setColor(QPalette.ColorRole.AlternateBase, QColor(233, 233, 233))
    palette.setColor(QPalette.ColorRole.Text, QColor(0, 0, 0))
    palette.setColor(QPalette.ColorRole.Button, QColor(224, 224, 224))
    palette.setColor(QPalette.ColorRole.ButtonText, QColor(0, 0, 0))
    palette.setColor(QPalette.ColorRole.Highlight, QColor(0, 0, 128))
    palette.setColor(QPalette.ColorRole.HighlightedText, QColor(255, 255, 255))
    app.setPalette(palette)

    app.setFont(QFont("Tahoma", 9))

    icon_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "icon.ico")
    if os.path.isfile(icon_path):
        app.setWindowIcon(QIcon(icon_path))

    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
