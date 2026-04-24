"""
VideoToASCII — Main application entry point.
Full desktop GUI: left control panel + right QImage-based ASCII preview.
Optimized: glyph atlas rendering, buffer reuse, MP4 export, image-to-ASCII.
"""

import sys
import os
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
from ascii_renderer import CHAR_SETS, image_to_ascii, render_to_rgb
from glyph_atlas import get_atlas
import numpy as np
import cv2


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("VideoToASCII — Video & Image to Colored ASCII Art")
        self._settings = load_settings()
        self.resize(
            self._settings.get("window_width", 1400),
            self._settings.get("window_height", 900),
        )
        self.setMinimumSize(900, 600)

        self._video_path = self._settings.get("last_video", "")
        self._image_path = ""
        self._mode = "video"  # "video" or "image"
        self._current_chars: np.ndarray | None = None
        self._current_colors: np.ndarray | None = None
        self._current_frame_no = 0
        self._total_frames = 0
        self._video_fps = 24.0
        self._mono_color = tuple(self._settings.get("mono_color", [255, 255, 255]))
        self._export_thread: ExportVideoThread | ExportMP4Thread | None = None

        # Settings debounce timer — prevents spamming render thread
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

        if self._video_path and os.path.isfile(self._video_path):
            self._load_video(self._video_path)

    # ── UI construction ───────────────────────────────────────────────

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QHBoxLayout(central)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        # ── Left panel (scrollable) ──────────────────────────────────
        left_scroll = QScrollArea()
        left_scroll.setFixedWidth(290)
        left_scroll.setWidgetResizable(True)
        left_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        left_scroll.setStyleSheet("QScrollArea { border: none; background: #1a1a2e; }")

        left_panel = QWidget()
        left_panel.setStyleSheet(
            "QWidget { background: #1a1a2e; color: #e0e0e0; }"
            "QGroupBox { border: 1px solid #333; border-radius: 6px; margin-top: 10px; padding-top: 14px; font-weight: bold; color: #b0b0d0; }"
            "QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 4px; }"
            "QPushButton { background: #16213e; border: 1px solid #444; border-radius: 5px; padding: 7px 12px; color: #e0e0e0; font-weight: bold; }"
            "QPushButton:hover { background: #1a3a5c; border-color: #66b2ff; }"
            "QPushButton:pressed { background: #0f3460; }"
            "QSlider::groove:horizontal { height: 6px; background: #333; border-radius: 3px; }"
            "QSlider::handle:horizontal { width: 14px; margin: -4px 0; background: #4fc3f7; border-radius: 7px; }"
            "QSlider::sub-page:horizontal { background: #4fc3f7; border-radius: 3px; }"
            "QComboBox { background: #16213e; border: 1px solid #444; border-radius: 4px; padding: 4px 8px; color: #e0e0e0; }"
            "QComboBox QAbstractItemView { background: #1a1a2e; color: #e0e0e0; selection-background-color: #1a3a5c; }"
            "QRadioButton { spacing: 6px; color: #e0e0e0; }"
            "QCheckBox { spacing: 6px; color: #e0e0e0; }"
            "QLineEdit { background: #16213e; border: 1px solid #444; border-radius: 4px; padding: 4px 6px; color: #e0e0e0; }"
            "QLabel { color: #c0c0d8; }"
        )
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(10, 10, 10, 10)
        left_layout.setSpacing(4)

        # Upload buttons
        upload_row = QHBoxLayout()
        self.btn_upload = QPushButton("📁  Video")
        self.btn_upload.setStyleSheet(
            "QPushButton { background: #0f3460; font-size: 13px; padding: 10px; }"
            "QPushButton:hover { background: #1a5276; border-color: #5dade2; }"
        )
        self.btn_upload.clicked.connect(self._on_upload)
        upload_row.addWidget(self.btn_upload)

        self.btn_upload_image = QPushButton("📷  Image")
        self.btn_upload_image.setStyleSheet(
            "QPushButton { background: #1b4332; font-size: 13px; padding: 10px; }"
            "QPushButton:hover { background: #2d6a4f; border-color: #52b788; }"
        )
        self.btn_upload_image.clicked.connect(self._on_upload_image)
        upload_row.addWidget(self.btn_upload_image)
        left_layout.addLayout(upload_row)

        self.lbl_filename = QLabel("No file loaded")
        self.lbl_filename.setWordWrap(True)
        self.lbl_filename.setStyleSheet("color: #888; font-size: 11px; padding: 2px;")
        left_layout.addWidget(self.lbl_filename)

        # ── Resolution ────────────────────────────────────────────────
        grp_res = QGroupBox("RESOLUTION")
        res_layout = QVBoxLayout(grp_res)

        res_layout.addWidget(QLabel("Width (chars):"))
        self.slider_width = QSlider(Qt.Orientation.Horizontal)
        self.slider_width.setRange(40, 1000)
        self.slider_width.setSingleStep(10)
        self.slider_width.setPageStep(50)
        self.slider_width.setValue(200)
        self.lbl_width = QLabel("200")
        w_row = QHBoxLayout()
        w_row.addWidget(self.slider_width)
        w_row.addWidget(self.lbl_width)
        res_layout.addLayout(w_row)

        res_layout.addWidget(QLabel("Height (chars):"))
        self.slider_height = QSlider(Qt.Orientation.Horizontal)
        self.slider_height.setRange(10, 500)
        self.slider_height.setSingleStep(5)
        self.slider_height.setPageStep(25)
        self.slider_height.setValue(100)
        self.lbl_height = QLabel("100")
        h_row = QHBoxLayout()
        h_row.addWidget(self.slider_height)
        h_row.addWidget(self.lbl_height)
        res_layout.addLayout(h_row)

        self.chk_aspect = QCheckBox("Lock aspect ratio")
        self.chk_aspect.setChecked(True)
        res_layout.addWidget(self.chk_aspect)

        self.slider_width.valueChanged.connect(self._on_width_changed)
        self.slider_height.valueChanged.connect(self._on_height_changed)
        self.chk_aspect.toggled.connect(self._on_setting_changed)

        left_layout.addWidget(grp_res)

        # ── Character set ─────────────────────────────────────────────
        grp_chars = QGroupBox("CHARACTER SET")
        chars_layout = QVBoxLayout(grp_chars)

        self.cmb_charset = QComboBox()
        self.cmb_charset.addItems(["Standard", "Dense", "Simple", "Custom"])
        chars_layout.addWidget(self.cmb_charset)

        self.txt_custom_chars = QLineEdit()
        self.txt_custom_chars.setPlaceholderText("Enter custom characters…")
        self.txt_custom_chars.setEnabled(False)
        chars_layout.addWidget(self.txt_custom_chars)

        self.cmb_charset.currentTextChanged.connect(self._on_charset_changed)
        self.txt_custom_chars.textChanged.connect(self._on_setting_changed)

        left_layout.addWidget(grp_chars)

        # ── Color mode ────────────────────────────────────────────────
        grp_color = QGroupBox("COLOR MODE")
        color_layout = QVBoxLayout(grp_color)

        self.radio_colored = QRadioButton("Colored")
        self.radio_gray = QRadioButton("Grayscale")
        self.radio_mono = QRadioButton("Monochrome")
        self.radio_colored.setChecked(True)

        self.color_group = QButtonGroup(self)
        self.color_group.addButton(self.radio_colored)
        self.color_group.addButton(self.radio_gray)
        self.color_group.addButton(self.radio_mono)

        color_layout.addWidget(self.radio_colored)
        color_layout.addWidget(self.radio_gray)

        mono_row = QHBoxLayout()
        mono_row.addWidget(self.radio_mono)
        self.btn_mono_color = QPushButton()
        self.btn_mono_color.setFixedSize(28, 28)
        self._update_mono_button_color()
        self.btn_mono_color.clicked.connect(self._on_pick_mono_color)
        mono_row.addWidget(self.btn_mono_color)
        mono_row.addStretch()
        color_layout.addLayout(mono_row)

        self.color_group.buttonToggled.connect(self._on_setting_changed)

        color_layout.addWidget(QLabel("Color intensity:"))
        self.slider_intensity = QSlider(Qt.Orientation.Horizontal)
        self.slider_intensity.setRange(0, 100)
        self.slider_intensity.setValue(80)
        self.lbl_intensity = QLabel("80%")
        int_row = QHBoxLayout()
        int_row.addWidget(self.slider_intensity)
        int_row.addWidget(self.lbl_intensity)
        color_layout.addLayout(int_row)
        self.slider_intensity.valueChanged.connect(self._on_intensity_changed)

        left_layout.addWidget(grp_color)

        # ── Playback ──────────────────────────────────────────────────
        grp_play = QGroupBox("PLAYBACK")
        play_layout = QVBoxLayout(grp_play)

        btn_row = QHBoxLayout()
        self.btn_play = QPushButton("▶  Play")
        self.btn_play.setCheckable(True)
        self.btn_play.clicked.connect(self._on_play_toggle)
        btn_row.addWidget(self.btn_play)

        self.btn_stop = QPushButton("■  Stop")
        self.btn_stop.clicked.connect(self._on_stop)
        btn_row.addWidget(self.btn_stop)
        play_layout.addLayout(btn_row)

        speed_row = QHBoxLayout()
        speed_row.addWidget(QLabel("Speed:"))
        self.cmb_speed = QComboBox()
        self.cmb_speed.addItems(["0.25x", "0.5x", "1x", "2x"])
        self.cmb_speed.setCurrentText("1x")
        self.cmb_speed.currentTextChanged.connect(self._on_setting_changed)
        speed_row.addWidget(self.cmb_speed)
        play_layout.addLayout(speed_row)

        play_layout.addWidget(QLabel("Frame:"))
        self.slider_seek = QSlider(Qt.Orientation.Horizontal)
        self.slider_seek.setRange(0, 1)
        self.slider_seek.setValue(0)
        self.slider_seek.sliderPressed.connect(self._on_seek_pressed)
        self.slider_seek.sliderReleased.connect(self._on_seek_released)
        self.slider_seek.valueChanged.connect(self._on_seek_changed)
        self._seeking = False
        play_layout.addWidget(self.slider_seek)

        left_layout.addWidget(grp_play)

        # ── Output ────────────────────────────────────────────────────
        grp_out = QGroupBox("OUTPUT")
        out_layout = QVBoxLayout(grp_out)

        fs_row = QHBoxLayout()
        fs_row.addWidget(QLabel("Export font size:"))
        self.slider_fontsize = QSlider(Qt.Orientation.Horizontal)
        self.slider_fontsize.setRange(4, 16)
        self.slider_fontsize.setValue(8)
        self.lbl_fontsize = QLabel("8px")
        self.slider_fontsize.valueChanged.connect(
            lambda v: self.lbl_fontsize.setText(f"{v}px")
        )
        fs_row.addWidget(self.slider_fontsize)
        fs_row.addWidget(self.lbl_fontsize)
        out_layout.addLayout(fs_row)

        self.btn_export_mp4 = QPushButton("🎬  Export as ASCII MP4")
        self.btn_export_mp4.setStyleSheet(
            "QPushButton { background: #1b4332; font-size: 12px; }"
            "QPushButton:hover { background: #2d6a4f; border-color: #52b788; }"
        )
        self.btn_export_mp4.clicked.connect(self._on_export_mp4)
        out_layout.addWidget(self.btn_export_mp4)

        self.btn_export_png = QPushButton("🖼️  Export as ASCII PNG")
        self.btn_export_png.setStyleSheet(
            "QPushButton { background: #1b4332; font-size: 12px; }"
            "QPushButton:hover { background: #2d6a4f; border-color: #52b788; }"
        )
        self.btn_export_png.clicked.connect(self._on_export_png)
        out_layout.addWidget(self.btn_export_png)

        self.btn_save_video = QPushButton("💾  Save ASCII Text (.txt)")
        self.btn_save_video.clicked.connect(self._on_save_video)
        out_layout.addWidget(self.btn_save_video)

        self.btn_save_frame = QPushButton("📄  Save Current Frame")
        self.btn_save_frame.clicked.connect(self._on_save_frame)
        out_layout.addWidget(self.btn_save_frame)

        self.btn_export_html = QPushButton("🌐  Export Frame as HTML")
        self.btn_export_html.clicked.connect(self._on_export_html)
        out_layout.addWidget(self.btn_export_html)

        left_layout.addWidget(grp_out)
        left_layout.addStretch()

        left_scroll.setWidget(left_panel)

        # ── Right panel (preview) ─────────────────────────────────────
        right_panel = QWidget()
        right_panel.setStyleSheet("background: #0e0e0e;")
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(0)

        self.preview = PreviewWidget()
        self.preview.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        right_layout.addWidget(self.preview)

        self.status_bar = QStatusBar()
        self.status_bar.setStyleSheet(
            "QStatusBar { background: #111; color: #999; font-size: 11px; "
            "padding: 2px 8px; border-top: 1px solid #333; }"
        )
        self.status_bar.showMessage("Ready — load a video to begin")
        right_layout.addWidget(self.status_bar)

        main_layout.addWidget(left_scroll)

        divider = QFrame()
        divider.setFrameShape(QFrame.Shape.VLine)
        divider.setStyleSheet("QFrame { color: #333; }")
        main_layout.addWidget(divider)

        main_layout.addWidget(right_panel, 1)

    # ── Settings restore/save ─────────────────────────────────────────

    def _restore_settings(self):
        s = self._settings
        self.slider_width.setValue(s.get("width", 200))
        self.slider_height.setValue(s.get("height", 100))
        self.chk_aspect.setChecked(s.get("aspect_lock", True))

        idx = self.cmb_charset.findText(s.get("char_set_name", "Standard"))
        if idx >= 0:
            self.cmb_charset.setCurrentIndex(idx)
        self.txt_custom_chars.setText(s.get("custom_chars", ""))

        cmode = s.get("color_mode", "Colored")
        {"Colored": self.radio_colored, "Grayscale": self.radio_gray}.get(
            cmode, self.radio_mono
        ).setChecked(True)

        self._mono_color = tuple(s.get("mono_color", [255, 255, 255]))
        self._update_mono_button_color()
        self.slider_intensity.setValue(s.get("intensity", 80))

        speed_str = f"{s.get('speed', 1.0)}x"
        idx = self.cmb_speed.findText(speed_str)
        if idx >= 0:
            self.cmb_speed.setCurrentIndex(idx)

        self.slider_fontsize.setValue(s.get("font_size", 8))

    def _persist_settings(self):
        self._settings.update({
            "width": self.slider_width.value(),
            "height": self.slider_height.value(),
            "aspect_lock": self.chk_aspect.isChecked(),
            "char_set_name": self.cmb_charset.currentText(),
            "custom_chars": self.txt_custom_chars.text(),
            "color_mode": self._get_color_mode(),
            "mono_color": list(self._mono_color),
            "intensity": self.slider_intensity.value(),
            "speed": self._get_speed(),
            "font_size": self.slider_fontsize.value(),
            "last_video": self._video_path,
            "window_width": self.width(),
            "window_height": self.height(),
        })
        save_settings(self._settings)

    # ── Helpers ────────────────────────────────────────────────────────

    def _get_char_set(self) -> str:
        name = self.cmb_charset.currentText()
        if name == "Custom":
            custom = self.txt_custom_chars.text().strip()
            return custom if custom else CHAR_SETS["Standard"]
        return CHAR_SETS.get(name, CHAR_SETS["Standard"])

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
            f"background-color: rgb({r},{g},{b}); border: 1px solid #666; border-radius: 4px;"
        )

    def _push_settings_to_thread(self):
        self._render.update_settings(
            width=self.slider_width.value(),
            height=self.slider_height.value(),
            char_set=self._get_char_set(),
            color_mode=self._get_color_mode(),
            intensity=self.slider_intensity.value(),
            mono_color=self._mono_color,
            speed=self._get_speed(),
            aspect_lock=self.chk_aspect.isChecked(),
        )

    # ── Debounced settings ────────────────────────────────────────────

    def _on_setting_changed(self, *_args):
        self._settings_timer.start()  # Restart debounce

    def _apply_settings(self):
        """Called after debounce timer expires."""
        self._push_settings_to_thread()
        self._persist_settings()
        # Re-render image if in image mode
        if self._mode == "image" and self._image_path:
            self._render_image()

    # ── Slots ──────────────────────────────────────────────────────────

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
        """Load an image and render it as ASCII art."""
        self._image_path = path
        self._video_path = ""
        self._mode = "image"
        self.lbl_filename.setText(f"🖼️ {os.path.basename(path)}")

        # Hide video-only controls
        self.btn_play.setEnabled(False)
        self.btn_stop.setEnabled(False)
        self.slider_seek.setEnabled(False)
        self.cmb_speed.setEnabled(False)
        self.btn_export_mp4.setEnabled(False)
        self.btn_save_video.setEnabled(False)
        self.btn_export_png.setEnabled(True)

        # Stop any video playback
        self._render.stop()

        # Render the image
        self._render_image()
        self._persist_settings()

    def _render_image(self):
        """Render the loaded image as ASCII art and display in preview."""
        if not self._image_path or not os.path.isfile(self._image_path):
            return

        try:
            img_bgr = cv2.imread(self._image_path, cv2.IMREAD_COLOR)
            if img_bgr is None:
                QMessageBox.warning(self, "Error", f"Cannot load image: {self._image_path}")
                return

            h_img, w_img = img_bgr.shape[:2]
            aspect = w_img / h_img if h_img > 0 else 1.77

            w = self.slider_width.value()
            h = max(1, int(w / aspect * 0.5))

            chars_2d, colors_rgb = image_to_ascii(
                self._image_path, w,
                self._get_char_set(), self._get_color_mode(),
                self.slider_intensity.value(), self._mono_color,
                aspect_ratio=aspect,
            )

            self._current_chars = chars_2d
            self._current_colors = colors_rgb

            # Compose preview image via glyph atlas
            font_px = 10 if w <= 150 else (7 if w <= 300 else (5 if w <= 500 else 4))
            atlas = get_atlas(self._get_char_set(), font_px)
            rgb_array = atlas.compose_frame(chars_2d, colors_rgb, (14, 14, 14))

            qimg = QImage(
                rgb_array.data, rgb_array.shape[1], rgb_array.shape[0],
                rgb_array.strides[0], QImage.Format.Format_RGB888,
            ).copy()

            self.preview.update_image(qimg)
            self.status_bar.showMessage(
                f"  🖼️ {os.path.basename(self._image_path)}  |  "
                f"{w}×{h} chars  |  Image mode"
            )
        except Exception as e:
            QMessageBox.warning(self, "Error", f"Image render error: {e}")

    def _load_video(self, path: str):
        self._video_path = path
        self._image_path = ""
        self._mode = "video"
        self.lbl_filename.setText(os.path.basename(path))

        # Re-enable video controls
        self.btn_play.setEnabled(True)
        self.btn_stop.setEnabled(True)
        self.slider_seek.setEnabled(True)
        self.cmb_speed.setEnabled(True)
        self.btn_export_mp4.setEnabled(True)
        self.btn_save_video.setEnabled(True)
        self.btn_export_png.setEnabled(True)

        # Force aspect lock on video load
        self.chk_aspect.setChecked(True)

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

    def _on_width_changed(self, val: int):
        snapped = round(val / 10) * 10
        if snapped != val:
            self.slider_width.blockSignals(True)
            self.slider_width.setValue(snapped)
            self.slider_width.blockSignals(False)
        self.lbl_width.setText(str(snapped))
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

    def _on_pick_mono_color(self):
        color = QColorDialog.getColor(QColor(*self._mono_color), self, "Pick Monochrome Color")
        if color.isValid():
            self._mono_color = (color.red(), color.green(), color.blue())
            self._update_mono_button_color()
            self._on_setting_changed()

    def _on_play_toggle(self, checked: bool):
        if not self._video_path:
            self.btn_play.setChecked(False)
            QMessageBox.information(self, "No Video", "Please upload a video first.")
            return
        if checked:
            self.btn_play.setText("❚❚  Pause")
            self._render.play()
        else:
            self.btn_play.setText("▶  Play")
            self._render.pause()

    def _on_stop(self):
        self.btn_play.setChecked(False)
        self.btn_play.setText("▶  Play")
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
        name = os.path.basename(self._video_path) if self._video_path else "—"
        w = self.slider_width.value()
        h = self.slider_height.value()
        self.status_bar.showMessage(
            f"  {name}  |  Frame {frame_no}/{total}  |  "
            f"FPS: {self._video_fps:.1f}  |  {w}×{h} chars  |  "
            f"Render: {render_ms:.1f} ms"
        )

    def _on_playback_finished(self):
        self.btn_play.setChecked(False)
        self.btn_play.setText("▶  Play")

    def _on_error(self, msg: str):
        QMessageBox.warning(self, "Error", msg)

    # ── Export: MP4 ────────────────────────────────────────────────────

    def _on_export_mp4(self):
        if not self._video_path:
            QMessageBox.information(self, "No Video", "Please upload a video first.")
            return

        path, _ = QFileDialog.getSaveFileName(
            self, "Export ASCII MP4", "", "MP4 Video (*.mp4)"
        )
        if not path:
            return

        progress = QProgressDialog("Converting to ASCII video…", "Cancel", 0, 100, self)
        progress.setWindowTitle("Exporting MP4")
        progress.setWindowModality(Qt.WindowModality.WindowModal)
        progress.setMinimumDuration(0)
        progress.setMinimumWidth(400)
        progress.setValue(0)

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
            aspect_lock=self.chk_aspect.isChecked(),
            parent=self,
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

    # ── Export: PNG ────────────────────────────────────────────────────

    def _on_export_png(self):
        if self._current_chars is None:
            QMessageBox.information(self, "No Frame", "No frame to export. Load a video or image first.")
            return

        path, _ = QFileDialog.getSaveFileName(
            self, "Export ASCII PNG", "", "PNG Image (*.png)"
        )
        if not path:
            return

        try:
            font_size = self.slider_fontsize.value()
            atlas = get_atlas(self._get_char_set(), font_size)
            rgb_frame = atlas.compose_frame(
                self._current_chars, self._current_colors, (17, 17, 17)
            )
            # Convert RGB to BGR for cv2.imwrite
            bgr_frame = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR)
            cv2.imwrite(path, bgr_frame)
            QMessageBox.information(self, "Saved", f"ASCII PNG saved to:\n{path}")
        except Exception as e:
            QMessageBox.warning(self, "Export Error", str(e))

    # ── Export: TXT ────────────────────────────────────────────────────

    def _on_save_video(self):
        if not self._video_path:
            QMessageBox.information(self, "No Video", "Please upload a video first.")
            return

        path, _ = QFileDialog.getSaveFileName(
            self, "Save ASCII Text Video", "", "Text Files (*.txt)"
        )
        if not path:
            return

        progress = QProgressDialog("Exporting ASCII text…", "Cancel", 0, 100, self)
        progress.setWindowTitle("Exporting")
        progress.setWindowModality(Qt.WindowModality.WindowModal)
        progress.setMinimumDuration(0)
        progress.setValue(0)

        self._export_thread = ExportVideoThread(
            video_path=self._video_path,
            output_path=path,
            width=self.slider_width.value(),
            height=self.slider_height.value(),
            char_set=self._get_char_set(),
            color_mode=self._get_color_mode(),
            intensity=self.slider_intensity.value(),
            mono_color=self._mono_color,
            aspect_lock=self.chk_aspect.isChecked(),
            parent=self,
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

    # ── Export: single frame ──────────────────────────────────────────

    def _on_save_frame(self):
        if self._current_chars is None:
            QMessageBox.information(self, "No Frame", "No frame to save. Play a video first.")
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
                    self._current_chars, self._current_colors, path, self.slider_fontsize.value()
                )
            else:
                save_current_frame_txt(self._current_chars, path)
            QMessageBox.information(self, "Saved", f"Frame saved to:\n{path}")
        except Exception as e:
            QMessageBox.warning(self, "Save Error", str(e))

    def _on_export_html(self):
        if not self._video_path:
            QMessageBox.information(self, "No Video", "Please upload a video first.")
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Export HTML", "", "HTML Files (*.html)"
        )
        if not path:
            return
        try:
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
                aspect_lock=self.chk_aspect.isChecked(),
            )
            QMessageBox.information(self, "Exported", f"HTML exported to:\n{path}")
        except Exception as e:
            QMessageBox.warning(self, "Export Error", str(e))

    # ── Cleanup ────────────────────────────────────────────────────────

    def closeEvent(self, event):
        self._persist_settings()
        self._render.shutdown()
        if self._export_thread and self._export_thread.isRunning():
            self._export_thread.cancel()
            self._export_thread.wait(3000)
        event.accept()


def main():
    app = QApplication(sys.argv)
    app.setStyle("Fusion")

    from PyQt6.QtGui import QPalette
    palette = QPalette()
    palette.setColor(QPalette.ColorRole.Window, QColor(26, 26, 46))
    palette.setColor(QPalette.ColorRole.WindowText, QColor(224, 224, 224))
    palette.setColor(QPalette.ColorRole.Base, QColor(14, 14, 14))
    palette.setColor(QPalette.ColorRole.AlternateBase, QColor(30, 30, 50))
    palette.setColor(QPalette.ColorRole.Text, QColor(224, 224, 224))
    palette.setColor(QPalette.ColorRole.Button, QColor(22, 33, 62))
    palette.setColor(QPalette.ColorRole.ButtonText, QColor(224, 224, 224))
    palette.setColor(QPalette.ColorRole.Highlight, QColor(79, 195, 247))
    palette.setColor(QPalette.ColorRole.HighlightedText, QColor(0, 0, 0))
    app.setPalette(palette)

    app.setFont(QFont("Segoe UI", 10))

    icon_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "icon.ico")
    if os.path.isfile(icon_path):
        app.setWindowIcon(QIcon(icon_path))

    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
