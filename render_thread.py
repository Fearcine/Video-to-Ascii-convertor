"""Background thread for real-time ASCII video playback.

Uses deadline-based frame timing with automatic frame skipping to maintain
smooth playback at the correct speed, regardless of how long each render takes.
"""

import time
import sys
import threading
import traceback
import numpy as np
import cv2
from PyQt6.QtCore import QThread, pyqtSignal
from PyQt6.QtGui import QImage
from ascii_renderer import frame_to_ascii
from glyph_atlas import get_atlas
from shared_utils import get_preview_font_px
from render_settings import RenderSettings


class RenderThread(QThread):

    frame_rendered = pyqtSignal(object, object, object, int, int, float)
    playback_finished = pyqtSignal()
    error_occurred = pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._lock = threading.Lock()
        self._video_path: str = ""
        self._playing = False
        self._stop_flag = False
        self._shutdown = False
        self._seek_frame: int | None = None
        self._total_frames = 0
        self._fps = 24.0
        self._current_frame = 0

        # Render settings (thread-safe via _lock)
        self._width = 200
        self._height = 100
        self._char_set = " .,:;+*?%S#@"
        self._charset_hint = ""
        self._color_mode = "Colored"
        self._intensity = 100
        self._brightness = 100
        self._invert_ascii = False
        self._mono_color = (255, 255, 255)
        self._bg_color = (14, 14, 14)
        self._speed = 1.0
        self._aspect_lock = True
        self._aspect_preset = "Source"
        self._loop = True
        self._video_aspect = 1.77
        self._frame_consumed = True
        self._out_buf: np.ndarray | None = None

    def load_video(self, path: str):
        with self._lock:
            self._stop_flag = True
        self.msleep(50)
        with self._lock:
            self._video_path = path
            self._stop_flag = False
            self._playing = False
            self._seek_frame = 0

    def get_video_info(self) -> dict:
        with self._lock:
            return {
                "total_frames": self._total_frames,
                "fps": self._fps,
                "current_frame": self._current_frame,
                "video_aspect": self._video_aspect,
            }

    def play(self):
        with self._lock:
            self._playing = True

    def pause(self):
        with self._lock:
            self._playing = False

    def stop(self):
        with self._lock:
            self._playing = False
            self._seek_frame = 0

    def seek(self, frame_no: int):
        with self._lock:
            self._seek_frame = frame_no

    def mark_frame_consumed(self):
        with self._lock:
            self._frame_consumed = True

    def apply_settings(self, rs: RenderSettings):
        """Apply a RenderSettings snapshot to the thread."""
        with self._lock:
            self._width = rs.width
            self._height = rs.height
            self._char_set = rs.char_set
            self._color_mode = rs.color_mode
            self._intensity = rs.intensity
            self._brightness = rs.brightness
            self._invert_ascii = rs.invert_ascii
            self._mono_color = rs.mono_color
            self._bg_color = rs.bg_color
            self._speed = rs.speed
            self._aspect_lock = rs.aspect_lock
            self._aspect_preset = rs.aspect_preset
            self._loop = rs.loop
            self._out_buf = None  # Invalidate buffer on settings change

    def get_settings(self) -> RenderSettings:
        """Return an immutable snapshot of the current render settings."""
        with self._lock:
            return RenderSettings(
                width=self._width,
                height=self._height,
                char_set=self._char_set,
                color_mode=self._color_mode,
                intensity=self._intensity,
                brightness=self._brightness,
                invert_ascii=self._invert_ascii,
                mono_color=self._mono_color,
                bg_color=self._bg_color,
                speed=self._speed,
                aspect_lock=self._aspect_lock,
                aspect_preset=self._aspect_preset,
                loop=self._loop,
            )

    def shutdown(self):
        with self._lock:
            self._shutdown = True
            self._stop_flag = True
            self._playing = False
        self.wait(5000)

    def run(self):
        while True:
            with self._lock:
                if self._shutdown:
                    return
                path = self._video_path
                self._stop_flag = False

            if not path:
                self.msleep(30)
                continue

            try:
                cap = cv2.VideoCapture(path)
                if not cap.isOpened():
                    self.error_occurred.emit(f"Cannot open: {path}")
                    with self._lock:
                        self._video_path = ""
                    continue
            except Exception as e:
                traceback.print_exc(file=sys.stderr)
                self.error_occurred.emit(str(e))
                with self._lock:
                    self._video_path = ""
                continue

            with self._lock:
                self._total_frames = max(1, int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))
                self._fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
                fw = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
                fh = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
                self._video_aspect = (fw / fh) if fh > 0 else 1.77
                self._current_frame = 0

            # Render first frame immediately
            self._render_current(cap)

            # Deadline-based playback loop
            next_frame_time = time.perf_counter()

            while True:
                with self._lock:
                    if self._stop_flag or self._shutdown:
                        break
                    playing = self._playing
                    seek = self._seek_frame
                    self._seek_frame = None
                    speed = self._speed
                    fps_local = self._fps

                # Handle seek requests
                if seek is not None:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, seek)
                    with self._lock:
                        self._current_frame = seek
                    self._render_current(cap)
                    next_frame_time = time.perf_counter()
                    if not playing:
                        self.msleep(16)
                        continue

                if not playing:
                    next_frame_time = time.perf_counter()
                    self.msleep(16)
                    continue

                # Check if the GUI has consumed the previous frame
                with self._lock:
                    consumed = self._frame_consumed
                if not consumed:
                    self.msleep(2)
                    continue

                # Deadline-based timing: compute how many frames behind we are
                now = time.perf_counter()
                frame_interval = (1.0 / fps_local) / speed if speed > 0 else 1.0 / fps_local

                if now < next_frame_time:
                    # We're ahead of schedule — sleep until the deadline
                    sleep_ms = max(1, int((next_frame_time - now) * 1000))
                    self.msleep(sleep_ms)
                    continue

                # How many frames should we have advanced by now?
                frames_behind = int((now - next_frame_time) / frame_interval)

                # Skip frames if we're more than 1 frame behind
                if frames_behind > 1:
                    skip = min(frames_behind - 1, 10)  # Cap skipping to avoid long hangs
                    for _ in range(skip):
                        ret, _ = cap.read()
                        if not ret:
                            break
                        with self._lock:
                            self._current_frame = int(cap.get(cv2.CAP_PROP_POS_FRAMES))

                # Read and render the next frame
                ret, frame = cap.read()
                if not ret:
                    with self._lock:
                        loop = self._loop
                    if loop:
                        # Loop: seek back to start and keep playing
                        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        with self._lock:
                            self._current_frame = 0
                        next_frame_time = time.perf_counter()
                        continue
                    else:
                        self.playback_finished.emit()
                        with self._lock:
                            self._playing = False
                            self._seek_frame = 0
                        continue

                with self._lock:
                    self._current_frame = int(cap.get(cv2.CAP_PROP_POS_FRAMES))

                self._do_render(frame)

                # Advance deadline by one frame interval
                next_frame_time += frame_interval
                # If we've fallen too far behind, reset the deadline
                if next_frame_time < now - frame_interval * 3:
                    next_frame_time = now

            cap.release()
            with self._lock:
                if self._shutdown:
                    return
                if self._video_path == path:
                    self._video_path = ""

    def _render_current(self, cap: cv2.VideoCapture):
        """Render the frame at the current position without advancing."""
        ret, frame = cap.read()
        if ret:
            pos = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
            cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, pos - 1))
            self._do_render(frame)

    def _do_render(self, frame: np.ndarray):
        """Convert a raw BGR frame to ASCII and emit the result as a QImage."""
        with self._lock:
            w = self._width
            h = self._height
            cs = self._char_set
            ch = self._charset_hint
            cm = self._color_mode
            intensity = self._intensity
            brightness = self._brightness
            invert_ascii = self._invert_ascii
            mc = self._mono_color
            bg = self._bg_color
            total = self._total_frames
            cur = self._current_frame
            aspect_lock = self._aspect_lock
            va = self._video_aspect
            out_buf = self._out_buf

        if aspect_lock and va > 0:
            h = max(1, int(w / va * 0.5))

        t0 = time.perf_counter()

        try:
            chars_2d, colors_rgb = frame_to_ascii(
                frame, w, h, cs, cm, intensity, mc, brightness, invert_ascii,
            )
        except Exception as e:
            traceback.print_exc(file=sys.stderr)
            self.error_occurred.emit(f"Render error: {e}")
            return

        font_px = get_preview_font_px(w)
        atlas = get_atlas(cs, font_px, ch)

        # Reuse or allocate output buffer
        img_h = h * atlas.cell_h
        img_w = w * atlas.cell_w
        if out_buf is not None and out_buf.shape == (img_h, img_w, 3):
            pass
        else:
            out_buf = np.full((img_h, img_w, 3), bg[0], dtype=np.uint8)
            with self._lock:
                self._out_buf = out_buf

        rgb_array = atlas.compose_frame(chars_2d, colors_rgb, bg, out_buf)

        qimg = QImage(
            rgb_array.data,
            rgb_array.shape[1],
            rgb_array.shape[0],
            rgb_array.strides[0],
            QImage.Format.Format_RGB888,
        ).copy()

        render_ms = (time.perf_counter() - t0) * 1000.0

        with self._lock:
            self._frame_consumed = False
        self.frame_rendered.emit(qimg, chars_2d, colors_rgb, cur, total, render_ms)
