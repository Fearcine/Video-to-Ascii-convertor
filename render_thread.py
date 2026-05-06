import time
import threading
import numpy as np
import cv2
from PyQt6.QtCore import QThread, pyqtSignal
from PyQt6.QtGui import QImage
from ascii_renderer import frame_to_ascii
from glyph_atlas import get_atlas


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


        self._width = 200
        self._height = 100
        self._char_set = " .,:;+*?%S#@"
        self._color_mode = "Colored"
        self._intensity = 80
        self._mono_color = (255, 255, 255)
        self._speed = 1.0
        self._aspect_lock = True
        self._video_aspect = 1.77
        self._frame_consumed = True
        self._out_buf: np.ndarray | None = None
        self._preview_font_px = 0


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

        self._frame_consumed = True

    def update_settings(
        self,
        width: int | None = None,
        height: int | None = None,
        char_set: str | None = None,
        color_mode: str | None = None,
        intensity: int | None = None,
        mono_color: tuple[int, int, int] | None = None,
        speed: float | None = None,
        aspect_lock: bool | None = None,
    ):
        with self._lock:
            if width is not None:
                self._width = width
            if height is not None:
                self._height = height
            if char_set is not None:
                self._char_set = char_set
            if color_mode is not None:
                self._color_mode = color_mode
            if intensity is not None:
                self._intensity = intensity
            if mono_color is not None:
                self._mono_color = mono_color
            if speed is not None:
                self._speed = speed
            if aspect_lock is not None:
                self._aspect_lock = aspect_lock
            self._out_buf = None

    def shutdown(self):
        with self._lock:
            self._shutdown = True
            self._stop_flag = True
            self._playing = False
        self.wait(5000)

   

    def _get_preview_font_px(self, ascii_width: int) -> int:
        
        if ascii_width <= 150:
            return 10
        elif ascii_width <= 300:
            return 7
        elif ascii_width <= 500:
            return 5
        else:
            return 4

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
                total = self._total_frames
                fps = self._fps


            self._render_current(cap)

            while True:
                loop_start = time.perf_counter()

                with self._lock:
                    if self._stop_flag or self._shutdown:
                        break
                    playing = self._playing
                    seek = self._seek_frame
                    self._seek_frame = None
                    speed = self._speed
                    fps_local = self._fps

                if seek is not None:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, seek)
                    with self._lock:
                        self._current_frame = seek
                    self._render_current(cap)
                    if not playing:
                        self.msleep(16)
                        continue

                if not playing:
                    self.msleep(16)
                    continue


                if not self._frame_consumed:
                    self.msleep(4)
                    continue

                ret, frame = cap.read()
                if not ret:
                    self.playback_finished.emit()
                    with self._lock:
                        self._playing = False
                        self._seek_frame = 0
                    continue

                with self._lock:
                    self._current_frame = int(cap.get(cv2.CAP_PROP_POS_FRAMES))

                self._do_render(frame)

                target_delay = (1.0 / fps_local) / speed if speed > 0 else 1.0 / fps_local
                elapsed = time.perf_counter() - loop_start
                sleep_ms = max(1, int((target_delay - elapsed) * 1000))
                self.msleep(sleep_ms)

            cap.release()
            with self._lock:
                if self._shutdown:
                    return

                if self._video_path == path:
                    self._video_path = ""

    def _render_current(self, cap: cv2.VideoCapture):
        ret, frame = cap.read()
        if ret:
            pos = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
            cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, pos - 1))
            self._do_render(frame)

    def _do_render(self, frame: np.ndarray):
        with self._lock:
            w = self._width
            h = self._height
            cs = self._char_set
            cm = self._color_mode
            intensity = self._intensity
            mc = self._mono_color
            total = self._total_frames
            cur = self._current_frame
            aspect_lock = self._aspect_lock
            va = self._video_aspect

        if aspect_lock and va > 0:
            h = max(1, int(w / va * 0.5))

        t0 = time.perf_counter()

        try:
            chars_2d, colors_rgb = frame_to_ascii(frame, w, h, cs, cm, intensity, mc)
        except Exception as e:
            self.error_occurred.emit(f"Render error: {e}")
            return


        font_px = self._get_preview_font_px(w)
        atlas = get_atlas(cs, font_px)


        img_h = h * atlas.cell_h
        img_w = w * atlas.cell_w
        if (self._out_buf is not None and 
            self._out_buf.shape == (img_h, img_w, 3)):
            out_buf = self._out_buf
        else:
            out_buf = np.full((img_h, img_w, 3), 14, dtype=np.uint8)
            self._out_buf = out_buf

        rgb_array = atlas.compose_frame(chars_2d, colors_rgb, (14, 14, 14), out_buf)


        qimg = QImage(
            rgb_array.data,
            rgb_array.shape[1],
            rgb_array.shape[0],
            rgb_array.strides[0],
            QImage.Format.Format_RGB888,
        )

        qimg = qimg.copy()

        render_ms = (time.perf_counter() - t0) * 1000.0

        self._frame_consumed = False
        self.frame_rendered.emit(qimg, chars_2d, colors_rgb, cur, total, render_ms)
