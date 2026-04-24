import os
import cv2
import numpy as np
from PyQt6.QtCore import QThread, pyqtSignal
from ascii_renderer import (
    frame_to_ascii,
    frame_to_plain_text,
    frame_to_html,
)
from glyph_atlas import get_atlas


class ExportVideoThread(QThread):
   

    progress = pyqtSignal(int)
    finished_ok = pyqtSignal(str)
    error_occurred = pyqtSignal(str)

    def __init__(
        self,
        video_path: str,
        output_path: str,
        width: int,
        height: int,
        char_set: str,
        color_mode: str,
        intensity: int,
        mono_color: tuple[int, int, int],
        aspect_lock: bool,
        parent=None,
    ):
        super().__init__(parent)
        self._video_path = video_path
        self._output_path = output_path
        self._width = width
        self._height = height
        self._char_set = char_set
        self._color_mode = color_mode
        self._intensity = intensity
        self._mono_color = mono_color
        self._aspect_lock = aspect_lock
        self._cancelled = False

    def cancel(self):
        self._cancelled = True

    def run(self):
        try:
            cap = cv2.VideoCapture(self._video_path)
            if not cap.isOpened():
                self.error_occurred.emit(f"Cannot open: {self._video_path}")
                return

            total = max(1, int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))
            fw = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
            fh = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
            va = (fw / fh) if fh > 0 else 1.77

            w, h = self._width, self._height
            if self._aspect_lock and va > 0:
                h = max(1, int(w / va * 0.5))

            with open(self._output_path, "w", encoding="utf-8") as f:
                n = 0
                while not self._cancelled:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    chars, colors = frame_to_ascii(
                        frame, w, h, self._char_set,
                        self._color_mode, self._intensity, self._mono_color,
                    )
                    text = frame_to_plain_text(chars)
                    f.write("FRAME_START\n")
                    f.write(text)
                    f.write("\nFRAME_END\n")
                    n += 1
                    self.progress.emit(min(100, int(n / total * 100)))

            cap.release()

            if self._cancelled:
                try:
                    os.remove(self._output_path)
                except OSError:
                    pass
                return

            self.finished_ok.emit(self._output_path)

        except Exception as e:
            self.error_occurred.emit(str(e))


class ExportMP4Thread(QThread):
    
    progress = pyqtSignal(int)
    finished_ok = pyqtSignal(str)
    error_occurred = pyqtSignal(str)

    def __init__(
        self,
        video_path: str,
        output_path: str,
        width: int,
        height: int,
        char_set: str,
        color_mode: str,
        intensity: int,
        mono_color: tuple[int, int, int],
        font_size: int,
        aspect_lock: bool,
        parent=None,
    ):
        super().__init__(parent)
        self._video_path = video_path
        self._output_path = output_path
        self._width = width
        self._height = height
        self._char_set = char_set
        self._color_mode = color_mode
        self._intensity = intensity
        self._mono_color = mono_color
        self._font_size = font_size
        self._aspect_lock = aspect_lock
        self._cancelled = False

    def cancel(self):
        self._cancelled = True

    def run(self):
        cap = None
        writer = None
        try:
            cap = cv2.VideoCapture(self._video_path)
            if not cap.isOpened():
                self.error_occurred.emit(f"Cannot open: {self._video_path}")
                return

            total = max(1, int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))
            src_fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
            fw = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
            fh = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
            va = (fw / fh) if fh > 0 else 1.77

            ascii_w = self._width
            ascii_h = self._height
            if self._aspect_lock and va > 0:
                ascii_h = max(1, int(ascii_w / va * 0.5))

            # Get the glyph atlas for this charset + font size
            atlas = get_atlas(self._char_set, self._font_size)

            # Compute output pixel dimensions from atlas cell size
            out_px_h = ascii_h * atlas.cell_h
            out_px_w = ascii_w * atlas.cell_w

            # Ensure dimensions are even (required by most codecs)
            out_px_w = out_px_w if out_px_w % 2 == 0 else out_px_w + 1
            out_px_h = out_px_h if out_px_h % 2 == 0 else out_px_h + 1

            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(self._output_path, fourcc, src_fps, (out_px_w, out_px_h))
            if not writer.isOpened():
                self.error_occurred.emit("Failed to create output video writer.")
                return

            # Pre-allocate output buffer — reused for every frame
            rgb_buf = np.full((ascii_h * atlas.cell_h, ascii_w * atlas.cell_w, 3), 17, dtype=np.uint8)

            n = 0
            while not self._cancelled:
                ret, frame = cap.read()
                if not ret:
                    break

                chars, colors = frame_to_ascii(
                    frame, ascii_w, ascii_h,
                    self._char_set, self._color_mode, self._intensity, self._mono_color,
                )

                # Render via atlas into reusable buffer
                rgb_frame = atlas.compose_frame(chars, colors, (17, 17, 17), rgb_buf)
                bgr_frame = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR)

                # Pad if needed 
                padded = self._pad_frame(bgr_frame, out_px_w, out_px_h)
                writer.write(padded)

                n += 1
                if n % 5 == 0 or n == total:
                    self.progress.emit(min(100, int(n / total * 100)))

            writer.release()
            cap.release()

            if self._cancelled:
                try:
                    os.remove(self._output_path)
                except OSError:
                    pass
                return

            self.finished_ok.emit(self._output_path)

        except Exception as e:
            self.error_occurred.emit(str(e))
        finally:
            if writer is not None and writer.isOpened():
                writer.release()
            if cap is not None and cap.isOpened():
                cap.release()

    @staticmethod
    def _pad_frame(frame: np.ndarray, target_w: int, target_h: int) -> np.ndarray:
       
        h, w = frame.shape[:2]
        if w == target_w and h == target_h:
            return frame
        padded = np.full((target_h, target_w, 3), 17, dtype=np.uint8)
        padded[:h, :w] = frame
        return padded

def save_current_frame_txt(chars_2d: np.ndarray, output_path: str):
    
    text = frame_to_plain_text(chars_2d)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(text)


def save_current_frame_html(
    chars_2d: np.ndarray,
    colors_rgb: np.ndarray,
    output_path: str,
    font_size: int = 8,
):
    
    html = frame_to_html(chars_2d, colors_rgb, font_size)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)


def export_full_html(
    video_path: str,
    output_path: str,
    frame_no: int,
    width: int,
    height: int,
    char_set: str,
    color_mode: str,
    intensity: int,
    mono_color: tuple[int, int, int],
    font_size: int,
    aspect_lock: bool,
):
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Cannot open: {video_path}")

    fw = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    fh = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    va = (fw / fh) if fh > 0 else 1.77
    if aspect_lock and va > 0:
        height = max(1, int(width / va * 0.5))

    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_no)
    ret, frame = cap.read()
    cap.release()
    if not ret:
        raise IOError(f"Cannot read frame {frame_no}")

    chars, colors = frame_to_ascii(frame, width, height, char_set, color_mode, intensity, mono_color)
    html = frame_to_html(chars, colors, font_size)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)
