"""Scaled preview widget for displaying ASCII art QImages."""

from PyQt6.QtWidgets import QWidget
from PyQt6.QtGui import QPainter, QImage, QColor
from PyQt6.QtCore import Qt, QRect


class PreviewWidget(QWidget):
    """Displays a QImage scaled to fit the widget while preserving aspect ratio."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._image: QImage | None = None
        self._bg_color = QColor(14, 14, 14)
        self.setMinimumSize(200, 150)

    def set_bg_color(self, r: int, g: int, b: int):
        """Set the background fill color (painted behind and around the image)."""
        self._bg_color = QColor(r, g, b)
        self.update()

    def update_image(self, qimage: QImage):
        """Replace the displayed image and repaint."""
        self._image = qimage
        self.update()

    def clear(self):
        self._image = None
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.fillRect(self.rect(), self._bg_color)

        if self._image is None or self._image.isNull():
            painter.setPen(QColor(80, 80, 80))
            painter.drawText(
                self.rect(), Qt.AlignmentFlag.AlignCenter,
                "Load a video or image to see preview",
            )
            painter.end()
            return

        img_w = self._image.width()
        img_h = self._image.height()
        wgt_w = self.width()
        wgt_h = self.height()

        if img_w <= 0 or img_h <= 0:
            painter.end()
            return

        scale = min(wgt_w / img_w, wgt_h / img_h)
        dst_w = int(img_w * scale)
        dst_h = int(img_h * scale)
        x = (wgt_w - dst_w) // 2
        y = (wgt_h - dst_h) // 2

        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, scale < 0.5)
        painter.drawImage(QRect(x, y, dst_w, dst_h), self._image)
        painter.end()
