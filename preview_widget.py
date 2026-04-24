from PyQt6.QtWidgets import QWidget
from PyQt6.QtGui import QPainter, QImage, QColor
from PyQt6.QtCore import Qt, QRect


class PreviewWidget(QWidget):
    

    def __init__(self, parent=None):
        super().__init__(parent)
        self._image: QImage | None = None
        self.setMinimumSize(200, 150)
        self.setStyleSheet("background-color: #0e0e0e;")

    def update_image(self, qimage: QImage):
        
        self._image = qimage
        self.update()

    def clear(self):
        self._image = None
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor(14, 14, 14))

        if self._image is None or self._image.isNull():
            painter.setPen(QColor(80, 80, 80))
            painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, "Load a video to see preview")
            painter.end()
            return

        # Scale image to fit the widget, maintaining aspect ratio
        img_w = self._image.width()
        img_h = self._image.height()
        wgt_w = self.width()
        wgt_h = self.height()

        if img_w <= 0 or img_h <= 0:
            painter.end()
            return

        scale = min(wgt_w / img_w, wgt_h / img_h)  # Scale to fill widget, maintaining aspect
        dst_w = int(img_w * scale)
        dst_h = int(img_h * scale)
        x = (wgt_w - dst_w) // 2
        y = (wgt_h - dst_h) // 2

        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, scale < 0.5)
        painter.drawImage(QRect(x, y, dst_w, dst_h), self._image)
        painter.end()
