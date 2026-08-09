import random
from PyQt6.QtWidgets import QWidget, QPushButton, QVBoxLayout
from PyQt6.QtGui import QPainter, QColor, QFont, QPixmap
from PyQt6.QtCore import QTimer, Qt

class MatrixLoadingScreen(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.font_size = 20
        self.drops = []
        self.buffer = None
        
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_matrix)
        
        layout = QVBoxLayout(self)
        self.btn_continue = QPushButton("Continue")
        self.btn_continue.setFixedSize(200, 40)
        self.btn_continue.setStyleSheet("""
            QPushButton {
                background: #e0e0e0;
                border: 2px outset #d4d0c8;
                border-radius: 0px;
                color: #000000;
                font-family: 'Tahoma', 'MS Sans Serif', 'Segoe UI', sans-serif;
                font-size: 14px;
                font-weight: bold;
            }
            QPushButton:hover { background: #d0d0d0; }
            QPushButton:pressed { background: #c0c0c0; border: 2px inset #a0a0a0; }
        """)
        self.btn_continue.setCursor(Qt.CursorShape.PointingHandCursor)
        
        layout.addStretch()
        layout.addWidget(self.btn_continue, alignment=Qt.AlignmentFlag.AlignCenter)
        layout.addStretch()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.buffer = QPixmap(self.size())
        self.buffer.fill(Qt.GlobalColor.white)
        self.columns = self.width() // (self.font_size * 2)
        self.drops = [random.randint(-80, 0) for _ in range(self.columns)]
        
    def showEvent(self, event):
        super().showEvent(event)
        self.timer.start(50)
        
    def hideEvent(self, event):
        super().hideEvent(event)
        self.timer.stop()

    def update_matrix(self):
        if not self.buffer: return
        painter = QPainter(self.buffer)
        painter.fillRect(self.buffer.rect(), QColor(255, 255, 255, 20)) # Fade effect
        
        painter.setFont(QFont("Consolas", self.font_size))
        for i in range(len(self.drops)):
            # Draw char
            char = chr(random.choice([
                random.randint(0x30A0, 0x30FF), # Katakana
                random.randint(0x4E00, 0x9FBF), # Kanji
                random.randint(0x0041, 0x005A)  # Latin
            ]))
            painter.setPen(QColor(0, 0, 0))
            x = i * self.font_size * 2
            y = self.drops[i] * self.font_size
            painter.drawText(x, y, char)
            
            if y > self.height() and random.random() > 0.85:
                self.drops[i] = random.randint(-80, -10)
            self.drops[i] += 1
        painter.end()
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        if self.buffer:
            painter.drawPixmap(0, 0, self.buffer)
