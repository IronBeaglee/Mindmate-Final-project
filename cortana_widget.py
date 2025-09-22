# cortana_widget.py
from PyQt5.QtWidgets import QWidget
from PyQt5.QtCore import QTimer, Qt
from PyQt5.QtGui import QPainter, QColor, QBrush

class CortanaWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.radius = 40
        self.grow = True
        self.active = False

        self.timer = QTimer()
        self.timer.timeout.connect(self.animate)
        self.timer.start(50)  # smooth animation

    def animate(self):
        if self.active:
            if self.grow:
                self.radius += 1
                if self.radius >= 60:
                    self.grow = False
            else:
                self.radius -= 1
                if self.radius <= 40:
                    self.grow = True
        self.update()

    def set_active(self, speaking: bool):
        self.active = speaking
        if not speaking:
            self.radius = 40
            self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        cx = self.width() // 2
        cy = self.height() // 2

        # Glow
        glow = QColor(0, 191, 255, 180)  # cyan glow
        painter.setBrush(QBrush(glow))
        painter.setPen(Qt.NoPen)
        painter.drawEllipse(cx - self.radius, cy - self.radius, self.radius * 2, self.radius * 2)

        # Core inner circle
        core = QColor(0, 191, 255, 255)
        painter.setBrush(QBrush(core))
        painter.drawEllipse(cx - 20, cy - 20, 40, 40)
