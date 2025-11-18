from PyQt5.QtWidgets import QLabel
from PyQt5.QtCore import Qt, QTimer, pyqtSignal
from PyQt5.QtGui import QDragEnterEvent, QDropEvent


class DragDropLabel(QLabel):
    """QLabel с поддержкой drag-and-drop"""

    clicked = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAlignment(Qt.AlignCenter)
        self.setText("Перетащите изображение сюда\nили нажмите для выбора")
        self.setStyleSheet("""
            QLabel {
                border: 2px dashed #ccc;
                border-radius: 10px;
                padding: 20px;
                background-color: #f9f9f9;
                font-size: 14px;
                color: #666;
            }
        """)
        self.setMinimumSize(400, 300)
        self.setMouseTracking(True)

    def enterEvent(self, event):
        self.setStyleSheet("""
            QLabel {
                border: 2px dashed #0078d7;
                border-radius: 10px;
                padding: 20px;
                background-color: #f0f8ff;
                font-size: 14px;
                color: #0078d7;
            }
        """)
        super().enterEvent(event)

    def leaveEvent(self, event):
        self.setStyleSheet("""
            QLabel {
                border: 2px dashed #ccc;
                border-radius: 10px;
                padding: 20px;
                background-color: #f9f9f9;
                font-size: 14px;
                color: #666;
            }
        """)
        super().leaveEvent(event)

    def dragEnterEvent(self, event: QDragEnterEvent):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
            self.setStyleSheet("""
                QLabel {
                    border: 2px solid #0078d7;
                    border-radius: 10px;
                    padding: 20px;
                    background-color: #e1f5fe;
                    font-size: 14px;
                    color: #0078d7;
                }
            """)

    def dragLeaveEvent(self, event):
        self.setStyleSheet("""
            QLabel {
                border: 2px dashed #0078d7;
                border-radius: 10px;
                padding: 20px;
                background-color: #f0f8ff;
                font-size: 14px;
                color: #0078d7;
            }
        """)
        super().dragLeaveEvent(event)

    def dropEvent(self, event: QDropEvent):
        urls = event.mimeData().urls()
        if urls:
            file_path = urls[0].toLocalFile()
            if file_path.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
                self.parent().load_image(file_path)
        self.setStyleSheet("""
            QLabel {
                border: 2px dashed #ccc;
                border-radius: 10px;
                padding: 20px;
                background-color: #f9f9f9;
                font-size: 14px;
                color: #666;
            }
        """)

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            # Визуальная обратная связь при клике
            self.setStyleSheet("""
                QLabel {
                    border: 2px solid #005a9e;
                    border-radius: 10px;
                    padding: 20px;
                    background-color: #daefff;
                    font-size: 14px;
                    color: #005a9e;
                }
            """)
            QTimer.singleShot(200, self.reset_style)  # Сбрасываем стиль через 200мс
            self.clicked.emit()
        super().mousePressEvent(event)

    def reset_style(self):
        self.setStyleSheet("""
            QLabel {
                border: 2px dashed #ccc;
                border-radius: 10px;
                padding: 20px;
                background-color: #f9f9f9;
                font-size: 14px;
                color: #666;
            }
        """)