from PyQt5.QtWidgets import QLabel, QVBoxLayout, QWidget, QPushButton
from PyQt5.QtCore import Qt, QTimer, pyqtSignal
from PyQt5.QtGui import QDragEnterEvent, QDropEvent


class DragDropLabel(QWidget):
    """Виджет с поддержкой drag-and-drop и кнопкой камеры"""

    clicked = pyqtSignal()
    camera_clicked = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(10)
        layout.setContentsMargins(10, 10, 10, 10)

        # Основная метка для drag-and-drop
        self.drop_label = QLabel()
        self.drop_label.setAlignment(Qt.AlignCenter)
        self.drop_label.setText("Перетащите изображение сюда\nили нажмите для выбора")
        self.drop_label.setStyleSheet("""
            QLabel {
                border: 2px dashed #ccc;
                border-radius: 10px;
                padding: 40px 20px;
                background-color: #f9f9f9;
                font-size: 14px;
                color: #666;
                min-height: 200px;
            }
        """)
        self.drop_label.setMinimumSize(400, 250)

        # Кнопка для камеры
        self.camera_button = QPushButton("Сделать фото")
        self.camera_button.setStyleSheet("""
            QPushButton {
                background-color: #2196F3;
                color: white;
                border: none;
                border-radius: 5px;
                padding: 10px 15px;
                font-size: 13px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #0b7dda;
            }
            QPushButton:pressed {
                background-color: #0d47a1;
            }
        """)
        self.camera_button.clicked.connect(self.camera_clicked.emit)

        layout.addWidget(self.drop_label)
        layout.addWidget(self.camera_button)

        self.setMouseTracking(True)

    def setText(self, text):
        """Устанавливает текст в основную метку"""
        self.drop_label.setText(text)

    def text(self):
        """Возвращает текст основной метки"""
        return self.drop_label.text()

    def enterEvent(self, event):
        self.drop_label.setStyleSheet("""
            QLabel {
                border: 2px dashed #0078d7;
                border-radius: 10px;
                padding: 40px 20px;
                background-color: #f0f8ff;
                font-size: 14px;
                color: #0078d7;
                min-height: 200px;
            }
        """)
        super().enterEvent(event)

    def leaveEvent(self, event):
        self.drop_label.setStyleSheet("""
            QLabel {
                border: 2px dashed #ccc;
                border-radius: 10px;
                padding: 40px 20px;
                background-color: #f9f9f9;
                font-size: 14px;
                color: #666;
                min-height: 200px;
            }
        """)
        super().leaveEvent(event)

    def dragEnterEvent(self, event: QDragEnterEvent):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
            self.drop_label.setStyleSheet("""
                QLabel {
                    border: 2px solid #0078d7;
                    border-radius: 10px;
                    padding: 40px 20px;
                    background-color: #e1f5fe;
                    font-size: 14px;
                    color: #0078d7;
                    min-height: 200px;
                }
            """)

    def dragLeaveEvent(self, event):
        self.drop_label.setStyleSheet("""
            QLabel {
                border: 2px dashed #0078d7;
                border-radius: 10px;
                padding: 40px 20px;
                background-color: #f0f8ff;
                font-size: 14px;
                color: #0078d7;
                min-height: 200px;
            }
        """)
        super().dragLeaveEvent(event)

    def dropEvent(self, event: QDropEvent):
        urls = event.mimeData().urls()
        if urls:
            file_path = urls[0].toLocalFile()
            if file_path.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
                self.parent().load_image(file_path)
        self.drop_label.setStyleSheet("""
            QLabel {
                border: 2px dashed #ccc;
                border-radius: 10px;
                padding: 40px 20px;
                background-color: #f9f9f9;
                font-size: 14px;
                color: #666;
                min-height: 200px;
            }
        """)

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            # Проверяем, была ли нажата основная метка
            if self.drop_label.geometry().contains(event.pos()):
                # Визуальная обратная связь при клике
                self.drop_label.setStyleSheet("""
                    QLabel {
                        border: 2px solid #005a9e;
                        border-radius: 10px;
                        padding: 40px 20px;
                        background-color: #daefff;
                        font-size: 14px;
                        color: #005a9e;
                        min-height: 200px;
                    }
                """)
                QTimer.singleShot(200, self.reset_style)
                self.clicked.emit()
        super().mousePressEvent(event)

    def reset_style(self):
        self.drop_label.setStyleSheet("""
            QLabel {
                border: 2px dashed #ccc;
                border-radius: 10px;
                padding: 40px 20px;
                background-color: #f9f9f9;
                font-size: 14px;
                color: #666;
                min-height: 200px;
            }
        """)
