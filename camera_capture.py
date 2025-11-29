import cv2
import numpy as np
from PyQt5.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout,
                             QPushButton, QLabel, QMessageBox)
from PyQt5.QtCore import QTimer, Qt, pyqtSignal
from PyQt5.QtGui import QImage, QPixmap, QFont


class CameraCaptureDialog(QDialog):
    """Диалог для захвата фото с веб-камеры"""

    image_captured = pyqtSignal(np.ndarray)  # Сигнал с захваченным изображением

    def __init__(self, parent=None):
        super().__init__(parent)
        self.camera = None
        self.captured_image = None
        self.init_ui()
        self.init_camera()

    def init_ui(self):
        self.setWindowTitle("Сделать фото - Веб-камера")
        self.setFixedSize(800, 700)

        layout = QVBoxLayout(self)

        # Заголовок
        title_label = QLabel("Сделать фото с веб-камеры")
        title_label.setFont(QFont("Arial", 14, QFont.Bold))
        title_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(title_label)

        # Область предпросмотра камеры
        self.camera_label = QLabel()
        self.camera_label.setAlignment(Qt.AlignCenter)
        self.camera_label.setMinimumSize(640, 480)
        self.camera_label.setStyleSheet("""
            QLabel {
                border: 2px solid #ccc;
                background-color: #000;
            }
        """)
        layout.addWidget(self.camera_label)

        # Информационная метка
        self.info_label = QLabel("Нажмите 'Сделать фото' для захвата изображения")
        self.info_label.setAlignment(Qt.AlignCenter)
        self.info_label.setStyleSheet("color: #666;")
        layout.addWidget(self.info_label)

        # Кнопки управления
        buttons_layout = QHBoxLayout()

        self.capture_btn = QPushButton("Сделать фото")
        self.capture_btn.setStyleSheet("""
            QPushButton {
                background-color: #4CAF50;
                color: white;
                font-weight: bold;
                padding: 10px;
                border: none;
                border-radius: 5px;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
        """)
        self.capture_btn.clicked.connect(self.capture_image)

        self.retry_btn = QPushButton("Повторить")
        self.retry_btn.setStyleSheet("""
            QPushButton {
                background-color: #ff9800;
                color: white;
                padding: 10px;
                border: none;
                border-radius: 5px;
            }
            QPushButton:hover {
                background-color: #e68900;
            }
        """)
        self.retry_btn.clicked.connect(self.retry_capture)
        self.retry_btn.setVisible(False)

        self.accept_btn = QPushButton("Использовать это фото")
        self.accept_btn.setStyleSheet("""
            QPushButton {
                background-color: #2196F3;
                color: white;
                font-weight: bold;
                padding: 10px;
                border: none;
                border-radius: 5px;
            }
            QPushButton:hover {
                background-color: #0b7dda;
            }
        """)
        self.accept_btn.clicked.connect(self.accept_image)
        self.accept_btn.setVisible(False)

        self.cancel_btn = QPushButton("Отмена")
        self.cancel_btn.clicked.connect(self.reject)

        buttons_layout.addWidget(self.capture_btn)
        buttons_layout.addWidget(self.retry_btn)
        buttons_layout.addWidget(self.accept_btn)
        buttons_layout.addWidget(self.cancel_btn)

        layout.addLayout(buttons_layout)

        # Таймер для обновления кадра
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)

    def init_camera(self):
        """Инициализация веб-камеры"""
        try:
            self.camera = cv2.VideoCapture(0)
            if not self.camera.isOpened():
                # Пробуем другие индексы камер
                for i in range(1, 5):
                    self.camera = cv2.VideoCapture(i)
                    if self.camera.isOpened():
                        break

            if not self.camera.isOpened():
                raise Exception("Не удалось найти доступную веб-камеру")

            # Устанавливаем разрешение
            self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

            self.timer.start(30)  # ~33 FPS
            self.info_label.setText("Камера активирована. Нажмите 'Сделать фото'")

        except Exception as e:
            self.show_error(f"Ошибка инициализации камеры: {str(e)}")

    def update_frame(self):
        """Обновление кадра с камеры"""
        if self.camera and self.camera.isOpened():
            ret, frame = self.camera.read()
            if ret:
                # Конвертируем BGR в RGB
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                # Создаем QImage из numpy массива
                h, w, ch = rgb_frame.shape
                bytes_per_line = ch * w
                qt_image = QImage(rgb_frame.data, w, h, bytes_per_line, QImage.Format_RGB888)

                # Масштабируем для отображения
                pixmap = QPixmap.fromImage(qt_image)
                scaled_pixmap = pixmap.scaled(self.camera_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
                self.camera_label.setPixmap(scaled_pixmap)

    def capture_image(self):
        """Захват текущего кадра"""
        if self.camera and self.camera.isOpened():
            ret, frame = self.camera.read()
            if ret:
                # Сохраняем захваченное изображение
                self.captured_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                # Показываем захваченное изображение
                self.show_captured_image()

                # Обновляем UI
                self.capture_btn.setVisible(False)
                self.retry_btn.setVisible(True)
                self.accept_btn.setVisible(True)
                self.info_label.setText("Фото захвачено! Нажмите 'Повторить' или 'Использовать это фото'")

                # Останавливаем камеру для экономии ресурсов
                self.timer.stop()
                if self.camera:
                    self.camera.release()

    def show_captured_image(self):
        """Показывает захваченное изображение"""
        if self.captured_image is not None:
            h, w, ch = self.captured_image.shape
            bytes_per_line = ch * w
            qt_image = QImage(self.captured_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(qt_image)
            scaled_pixmap = pixmap.scaled(self.camera_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
            self.camera_label.setPixmap(scaled_pixmap)

    def retry_capture(self):
        """Повторный захват фото"""
        self.captured_image = None
        self.init_camera()

        # Обновляем UI
        self.capture_btn.setVisible(True)
        self.retry_btn.setVisible(False)
        self.accept_btn.setVisible(False)
        self.info_label.setText("Нажмите 'Сделать фото' для захвата изображения")

    def accept_image(self):
        """Принятие захваченного изображения"""
        if self.captured_image is not None:
            self.image_captured.emit(self.captured_image)
            self.accept()

    def show_error(self, message):
        """Показ сообщения об ошибке"""
        self.info_label.setText(f"Ошибка: {message}")
        self.capture_btn.setEnabled(False)
        QMessageBox.critical(self, "Ошибка камеры", message)

    def closeEvent(self, event):
        """Очистка ресурсов при закрытии"""
        self.cleanup_camera()
        event.accept()

    def cleanup_camera(self):
        """Освобождение ресурсов камеры"""
        if self.timer.isActive():
            self.timer.stop()
        if self.camera and self.camera.isOpened():
            self.camera.release()

    def reject(self):
        """Отмена захвата"""
        self.cleanup_camera()
        super().reject()