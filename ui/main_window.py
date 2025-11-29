import cv2
from PyQt5.QtWidgets import (QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
                             QLabel, QTabWidget, QSplitter, QScrollArea,
                             QProgressBar, QMessageBox, QFileDialog)
from PyQt5.QtCore import QTimer, Qt
from PyQt5.QtGui import QFont, QPixmap, QImage
import numpy as np
import traceback
import os
from PIL import Image

from image_processor import ImageProcessor
from widgets import DragDropLabel
from processing_thread import ProcessingThread
from ui.tabs.correction_tab import CorrectionTab
from ui.tabs.quantization_tab import QuantizationTab
from ui.tabs.contours_tab import ContoursTab
from ui.tabs.export_tab import ExportTab
from ui.tabs.semantic_segmentation_tab import SemanticSegmentationTab
from ui.tabs.developer_console_tab import DeveloperConsoleTab
from camera_capture import CameraCaptureDialog


class PaintByNumbersApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.processor = ImageProcessor()
        self.current_image = None
        self.init_ui()

    def init_ui(self):
        self.setWindowTitle("Paint By Numbers - PyQt")
        self.setGeometry(100, 100, 1400, 900)

        # Центральный виджет
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        # Основной layout
        main_layout = QHBoxLayout(central_widget)

        # Splitter для резиновой разметки
        splitter = QSplitter(Qt.Horizontal)

        # Левая панель - настройки
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)

        # Виджет drag-and-drop с кнопкой камеры
        self.drop_label = DragDropLabel(self)
        self.drop_label.clicked.connect(self.browse_image)
        self.drop_label.camera_clicked.connect(self.capture_image)
        left_layout.addWidget(self.drop_label)

        # Вкладки настроек
        self.tabs = QTabWidget()

        # Инициализация вкладок
        self.correction_tab = CorrectionTab(self)
        self.quantization_tab = QuantizationTab(self)
        self.contours_tab = ContoursTab(self)
        self.export_tab = ExportTab(self)
        self.semantic_tab = SemanticSegmentationTab(self)
        self.developer_tab = DeveloperConsoleTab(self)

        self.tabs.addTab(self.correction_tab, "Коррекция")
        self.tabs.addTab(self.quantization_tab, "Квантование")
        self.tabs.addTab(self.contours_tab, "Контуры")
        self.tabs.addTab(self.semantic_tab, "Умная сегментация")
        self.tabs.addTab(self.developer_tab, "Консоль разработчика")
        self.tabs.addTab(self.export_tab, "Сохранение")

        left_layout.addWidget(self.tabs)

        # Правая панель - просмотр
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)

        # Заголовок предпросмотра
        preview_label = QLabel("Предпросмотр")
        preview_label.setFont(QFont("Arial", 12, QFont.Bold))
        right_layout.addWidget(preview_label)

        # Область просмотра с прокруткой
        self.scroll_area = QScrollArea()
        self.preview_label = QLabel()
        self.preview_label.setAlignment(Qt.AlignCenter)
        self.preview_label.setMinimumSize(400, 400)
        self.scroll_area.setWidget(self.preview_label)
        self.scroll_area.setWidgetResizable(True)
        right_layout.addWidget(self.scroll_area)

        # Прогресс бар
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        right_layout.addWidget(self.progress_bar)

        # Добавляем панели в splitter
        splitter.addWidget(left_panel)
        splitter.addWidget(right_panel)
        splitter.setSizes([400, 1000])

        main_layout.addWidget(splitter)

        # Настраиваем callback для метрик
        self.processor.set_metrics_callback(self.developer_tab.log_message)

    def browse_image(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Выберите изображение", "",
            "Images (*.png *.jpg *.jpeg *.bmp *.tiff)")
        if file_path:
            self.load_image(file_path)

    def capture_image(self):
        """Захват изображения с веб-камеры"""
        try:
            camera_dialog = CameraCaptureDialog(self)
            camera_dialog.image_captured.connect(self.on_image_captured)
            camera_dialog.exec_()

        except Exception as e:
            QMessageBox.critical(self, "Ошибка камеры",
                                 f"Не удалось запустить веб-камеру:\n{str(e)}")

    def on_image_captured(self, image_array):
        """Обработка захваченного с камеры изображения"""
        try:
            self.processor.load_image(image_array)
            self.current_image = image_array

            # Показываем превью
            self.update_preview(image_array)

            # Обновляем текст drag-drop области
            self.drop_label.setText(f"Загружено: фото с камеры\n"
                                    f"Размер: {image_array.shape[1]}x{image_array.shape[0]}")

            # Логируем информацию о загрузке
            self.developer_tab.log_message(f"Изображение захвачено с камеры", "info")
            self.developer_tab.log_message(
                f"Размер: {image_array.shape[1]}x{image_array.shape[0]}, Каналы: {image_array.shape[2]}", "info")

        except Exception as e:
            QMessageBox.critical(self, "Ошибка", f"Ошибка при загрузке изображения с камеры: {str(e)}")
            print(f"Полная ошибка: {traceback.format_exc()}")

    def load_image(self, file_path):
        try:
            print(f"Пытаемся загрузить: {file_path}")

            # Проверяем существование файла
            if not os.path.exists(file_path):
                QMessageBox.critical(self, "Ошибка", f"Файл не существует: {file_path}")
                return

            # Загружаем через PIL
            try:
                pil_image = Image.open(file_path)

                # Конвертируем в RGB если нужно
                if pil_image.mode != 'RGB':
                    pil_image = pil_image.convert('RGB')

                # Конвертируем PIL Image в numpy array
                rgb = np.array(pil_image)
                print(f"Успешно загружено через PIL, размер: {rgb.shape}")

            except Exception as pil_error:
                QMessageBox.critical(self, "Ошибка",
                                     f"Не удалось загрузить изображение через PIL: {str(pil_error)}")
                return

            # Проверяем, что изображение загружено корректно
            if rgb.size == 0:
                QMessageBox.critical(self, "Ошибка", "Изображение загружено пустым")
                return

            self.processor.load_image(rgb)
            self.current_image = rgb

            # Показываем превью
            self.update_preview(rgb)

            # Обновляем текст drag-drop области
            self.drop_label.setText(f"Загружено: {os.path.basename(file_path)}\n"
                                    f"Размер: {rgb.shape[1]}x{rgb.shape[0]}")

            # Логируем информацию о загрузке
            self.developer_tab.log_message(f"Изображение загружено: {os.path.basename(file_path)}", "info")
            self.developer_tab.log_message(f"Размер: {rgb.shape[1]}x{rgb.shape[0]}, Каналы: {rgb.shape[2]}", "info")

        except Exception as e:
            QMessageBox.critical(self, "Ошибка", f"Ошибка при загрузке: {str(e)}")
            print(f"Полная ошибка: {traceback.format_exc()}")

    def apply_correction(self):
        if self.current_image is None:
            QMessageBox.warning(self, "Предупреждение", "Сначала загрузите изображение")
            return

        brightness = self.correction_tab.brightness_slider.value()
        contrast = self.correction_tab.contrast_slider.value() / 100.0
        saturation = self.correction_tab.saturation_slider.value() / 100.0

        # Логируем параметры коррекции
        self.developer_tab.log_message(
            f"Применение коррекции: яркость={brightness}, контраст={contrast:.2f}, насыщенность={saturation:.2f}",
            "info")

        # Запускаем в отдельном потоке
        self.start_processing("adjust",
                              brightness=brightness,
                              contrast=contrast,
                              saturation=saturation)

    def apply_quantization(self):
        if self.current_image is None:
            QMessageBox.warning(self, "Предупреждение", "Сначала загрузите изображение")
            return

        method_map = {
            "KMeans (sklearn)": "kmeans_sklearn",
            "Median Cut": "median_cut",
            "Frequency Smooth": "frequency_smooth"
        }

        method = method_map.get(self.quantization_tab.method_combo.currentText(), "kmeans_sklearn")
        n_colors = self.quantization_tab.colors_spin.value()
        blur_k = self.quantization_tab.blur_spin.value()
        min_area = self.quantization_tab.area_spin.value()

        # Логируем параметры квантования
        self.developer_tab.log_message(f"Применение квантования: метод={method}, цвета={n_colors}, размытие={blur_k}",
                                       "info")

        self.start_processing("quantize",
                              n_colors=n_colors,
                              blur_k=blur_k,
                              method=method)

        # После квантования объединяем мелкие регионы
        def apply_merge():
            quantized, labels = self.processor.quantized, self.processor.labels
            if quantized is not None and labels is not None:
                merged, new_labels = self.processor.merge_small_regions(min_area_px=min_area)
                self.update_preview(merged)
                self.developer_tab.log_message(f"Объединение мелких регионов: мин. площадь={min_area} пикс.", "info")

        QTimer.singleShot(100, apply_merge)

    def apply_contours(self):
        if self.processor.labels is None:
            QMessageBox.warning(self, "Предупреждение", "Сначала примените квантование")
            return

        method_map = {
            "Cluster Borders": "cluster_borders",
            "Canny Edges": "canny"
        }

        method = method_map.get(self.contours_tab.contour_method_combo.currentText(), "cluster_borders")
        borders = self.processor.get_borders(method=method)

        result = self.processor.render_with_borders_and_numbers(
            border_mask=borders,
            border_thickness=self.contours_tab.thickness_spin.value(),
            show_numbers=self.contours_tab.show_numbers_check.isChecked()
        )

        self.update_preview(result)

        # Логируем параметры контуров
        self.developer_tab.log_message(
            f"Применение контуров: метод={method}, толщина={self.contours_tab.thickness_spin.value()}, номера={self.contours_tab.show_numbers_check.isChecked()}",
            "info")

    def preview_semantic_segmentation(self):
        """Предпросмотр семантической сегментации"""
        if self.current_image is None:
            QMessageBox.warning(self, "Предупреждение", "Сначала загрузите изображение")
            return

        try:
            parameters = self.semantic_tab.get_parameters()

            # Логируем параметры семантической сегментации
            self.developer_tab.log_message(f"Предпросмотр семантической сегментации: {parameters}", "info")

            # Применяем семантическую сегментацию
            semantic_mask = self.processor.semantic_segment(
                self.current_image,
                max_classes=parameters['max_classes']
            )

            # Визуализируем результат
            visualized = self.processor.visualize_semantic_mask(self.current_image)
            self.update_preview(visualized)

            QMessageBox.information(self, "Успех",
                                    f"Семантическая сегментация применена. Обнаружено классов: {len(self.processor.semantic_classes)}")

        except Exception as e:
            QMessageBox.critical(self, "Ошибка", f"Ошибка семантической сегментации: {str(e)}")

    def apply_semantic_segmentation(self):
        """Применяет полную семантическую обработку"""
        if self.current_image is None:
            QMessageBox.warning(self, "Предупреждение", "Сначала загрузите изображение")
            return

        self.semantic_tab.segmentation_progress.setVisible(True)
        self.semantic_tab.segmentation_progress.setRange(0, 0)  # Индетерминантный прогресс

        parameters = self.semantic_tab.get_parameters()
        self.developer_tab.log_message(f"Запуск полной семантической обработки: {parameters}", "info")

        # Запускаем в отдельном потоке
        self.start_processing("semantic_full",
                              n_colors=12,  # Базовое значение, будет адаптироваться
                              adaptive_params=parameters['adaptive_params'],
                              face_quality=parameters['face_quality'])

    def reset_to_normal_processing(self):
        """Возвращает к обычной обработке"""
        self.processor.reset_semantic_segmentation()
        if self.current_image is not None:
            self.update_preview(self.current_image)
        self.developer_tab.log_message("Возврат к обычной обработке", "info")
        QMessageBox.information(self, "Информация", "Возврат к обычной обработке")

    def start_processing(self, operation, **kwargs):
        self.progress_bar.setVisible(True)
        self.progress_bar.setRange(0, 0)  # indeterminate progress

        self.thread = ProcessingThread(self.processor, operation, **kwargs)
        self.thread.finished.connect(self.on_processing_finished)
        self.thread.start()

    def on_processing_finished(self, result):
        self.progress_bar.setVisible(False)
        self.semantic_tab.segmentation_progress.setVisible(False)  # Скрываем прогресс семантической сегментации

        if isinstance(result, Exception):
            error_msg = f"Ошибка обработки: {str(result)}"
            self.developer_tab.log_message(error_msg, "error")
            QMessageBox.critical(self, "Ошибка", error_msg)
            return

        # Обрабатываем разные типы результатов
        if result is not None:
            # Если результат - кортеж (например, из quantize), берем первое изображение
            if isinstance(result, tuple):
                image_to_show = result[0]  # Берем первое изображение из кортежа
            else:
                image_to_show = result

            self.update_preview(image_to_show)

            # Особое сообщение для семантической обработки
            if hasattr(self.processor, 'using_semantic_segmentation') and self.processor.using_semantic_segmentation:
                success_msg = f"Семантическая обработка завершена! Обнаружено семантических классов: {len(self.processor.semantic_classes)}"
                self.developer_tab.log_message(success_msg, "success")
                QMessageBox.information(self, "Успех", success_msg)
            else:
                self.developer_tab.log_message("Обработка завершена успешно", "success")

    def update_preview(self, image):
        if image is None:
            return

        try:
            # Если пришел кортеж, берем первый элемент (изображение)
            if isinstance(image, tuple):
                image = image[0]

            # Проверяем, что это numpy array с правильной структурой
            if not hasattr(image, 'shape'):
                print(f"Ошибка: передан объект без атрибута shape: {type(image)}")
                return

            h, w = image.shape[:2]
            ch = image.shape[2] if len(image.shape) > 2 else 1

            # Убедимся, что тип данных правильный
            if image.dtype != np.uint8:
                image = image.astype(np.uint8)

            # Конвертируем numpy array в QPixmap
            bytes_per_line = ch * w

            # Создаем QImage в зависимости от количества каналов
            if ch == 3:  # RGB
                q_img = QImage(image.data, w, h, bytes_per_line, QImage.Format_RGB888)
            elif ch == 4:  # RGBA
                q_img = QImage(image.data, w, h, bytes_per_line, QImage.Format_RGBA8888)
            elif ch == 1:  # Grayscale
                q_img = QImage(image.data, w, h, bytes_per_line, QImage.Format_Grayscale8)
            else:
                print(f"Неизвестный формат изображения: {ch} каналов")
                return

            pixmap = QPixmap.fromImage(q_img)

            # Масштабируем для предпросмотра
            scroll_size = self.scroll_area.size()
            if scroll_size.width() > 0 and scroll_size.height() > 0:
                scaled_pixmap = pixmap.scaled(scroll_size.width() - 20,
                                              scroll_size.height() - 20,
                                              Qt.KeepAspectRatio,
                                              Qt.SmoothTransformation)
                self.preview_label.setPixmap(scaled_pixmap)
            else:
                self.preview_label.setPixmap(pixmap)

        except Exception as e:
            print(f"Ошибка при обновлении предпросмотра: {str(e)}")
            print(f"Тип переданного объекта: {type(image)}")
            if hasattr(image, 'shape'):
                print(f"Форма: {image.shape}")

    def save_image(self, image_type):
        if self.current_image is None:
            QMessageBox.warning(self, "Предупреждение", "Нет изображения для сохранения")
            return

        file_path, _ = QFileDialog.getSaveFileName(
            self, "Сохранить изображение", "",
            "PNG (*.png);;JPEG (*.jpg);;All Files (*)")

        if not file_path:
            return

        try:
            if image_type == "original":
                img_to_save = self.processor.orig
                self.developer_tab.log_message(f"Сохранение исходного изображения: {file_path}", "info")
            elif image_type == "quantized":
                img_to_save = self.processor.quantized
                self.developer_tab.log_message(f"Сохранение квантованного изображения: {file_path}", "info")
            elif image_type == "final":
                # Текущее изображение в preview
                if self.preview_label.pixmap():
                    # Это упрощенная версия - в реальном приложении нужно сохранять актуальный результат
                    img_to_save = self.processor.quantized
                else:
                    img_to_save = self.processor.orig
                self.developer_tab.log_message(f"Сохранение финального результата: {file_path}", "info")
            elif image_type == "scheme":
                borders = self.processor.get_borders(method="cluster_borders")
                img_to_save = self.processor.render_with_borders_and_numbers(
                    borders, show_numbers=True)
                self.developer_tab.log_message(f"Сохранение схемы с контурами: {file_path}", "info")
            else:
                img_to_save = self.processor.orig

            if img_to_save is not None:
                # Конвертируем RGB в BGR для OpenCV
                bgr = cv2.cvtColor(img_to_save, cv2.COLOR_RGB2BGR)
                cv2.imwrite(file_path, bgr)
                success_msg = f"Изображение сохранено: {file_path}"
                self.developer_tab.log_message(success_msg, "success")
                QMessageBox.information(self, "Успех", success_msg)
            else:
                warning_msg = "Нет данных для сохранения"
                self.developer_tab.log_message(warning_msg, "warning")
                QMessageBox.warning(self, "Предупреждение", warning_msg)

        except Exception as e:
            error_msg = f"Ошибка при сохранении: {str(e)}"
            self.developer_tab.log_message(error_msg, "error")
            QMessageBox.critical(self, "Ошибка", error_msg)