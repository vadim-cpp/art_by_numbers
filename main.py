import sys
import os
import traceback

import cv2
import numpy as np
from sklearn.cluster import KMeans
from scipy.fftpack import dct, idct
from scipy import ndimage
from PIL import Image, ImageFilter

from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
                             QLabel, QPushButton, QComboBox, QSlider, QSpinBox, QDoubleSpinBox,
                             QTabWidget, QGroupBox, QFileDialog, QMessageBox, QCheckBox,
                             QSplitter, QScrollArea, QSizePolicy, QProgressBar)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QTimer
from PyQt5.QtGui import QPixmap, QImage, QPainter, QFont, QDragEnterEvent, QDropEvent


class ImageProcessor:
    """Универсальный процессор изображений с методами из обоих примеров"""

    def __init__(self, rgb_image: np.ndarray = None):
        if rgb_image is not None:
            self.orig = rgb_image.copy()
            self.work = rgb_image.copy()
        else:
            self.orig = None
            self.work = None
        self.quantized: np.ndarray = None
        self.labels: np.ndarray = None
        self.regions = None
        self.color_palette = None

    def load_image(self, rgb_image: np.ndarray):
        self.orig = rgb_image.copy()
        self.work = rgb_image.copy()
        self.quantized = None
        self.labels = None
        self.regions = None
        self.color_palette = None

    def adjust(self, brightness: int = 0, contrast: float = 1.0, saturation: float = 1.0) -> np.ndarray:
        if self.work is None:
            return None
        img = self.work.astype(np.float32)
        img = img * contrast + brightness
        img = np.clip(img, 0, 255).astype(np.uint8)
        hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV).astype(np.float32)
        hsv[..., 1] *= saturation
        hsv[..., 1] = np.clip(hsv[..., 1], 0, 255)
        img = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)
        self.work = img
        return self.work

    def quantize(self, n_colors: int = 12, blur_k: int = 7, method: str = "kmeans_sklearn",
                 sample_fraction: float = 1.0) -> tuple:
        if self.work is None:
            return None, None

        img = self.work.copy()
        if blur_k and blur_k > 1:
            k = blur_k if blur_k % 2 == 1 else blur_k + 1
            img = cv2.GaussianBlur(img, (k, k), 0)

        h, w = img.shape[:2]
        pixels = img.reshape(-1, 3)
        method = method.lower()

        if method == "kmeans_sklearn":
            data = pixels.astype(np.float32)
            if sample_fraction < 1.0:
                rng = np.random.default_rng(42)
                idx = rng.choice(len(data), size=max(int(len(data) * sample_fraction), n_colors * 10), replace=False)
                sample = data[idx]
            else:
                sample = data

            kmeans = KMeans(n_clusters=max(2, n_colors), random_state=42, n_init=10)
            kmeans.fit(sample)
            centers = kmeans.cluster_centers_.astype(np.uint8)
            labels_flat = kmeans.predict(data).astype(int)
            labels = labels_flat.reshape((h, w))
            out = np.zeros_like(img)
            for i, c in enumerate(centers):
                out[labels == i] = c

            self.quantized = out
            self.labels = labels
            self.color_palette = centers
            return self.quantized, self.labels

        elif method == "median_cut":
            pil = Image.fromarray(img)
            pal = pil.convert("RGB").quantize(colors=max(2, n_colors), method=Image.MEDIANCUT)
            rgb = np.array(pal.convert("RGB"), dtype=np.uint8)
            h2, w2 = rgb.shape[:2]
            if (h2, w2) != (h, w):
                rgb = cv2.resize(rgb, (w, h), interpolation=cv2.INTER_NEAREST)

            flat = rgb.reshape(-1, 3)
            uniq, inv = np.unique(flat, axis=0, return_inverse=True)
            labels = inv.reshape(h, w).astype(int)

            self.quantized = rgb
            self.labels = labels
            self.color_palette = uniq
            return self.quantized, self.labels

        elif method == "frequency_smooth":
            return self._frequency_smooth_quantize(img, n_colors)

    def _frequency_smooth_quantize(self, image: np.ndarray, n_colors: int = 12,
                                   low_freq_ratio: float = 0.5, smoothness: float = 2.0) -> tuple:
        """Метод сглаживания в частотной области из gan_test.py"""

        # Сглаживание перед квантованием
        blur_size = max(1, int(3 * smoothness))
        blur_size = blur_size + 1 if blur_size % 2 == 0 else blur_size
        blur_size = min(blur_size, 15)

        smoothed = cv2.GaussianBlur(image, (blur_size, blur_size), smoothness)

        # Квантование цветов
        pixels = smoothed.reshape(-1, 3)
        kmeans = KMeans(n_clusters=n_colors, random_state=42, n_init=10)
        labels = kmeans.fit_predict(pixels)
        centers = kmeans.cluster_centers_.astype(np.uint8)

        # Базовое квантованное изображение
        quantized_basic = centers[labels].reshape(image.shape)

        # Частотное смешивание
        if image.dtype != np.uint8:
            image = image.astype(np.uint8)

        # DCT преобразование
        ycrcb_orig = cv2.cvtColor(image, cv2.COLOR_RGB2YCrCb)
        ycrcb_quant = cv2.cvtColor(quantized_basic, cv2.COLOR_RGB2YCrCb)

        # Применяем DCT к каждому каналу и смешиваем
        blended = np.zeros_like(ycrcb_orig, dtype=np.float64)
        for i in range(3):
            orig_dct = dct(dct(ycrcb_orig[:, :, i].astype(np.float64), axis=0, norm='ortho'), axis=1, norm='ortho')
            quant_dct = dct(dct(ycrcb_quant[:, :, i].astype(np.float64), axis=0, norm='ortho'), axis=1, norm='ortho')

            # Создаем маску для смешивания
            h, w = orig_dct.shape
            y, x = np.ogrid[:h, :w]
            center_y, center_x = h // 2, w // 2
            sigma = min(h, w) * low_freq_ratio * 0.8
            mask = np.exp(-((x - center_x) ** 2 + (y - center_y) ** 2) / (2 * sigma ** 2))

            # Смешивание
            blended_dct = orig_dct * mask + quant_dct * (1 - mask)

            # Обратное DCT
            blended[:, :, i] = idct(idct(blended_dct, axis=0, norm='ortho'), axis=1, norm='ortho')

        # Конвертируем обратно в RGB
        blended = np.clip(blended, 0, 255).astype(np.uint8)
        final = cv2.cvtColor(blended, cv2.COLOR_YCrCb2RGB)
        final = cv2.bilateralFilter(final, 9, 75, 75)

        self.quantized = final
        self.labels = labels.reshape(image.shape[:2])
        self.color_palette = centers

        return self.quantized, self.labels

    def merge_small_regions(self, min_area_px: int = 100) -> tuple:
        if self.labels is None:
            return self.quantized, self.labels

        labels = self.labels.copy()
        h, w = labels.shape

        # Упрощенная версия объединения регионов
        unique_labels = np.unique(labels)
        for lab in unique_labels:
            mask = (labels == lab).astype(np.uint8)
            if np.sum(mask) < min_area_px:
                # Находим соседний регион с наибольшей границей
                kernel = np.ones((3, 3), np.uint8)
                dilated = cv2.dilate(mask, kernel, iterations=1)
                border = dilated - mask

                neighbor_labels = []
                for y, x in zip(*np.where(border > 0)):
                    if 0 <= y < h and 0 <= x < w:
                        neighbor_labels.append(labels[y, x])

                if neighbor_labels:
                    new_label = max(set(neighbor_labels), key=neighbor_labels.count)
                    labels[labels == lab] = new_label

        self.labels = labels

        # Обновляем квантованное изображение
        if self.color_palette is not None:
            out = np.zeros_like(self.work)
            for i, color in enumerate(self.color_palette):
                out[self.labels == i] = color
            self.quantized = out

        return self.quantized, self.labels

    def get_borders(self, method: str = "cluster_borders", **kwargs) -> np.ndarray:
        if self.labels is None:
            return np.zeros(self.work.shape[:2], dtype=np.uint8)

        if method == "cluster_borders":
            h, w = self.labels.shape
            borders = np.zeros((h, w), dtype=np.uint8)
            v = np.zeros_like(borders, dtype=bool)
            v[:-1, :] = self.labels[:-1, :] != self.labels[1:, :]
            borders[v] = 255
            h_mask = np.zeros_like(borders, dtype=bool)
            h_mask[:, :-1] = self.labels[:, :-1] != self.labels[:, 1:]
            borders[h_mask] = 255
            return borders

        elif method == "canny":
            src = self.quantized if self.quantized is not None else self.work
            gray = cv2.cvtColor(src, cv2.COLOR_RGB2GRAY)
            low_thr = kwargs.get('low_thr', 100)
            high_thr = kwargs.get('high_thr', 200)
            edges = cv2.Canny(gray, low_thr, high_thr)
            return edges

    def get_regions_with_centroids(self, min_area_px: int = 1) -> list:
        if self.labels is None:
            return []

        regions = []
        unique = np.unique(self.labels)
        for lab in unique:
            mask = (self.labels == lab).astype(np.uint8) * 255
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for c in contours:
                area = cv2.contourArea(c)
                if area < min_area_px:
                    continue
                M = cv2.moments(c)
                if M["m00"] == 0:
                    continue
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                regions.append({
                    "label": int(lab),
                    "contour": c,
                    "centroid": (cx, cy),
                    "area": int(area)
                })
        self.regions = regions
        return regions

    def render_with_borders_and_numbers(self, border_mask: np.ndarray,
                                        border_thickness: int = 2,
                                        show_numbers: bool = True,
                                        number_color: tuple = (0, 0, 0)) -> np.ndarray:
        base = self.quantized.copy() if self.quantized is not None else self.work.copy()

        if border_thickness > 1:
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (border_thickness, border_thickness))
            border_mask = cv2.dilate(border_mask, kernel, iterations=1)

        base_out = base.copy()
        base_out[border_mask > 0] = (0, 0, 0)

        if show_numbers and self.color_palette is not None:
            regs = self.get_regions_with_centroids(min_area_px=1)
            for idx, r in enumerate(regs, start=1):
                cx, cy = r['centroid']
                cv2.putText(base_out, str(idx), (cx, cy),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, number_color, 1, cv2.LINE_AA)

        return base_out


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


class ProcessingThread(QThread):
    """Поток для обработки изображения"""

    finished = pyqtSignal(object)
    progress = pyqtSignal(int)

    def __init__(self, processor, operation, **kwargs):
        super().__init__()
        self.processor = processor
        self.operation = operation
        self.kwargs = kwargs

    def run(self):
        try:
            if self.operation == "quantize":
                result = self.processor.quantize(**self.kwargs)
            elif self.operation == "adjust":
                result = self.processor.adjust(**self.kwargs)
            elif self.operation == "merge_regions":
                result = self.processor.merge_small_regions(**self.kwargs)
            elif self.operation == "render":
                result = self.processor.render_with_borders_and_numbers(**self.kwargs)
            else:
                result = None

            self.finished.emit(result)
        except Exception as e:
            self.finished.emit(e)


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

        # Виджет drag-and-drop
        self.drop_label = DragDropLabel(self)
        self.drop_label.clicked.connect(self.browse_image)  # Подключаем сигнал
        left_layout.addWidget(self.drop_label)

        # Вкладки настроек
        self.tabs = QTabWidget()

        # Вкладка коррекции изображения
        self.setup_correction_tab()
        # Вкладка квантования
        self.setup_quantization_tab()
        # Вкладка контуров
        self.setup_contours_tab()
        # Вкладка сохранения
        self.setup_export_tab()

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

    def setup_correction_tab(self):
        tab = QWidget()
        layout = QVBoxLayout(tab)

        # Яркость
        brightness_group = QGroupBox("Яркость")
        brightness_layout = QVBoxLayout(brightness_group)
        self.brightness_slider = QSlider(Qt.Horizontal)
        self.brightness_slider.setRange(-150, 150)
        self.brightness_slider.setValue(0)
        brightness_layout.addWidget(self.brightness_slider)

        # Контраст
        contrast_group = QGroupBox("Контраст (%)")
        contrast_layout = QVBoxLayout(contrast_group)
        self.contrast_slider = QSlider(Qt.Horizontal)
        self.contrast_slider.setRange(50, 200)
        self.contrast_slider.setValue(100)
        contrast_layout.addWidget(self.contrast_slider)

        # Насыщенность
        saturation_group = QGroupBox("Насыщенность (%)")
        saturation_layout = QVBoxLayout(saturation_group)
        self.saturation_slider = QSlider(Qt.Horizontal)
        self.saturation_slider.setRange(50, 200)
        self.saturation_slider.setValue(100)
        saturation_layout.addWidget(self.saturation_slider)

        # Кнопка применения
        apply_btn = QPushButton("Применить коррекцию")
        apply_btn.clicked.connect(self.apply_correction)

        layout.addWidget(brightness_group)
        layout.addWidget(contrast_group)
        layout.addWidget(saturation_group)
        layout.addWidget(apply_btn)
        layout.addStretch()

        self.tabs.addTab(tab, "Коррекция")

    def setup_quantization_tab(self):
        tab = QWidget()
        layout = QVBoxLayout(tab)

        # Метод квантования
        method_group = QGroupBox("Метод квантования")
        method_layout = QVBoxLayout(method_group)
        self.method_combo = QComboBox()
        self.method_combo.addItems([
            "KMeans (sklearn)",
            "Median Cut",
            "Frequency Smooth"
        ])
        method_layout.addWidget(self.method_combo)

        # Количество цветов
        colors_group = QGroupBox("Количество цветов")
        colors_layout = QVBoxLayout(colors_group)
        self.colors_spin = QSpinBox()
        self.colors_spin.setRange(2, 40)
        self.colors_spin.setValue(12)
        colors_layout.addWidget(self.colors_spin)

        # Размытие
        blur_group = QGroupBox("Размытие (kernel)")
        blur_layout = QVBoxLayout(blur_group)
        self.blur_spin = QSpinBox()
        self.blur_spin.setRange(1, 21)
        self.blur_spin.setValue(7)
        blur_layout.addWidget(self.blur_spin)

        # Минимальная площадь
        area_group = QGroupBox("Мин. площадь области")
        area_layout = QVBoxLayout(area_group)
        self.area_spin = QSpinBox()
        self.area_spin.setRange(1, 2000)
        self.area_spin.setValue(150)
        area_layout.addWidget(self.area_spin)

        # Кнопка применения
        quantize_btn = QPushButton("Применить квантование")
        quantize_btn.clicked.connect(self.apply_quantization)

        layout.addWidget(method_group)
        layout.addWidget(colors_group)
        layout.addWidget(blur_group)
        layout.addWidget(area_group)
        layout.addWidget(quantize_btn)
        layout.addStretch()

        self.tabs.addTab(tab, "Квантование")

    def setup_contours_tab(self):
        tab = QWidget()
        layout = QVBoxLayout(tab)

        # Метод контуров
        method_group = QGroupBox("Метод контуров")
        method_layout = QVBoxLayout(method_group)
        self.contour_method_combo = QComboBox()
        self.contour_method_combo.addItems([
            "Cluster Borders",
            "Canny Edges"
        ])
        method_layout.addWidget(self.contour_method_combo)

        # Толщина контура
        thickness_group = QGroupBox("Толщина контура")
        thickness_layout = QVBoxLayout(thickness_group)
        self.thickness_spin = QSpinBox()
        self.thickness_spin.setRange(1, 10)
        self.thickness_spin.setValue(2)
        thickness_layout.addWidget(self.thickness_spin)

        # Показывать номера
        self.show_numbers_check = QCheckBox("Показывать номера областей")
        self.show_numbers_check.setChecked(True)

        # Кнопка применения
        contours_btn = QPushButton("Применить контуры")
        contours_btn.clicked.connect(self.apply_contours)

        layout.addWidget(method_group)
        layout.addWidget(thickness_group)
        layout.addWidget(self.show_numbers_check)
        layout.addWidget(contours_btn)
        layout.addStretch()

        self.tabs.addTab(tab, "Контуры")

    def setup_export_tab(self):
        tab = QWidget()
        layout = QVBoxLayout(tab)

        # Кнопки сохранения
        save_original_btn = QPushButton("Сохранить исходное изображение")
        save_original_btn.clicked.connect(lambda: self.save_image("original"))

        save_quantized_btn = QPushButton("Сохранить квантованное изображение")
        save_quantized_btn.clicked.connect(lambda: self.save_image("quantized"))

        save_final_btn = QPushButton("Сохранить финальный результат")
        save_final_btn.clicked.connect(lambda: self.save_image("final"))

        save_scheme_btn = QPushButton("Сохранить схему (контуры + номера)")
        save_scheme_btn.clicked.connect(lambda: self.save_image("scheme"))

        layout.addWidget(save_original_btn)
        layout.addWidget(save_quantized_btn)
        layout.addWidget(save_final_btn)
        layout.addWidget(save_scheme_btn)
        layout.addStretch()

        self.tabs.addTab(tab, "Сохранение")

    def browse_image(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Выберите изображение", "",
            "Images (*.png *.jpg *.jpeg *.bmp *.tiff)")
        if file_path:
            self.load_image(file_path)

    def load_image(self, file_path):
        try:
            print(f"Пытаемся загрузить: {file_path}")

            # Проверяем существование файла
            if not os.path.exists(file_path):
                QMessageBox.critical(self, "Ошибка", f"Файл не существует: {file_path}")
                return

            # Загружаем через PIL (более надежно с путями и форматами)
            try:
                from PIL import Image
                pil_image = Image.open(file_path)

                # Конвертируем в RGB если нужно (на случай PNG с альфа-каналом и т.д.)
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

        except Exception as e:
            QMessageBox.critical(self, "Ошибка", f"Ошибка при загрузке: {str(e)}")
            print(f"Полная ошибка: {traceback.format_exc()}")

    def apply_correction(self):
        if self.current_image is None:
            QMessageBox.warning(self, "Предупреждение", "Сначала загрузите изображение")
            return

        brightness = self.brightness_slider.value()
        contrast = self.contrast_slider.value() / 100.0
        saturation = self.saturation_slider.value() / 100.0

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

        method = method_map.get(self.method_combo.currentText(), "kmeans_sklearn")
        n_colors = self.colors_spin.value()
        blur_k = self.blur_spin.value()
        min_area = self.area_spin.value()

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

        QTimer.singleShot(100, apply_merge)

    def apply_contours(self):
        if self.processor.labels is None:
            QMessageBox.warning(self, "Предупреждение", "Сначала примените квантование")
            return

        method_map = {
            "Cluster Borders": "cluster_borders",
            "Canny Edges": "canny"
        }

        method = method_map.get(self.contour_method_combo.currentText(), "cluster_borders")
        borders = self.processor.get_borders(method=method)

        result = self.processor.render_with_borders_and_numbers(
            border_mask=borders,
            border_thickness=self.thickness_spin.value(),
            show_numbers=self.show_numbers_check.isChecked()
        )

        self.update_preview(result)

    def start_processing(self, operation, **kwargs):
        self.progress_bar.setVisible(True)
        self.progress_bar.setRange(0, 0)  # indeterminate progress

        self.thread = ProcessingThread(self.processor, operation, **kwargs)
        self.thread.finished.connect(self.on_processing_finished)
        self.thread.start()

    def on_processing_finished(self, result):
        self.progress_bar.setVisible(False)

        if isinstance(result, Exception):
            QMessageBox.critical(self, "Ошибка", f"Ошибка обработки: {str(result)}")
            return

        # Обрабатываем разные типы результатов
        if result is not None:
            # Если результат - кортеж (например, из quantize), берем первое изображение
            if isinstance(result, tuple):
                image_to_show = result[0]  # Берем первое изображение из кортежа
            else:
                image_to_show = result

            self.update_preview(image_to_show)

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
            elif image_type == "quantized":
                img_to_save = self.processor.quantized
            elif image_type == "final":
                # Текущее изображение в preview
                if self.preview_label.pixmap():
                    # Это упрощенная версия - в реальном приложении нужно сохранять актуальный результат
                    img_to_save = self.processor.quantized
                else:
                    img_to_save = self.processor.orig
            elif image_type == "scheme":
                borders = self.processor.get_borders(method="cluster_borders")
                img_to_save = self.processor.render_with_borders_and_numbers(
                    borders, show_numbers=True)
            else:
                img_to_save = self.processor.orig

            if img_to_save is not None:
                # Конвертируем RGB в BGR для OpenCV
                bgr = cv2.cvtColor(img_to_save, cv2.COLOR_RGB2BGR)
                cv2.imwrite(file_path, bgr)
                QMessageBox.information(self, "Успех", f"Изображение сохранено: {file_path}")
            else:
                QMessageBox.warning(self, "Предупреждение", "Нет данных для сохранения")

        except Exception as e:
            QMessageBox.critical(self, "Ошибка", f"Ошибка при сохранении: {str(e)}")


if __name__ == '__main__':
    app = QApplication(sys.argv)
    app.setStyle('Fusion')

    window = PaintByNumbersApp()
    window.show()

    sys.exit(app.exec_())