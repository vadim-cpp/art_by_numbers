import cv2
import numpy as np
from sklearn.cluster import KMeans
from scipy.fftpack import dct, idct
from scipy import ndimage
from PIL import Image, ImageFilter
import torch
import torchvision.transforms as transforms
from torchvision.models.segmentation import deeplabv3_resnet50, DeepLabV3_ResNet50_Weights
import warnings
import time
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score

warnings.filterwarnings('ignore')


class ImageProcessor:
    """Универсальный процессор изображений с семантической сегментацией и метриками"""

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

        self.semantic_mask = None
        self.semantic_classes = None
        self.using_semantic_segmentation = False
        self.segmentation_model = None

        # Для метрик
        self.metrics_callback = None

    def set_metrics_callback(self, callback):
        """Устанавливает callback для отправки метрик"""
        self.metrics_callback = callback

    def _log_metric(self, message, message_type="metric"):
        """Отправляет метрику через callback"""
        if self.metrics_callback:
            self.metrics_callback(message, message_type)

    def load_image(self, rgb_image: np.ndarray):
        self.orig = rgb_image.copy()
        self.work = rgb_image.copy()
        self.quantized = None
        self.labels = None
        self.regions = None
        self.color_palette = None
        # Сбрасываем семантическую сегментацию при загрузке нового изображения
        self.semantic_mask = None
        self.semantic_classes = None
        self.using_semantic_segmentation = False

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
                 sample_fraction: float = 1.0, calculate_metrics: bool = True) -> tuple:
        start_time = time.time()

        if self.work is None:
            return None, None

        img = self.work.copy()
        if blur_k and blur_k > 1:
            k = blur_k if blur_k % 2 == 1 else blur_k + 1
            img = cv2.GaussianBlur(img, (k, k), 0)

        h, w = img.shape[:2]
        pixels = img.reshape(-1, 3)
        method = method.lower()

        self._log_metric(f"Начало квантования: {method}", "info")
        self._log_metric(f"Параметры: цвета={n_colors}, размытие={blur_k}, выборка={sample_fraction}")

        if method == "kmeans_sklearn":
            data = pixels.astype(np.float32)
            if sample_fraction < 1.0:
                rng = np.random.default_rng(42)
                idx = rng.choice(len(data), size=max(int(len(data) * sample_fraction), n_colors * 10), replace=False)
                sample = data[idx]
                self._log_metric(f"Используется подвыборка: {len(sample)} пикселей")
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

            # Вычисляем метрики
            if calculate_metrics:
                metrics = self._calculate_clustering_metrics(data, labels_flat, centers, "KMeans")
                execution_time = time.time() - start_time
                self._log_clustering_metrics(metrics, execution_time)

            return self.quantized, self.labels

        elif method == "median_cut":
            self._log_metric("Метод: Median Cut (PIL)", "info")
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

            # Для Median Cut вычисляем приблизительные метрики
            if calculate_metrics:
                data = pixels.astype(np.float32)
                centers = uniq.astype(np.float32)
                metrics = self._calculate_clustering_metrics(data, labels.ravel(), centers, "Median Cut")
                execution_time = time.time() - start_time
                self._log_clustering_metrics(metrics, execution_time)

            return self.quantized, self.labels

        elif method == "frequency_smooth":
            self._log_metric("Метод: Frequency Smooth", "info")
            result = self._frequency_smooth_quantize(img, n_colors, calculate_metrics=calculate_metrics)
            execution_time = time.time() - start_time
            self._log_metric(f"Frequency Smooth завершен за {execution_time:.2f} сек", "success")
            return result

    def _frequency_smooth_quantize(self, image: np.ndarray, n_colors: int = 12,
                                   low_freq_ratio: float = 0.5, smoothness: float = 2.0,
                                   calculate_metrics: bool = True) -> tuple:
        """Метод сглаживания в частотной области"""

        start_time = time.time()

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

        # После квантования вычисляем метрики
        if calculate_metrics:
            pixels = image.reshape(-1, 3).astype(np.float32)
            labels_flat = self.labels.ravel()
            metrics = self._calculate_clustering_metrics(pixels, labels_flat, self.color_palette, "Frequency Smooth")
            execution_time = time.time() - start_time
            self._log_clustering_metrics(metrics, execution_time)

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

    def load_segmentation_model(self, model_name="DeepLabV3"):
        """Загружает модель семантической сегментации"""
        try:
            if self.segmentation_model is not None:
                return True

            from torchvision.models.segmentation import deeplabv3_resnet50

            # Универсальный способ загрузки для разных версий torchvision
            try:
                # Для torchvision >= 0.13
                from torchvision.models.segmentation import DeepLabV3_ResNet50_Weights
                weights = DeepLabV3_ResNet50_Weights.DEFAULT
                self.segmentation_model = deeplabv3_resnet50(weights=weights)
                print(f"Модель загружена с весами: {weights}")

            except (AttributeError, ImportError):
                try:
                    # Для более старых версий
                    self.segmentation_model = deeplabv3_resnet50(pretrained=True)
                    print("Модель загружена с pretrained=True")
                except:
                    # Резервный вариант
                    self.segmentation_model = deeplabv3_resnet50(num_classes=21)
                    print("Модель создана без предобученных весов")

            self.segmentation_model.eval()
            print("Модель семантической сегментации успешно загружена")
            return True

        except Exception as e:
            print(f"Ошибка загрузки модели сегментации: {e}")
            # Создаем простую модель-заглушку для продолжения работы
            self._create_dummy_segmentation_model()
            return False

    def _create_dummy_segmentation_model(self):
        """Создает простую модель-заглушку для продолжения работы"""

        class DummySegmentationModel:
            def __init__(self):
                self.eval_mode = True

            def eval(self):
                self.eval_mode = True

            def __call__(self, x):
                # Возвращает фиктивные сегменты (все пиксели принадлежат классу 0 - фон)
                import torch
                batch_size, _, h, w = x.shape
                return {'out': torch.zeros(batch_size, 21, h, w)}

        self.segmentation_model = DummySegmentationModel()
        print("Создана модель-заглушка для семантической сегментации")

    def semantic_segment(self, image: np.ndarray, max_classes: int = 15) -> np.ndarray:
        """Применяет семантическую сегментацию к изображению"""
        if not self.load_segmentation_model():
            # Если модель не загрузилась, возвращаем единую маску
            self.semantic_mask = np.zeros(image.shape[:2], dtype=np.uint8)
            self.semantic_classes = np.array([0])
            self.using_semantic_segmentation = False
            print("Используется резервный режим сегментации")
            return self.semantic_mask

        try:
            # Преобразуем изображение для модели
            transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((520, 520)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])

            input_tensor = transform(image).unsqueeze(0)

            with torch.no_grad():
                output = self.segmentation_model(input_tensor)
                # Обрабатываем разные форматы вывода
                if isinstance(output, dict) and 'out' in output:
                    predictions = output['out'][0].argmax(0).byte().cpu().numpy()
                else:
                    predictions = output[0].argmax(0).byte().cpu().numpy()

            # Масштабируем обратно к исходному размеру
            h, w = image.shape[:2]
            semantic_mask = cv2.resize(predictions, (w, h), interpolation=cv2.INTER_NEAREST)

            # Ограничиваем количество классов
            unique_classes = np.unique(semantic_mask)
            if len(unique_classes) > max_classes:
                # Оставляем только самые частые классы
                counts = [(cls, np.sum(semantic_mask == cls)) for cls in unique_classes]
                counts.sort(key=lambda x: x[1], reverse=True)
                top_classes = [cls for cls, count in counts[:max_classes]]

                # Все остальные классы объединяем в фон
                new_mask = np.zeros_like(semantic_mask)
                for i, cls in enumerate(top_classes):
                    new_mask[semantic_mask == cls] = i + 1
                semantic_mask = new_mask

            self.semantic_mask = semantic_mask
            self.semantic_classes = np.unique(semantic_mask)
            self.using_semantic_segmentation = True

            print(f"Семантическая сегментация завершена. Классы: {len(self.semantic_classes)}")
            return semantic_mask

        except Exception as e:
            print(f"Ошибка семантической сегментации: {e}")
            # Резервный вариант - разбиваем изображение на сетку
            self._fallback_segmentation(image)
            return self.semantic_mask

    def _fallback_segmentation(self, image: np.ndarray):
        """Резервный метод сегментации - разбивает изображение на сетку"""
        h, w = image.shape[:2]
        grid_size = 8
        self.semantic_mask = np.zeros((h, w), dtype=np.uint8)

        for i in range(grid_size):
            for j in range(grid_size):
                y_start = i * h // grid_size
                y_end = (i + 1) * h // grid_size
                x_start = j * w // grid_size
                x_end = (j + 1) * w // grid_size
                self.semantic_mask[y_start:y_end, x_start:x_end] = i * grid_size + j

        self.semantic_classes = np.unique(self.semantic_mask)
        self.using_semantic_segmentation = True
        print("Использована резервная сегментация (сетка)")

    def semantic_aware_quantize(self, image: np.ndarray, n_colors: int = 12,
                                adaptive_params: bool = True, face_quality: bool = True) -> tuple:
        """Квантование с учетом семантических областей"""
        start_time = time.time()

        if self.semantic_mask is None:
            return self.quantize(image, n_colors=n_colors)

        h, w = image.shape[:2]
        result = np.zeros_like(image)
        labels = np.zeros((h, w), dtype=int)

        current_label = 0
        color_palette = []

        self._log_metric("Начало семантического квантования", "info")
        self._log_metric(f"Семантических классов: {len(self.semantic_classes)}")

        for class_id in self.semantic_classes:
            if class_id == 0:  # Фон
                continue

            mask = self.semantic_mask == class_id
            if np.sum(mask) < 100:  # Пропускаем очень маленькие области
                continue

            # Адаптивные параметры для разных типов областей
            if adaptive_params:
                class_n_colors = self._get_optimal_colors_for_class(class_id, n_colors, face_quality)
                class_blur_k = self._get_optimal_blur_for_class(class_id)
            else:
                class_n_colors = n_colors
                class_blur_k = 3

            # Извлекаем область изображения
            region_pixels = image[mask]

            if len(region_pixels) > class_n_colors * 10:  # Достаточно пикселей для кластеризации
                # Квантуем эту область отдельно
                kmeans = KMeans(n_clusters=class_n_colors, random_state=42, n_init=5)
                region_labels = kmeans.fit_predict(region_pixels)
                region_colors = kmeans.cluster_centers_.astype(np.uint8)

                # Записываем результаты
                for i, color in enumerate(region_colors):
                    result[mask][region_labels == i] = color
                    labels[mask][region_labels == i] = current_label + i

                current_label += class_n_colors
                color_palette.extend(region_colors)
            else:
                # Для маленьких областей используем средний цвет
                mean_color = np.mean(region_pixels, axis=0).astype(np.uint8)
                result[mask] = mean_color
                labels[mask] = current_label
                current_label += 1
                color_palette.append(mean_color)

        self.quantized = result
        self.labels = labels
        self.color_palette = np.array(color_palette)

        # Вычисляем общие метрики для всего изображения
        if self.labels is not None:
            pixels = image.reshape(-1, 3).astype(np.float32)
            labels_flat = self.labels.ravel()
            metrics = self._calculate_clustering_metrics(pixels, labels_flat, self.color_palette, "Semantic Aware")
            execution_time = time.time() - start_time
            self._log_clustering_metrics(metrics, execution_time)

            # Дополнительно: метрики по семантическим классам
            self._log_semantic_metrics(image)

        return result, labels

    def _get_optimal_colors_for_class(self, class_id: int, base_colors: int, face_quality: bool) -> int:
        """Определяет оптимальное количество цветов для семантического класса"""
        # Эвристики для разных типов областей
        if class_id in [12, 13, 14, 15]:  # Небо, вода
            return max(3, base_colors // 2)
        elif class_id in [1, 2, 3]:  # Люди, лица
            return base_colors * 2 if face_quality else base_colors
        elif class_id in [6, 7, 8]:  # Транспорт
            return base_colors
        elif class_id in [4, 5, 9, 10]:  # Здания, сооружения
            return base_colors
        else:  # Остальное
            return max(2, base_colors // 3)

    def _get_optimal_blur_for_class(self, class_id: int) -> int:
        """Определяет оптимальное размытие для семантического класса"""
        if class_id in [1, 2, 3]:  # Люди, лица - минимальное размытие
            return 1
        elif class_id in [12, 13]:  # Небо, вода - сильное размытие
            return 7
        else:  # Остальное - среднее размытие
            return 3

    def visualize_semantic_mask(self, image: np.ndarray) -> np.ndarray:
        """Визуализирует семантическую маску поверх изображения"""
        if self.semantic_mask is None:
            return image

        # Создаем цветовую карту для визуализации
        h, w = self.semantic_mask.shape
        colored_mask = np.zeros((h, w, 3), dtype=np.uint8)

        # Генерируем цвета для каждого класса
        for class_id in self.semantic_classes:
            # Генерируем уникальный цвет для каждого класса
            color = self._generate_color(class_id)
            colored_mask[self.semantic_mask == class_id] = color

        # Наложение маски на изображение с прозрачностью
        alpha = 0.6
        result = cv2.addWeighted(image, 1 - alpha, colored_mask, alpha, 0)

        return result

    def _generate_color(self, class_id: int) -> tuple:
        """Генерирует уникальный цвет для класса"""
        # Простая хэш-функция для генерации цвета
        r = (class_id * 67) % 256
        g = (class_id * 137) % 256
        b = (class_id * 199) % 256
        return (r, g, b)

    def reset_semantic_segmentation(self):
        """Сбрасывает семантическую сегментацию"""
        self.semantic_mask = None
        self.semantic_classes = None
        self.using_semantic_segmentation = False

    def _calculate_clustering_metrics(self, pixels, labels, centers, method_name):
        """Вычисляет метрики качества кластеризации"""
        try:
            # Берем подвыборку для вычисления метрик (для производительности)
            if len(pixels) > 10000:
                rng = np.random.default_rng(42)
                indices = rng.choice(len(pixels), size=10000, replace=False)
                sample_pixels = pixels[indices]
                sample_labels = labels[indices]
            else:
                sample_pixels = pixels
                sample_labels = labels

            metrics = {}

            # Silhouette Score (-1 до 1, чем выше тем лучше)
            if len(np.unique(sample_labels)) > 1:
                try:
                    silhouette_avg = silhouette_score(sample_pixels, sample_labels)
                    metrics['silhouette'] = silhouette_avg
                except:
                    metrics['silhouette'] = None

            # Calinski-Harabasz Index (чем выше тем лучше)
            try:
                calinski_harabasz = calinski_harabasz_score(sample_pixels, sample_labels)
                metrics['calinski_harabasz'] = calinski_harabasz
            except:
                metrics['calinski_harabasz'] = None

            # Davies-Bouldin Index (чем ниже тем лучше)
            try:
                davies_bouldin = davies_bouldin_score(sample_pixels, sample_labels)
                metrics['davies_bouldin'] = davies_bouldin
            except:
                metrics['davies_bouldin'] = None

            # Inertia (within-cluster sum of squares)
            try:
                inertia = 0
                for i, center in enumerate(centers):
                    cluster_points = sample_pixels[sample_labels == i]
                    if len(cluster_points) > 0:
                        inertia += np.sum((cluster_points - center) ** 2)
                metrics['inertia'] = inertia
            except:
                metrics['inertia'] = None

            # Дополнительные метрики
            metrics['n_clusters'] = len(centers)
            metrics['n_samples'] = len(pixels)
            metrics['method'] = method_name

            return metrics

        except Exception as e:
            self._log_metric(f"Ошибка вычисления метрик: {str(e)}", "error")
            return None

    def _log_clustering_metrics(self, metrics, execution_time):
        """Логирует метрики кластеризации"""
        if not metrics:
            return

        self._log_metric("=" * 60, "header")
        self._log_metric(f"МЕТРИКИ КЛАСТЕРИЗАЦИИ - {metrics['method']}", "header")
        self._log_metric("=" * 60, "header")

        self._log_metric(f"Время выполнения: {execution_time:.2f} сек")
        self._log_metric(f"Количество кластеров: {metrics['n_clusters']}")
        self._log_metric(f"Количество пикселей: {metrics['n_samples']:,}")

        self._log_metric("--- Качество кластеризации ---", "info")

        if metrics['silhouette'] is not None:
            silhouette_quality = "Отличное" if metrics['silhouette'] > 0.7 else \
                "Хорошее" if metrics['silhouette'] > 0.5 else \
                    "Среднее" if metrics['silhouette'] > 0.25 else "Плохое"
            self._log_metric(f"Silhouette Score: {metrics['silhouette']:.4f} ({silhouette_quality})")

        if metrics['calinski_harabasz'] is not None:
            self._log_metric(f"Calinski-Harabasz: {metrics['calinski_harabasz']:,.2f}")

        if metrics['davies_bouldin'] is not None:
            db_quality = "Отличное" if metrics['davies_bouldin'] < 0.5 else \
                "Хорошее" if metrics['davies_bouldin'] < 1.0 else \
                    "Среднее" if metrics['davies_bouldin'] < 2.0 else "Плохое"
            self._log_metric(f"Davies-Bouldin: {metrics['davies_bouldin']:.4f} ({db_quality})")

        if metrics['inertia'] is not None:
            self._log_metric(f"Inertia (WCSS): {metrics['inertia']:,.2f}")

        self._log_metric("=" * 60, "header")

    def _log_semantic_metrics(self, image):
        """Логирует метрики по семантическим классам"""
        if not self.semantic_mask or not self.labels:
            return

        self._log_metric("--- Метрики по семантическим классам ---", "info")

        for class_id in self.semantic_classes:
            if class_id == 0:  # Пропускаем фон
                continue

            mask = self.semantic_mask == class_id
            class_pixels = image[mask]
            class_labels = self.labels[mask]

            if len(class_pixels) > 100:  # Только для значительных областей
                unique_colors = len(np.unique(class_labels))
                area_pixels = len(class_pixels)
                area_percent = (area_pixels / (image.shape[0] * image.shape[1])) * 100

                self._log_metric(f"Класс {class_id}: {area_pixels} пикс. ({area_percent:.1f}%), "
                                 f"цветов: {unique_colors}")