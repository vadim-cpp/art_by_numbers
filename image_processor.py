import cv2
import numpy as np
from sklearn.cluster import KMeans
from scipy.fftpack import dct, idct
from scipy import ndimage
from PIL import Image, ImageFilter


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
        """Метод сглаживания в частотной области"""

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