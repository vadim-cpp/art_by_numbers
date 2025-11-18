import cv2
import numpy as np
from PIL import Image
import os
from PyQt5.QtGui import QImage


def load_image_pil(file_path):
    """Загрузка изображения через PIL с обработкой ошибок"""
    try:
        pil_image = Image.open(file_path)
        if pil_image.mode != 'RGB':
            pil_image = pil_image.convert('RGB')
        rgb = np.array(pil_image)
        return rgb
    except Exception as e:
        raise ValueError(f"Ошибка загрузки через PIL: {str(e)}")


def numpy_to_qimage(image):
    """Конвертация numpy array в QImage"""
    if image is None:
        return None

    h, w = image.shape[:2]
    ch = image.shape[2] if len(image.shape) > 2 else 1

    if image.dtype != np.uint8:
        image = image.astype(np.uint8)

    bytes_per_line = ch * w

    if ch == 3:  # RGB
        return QImage(image.data, w, h, bytes_per_line, QImage.Format_RGB888)
    elif ch == 4:  # RGBA
        return QImage(image.data, w, h, bytes_per_line, QImage.Format_RGBA8888)
    elif ch == 1:  # Grayscale
        return QImage(image.data, w, h, bytes_per_line, QImage.Format_Grayscale8)

    return None


def validate_image_path(file_path):
    """Проверка существования файла и его формата"""
    if not os.path.exists(file_path):
        return False, "Файл не существует"

    valid_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff')
    if not file_path.lower().endswith(valid_extensions):
        return False, "Неподдерживаемый формат файла"

    return True, "OK"