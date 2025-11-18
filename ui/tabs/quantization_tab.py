from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QGroupBox, QPushButton,
                             QComboBox, QSpinBox)
from PyQt5.QtCore import QTimer


class QuantizationTab(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.parent = parent
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout(self)

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

    def apply_quantization(self):
        if hasattr(self.parent, 'apply_quantization'):
            self.parent.apply_quantization()