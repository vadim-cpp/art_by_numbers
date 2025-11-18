from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QGroupBox, QPushButton,
                             QSlider)
from PyQt5.QtCore import Qt


class CorrectionTab(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.parent = parent
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout(self)

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

    def apply_correction(self):
        if hasattr(self.parent, 'apply_correction'):
            self.parent.apply_correction()