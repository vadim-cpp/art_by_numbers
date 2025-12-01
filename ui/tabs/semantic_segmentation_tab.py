from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QGroupBox, QPushButton,
                             QComboBox, QSpinBox, QCheckBox, QLabel, QProgressBar)
from PyQt5.QtCore import Qt


class SemanticSegmentationTab(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.parent = parent
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout(self)

        # Информационная группа
        info_group = QGroupBox("Умная семантическая сегментация")
        info_layout = QVBoxLayout(info_group)
        info_label = QLabel(
            "Использует нейросети для интеллектуального разделения изображения\n"
            "на смысловые области (лица, небо, вода, здания и т.д.)\n"
            "Разные области обрабатываются с разными параметрами для лучшего качества."
        )
        info_label.setWordWrap(True)
        info_label.setStyleSheet("color: #666; font-size: 12px;")
        info_layout.addWidget(info_label)

        # Модель сегментации
        model_group = QGroupBox("Модель сегментации")
        model_layout = QVBoxLayout(model_group)
        self.model_combo = QComboBox()
        self.model_combo.addItems([
            "SegFormer (универсальная)",
            "DeepLabV3 (точность)",
            "UPerNet (детализация)"
        ])
        model_layout.addWidget(self.model_combo)

        # Параметры сегментации
        params_group = QGroupBox("Параметры сегментации")
        params_layout = QVBoxLayout(params_group)

        # Количество семантических классов
        classes_layout = QVBoxLayout()
        classes_label = QLabel("Максимальное количество семантических классов:")
        self.classes_spin = QSpinBox()
        self.classes_spin.setRange(5, 50)
        self.classes_spin.setValue(15)
        classes_layout.addWidget(classes_label)
        classes_layout.addWidget(self.classes_spin)
        params_layout.addLayout(classes_layout)

        # Разные параметры для разных областей
        self.adaptive_params_check = QCheckBox("Адаптивные параметры для разных областей")
        self.adaptive_params_check.setChecked(True)
        params_layout.addWidget(self.adaptive_params_check)

        # Особое качество для лиц
        self.face_quality_check = QCheckBox("Особое качество для лиц и людей")
        self.face_quality_check.setChecked(True)
        params_layout.addWidget(self.face_quality_check)

        # Прогресс-бар
        self.segmentation_progress = QProgressBar()
        self.segmentation_progress.setVisible(False)
        params_layout.addWidget(self.segmentation_progress)

        # Кнопки
        buttons_layout = QVBoxLayout()

        # Предпросмотр сегментации
        preview_btn = QPushButton("Предпросмотр сегментации")
        preview_btn.clicked.connect(self.preview_segmentation)

        # # Применить семантическую сегментацию
        # apply_btn = QPushButton("Применить семантическую сегментацию")
        # apply_btn.setStyleSheet("QPushButton { background-color: #4CAF50; color: white; font-weight: bold; }")
        # apply_btn.clicked.connect(self.apply_semantic_segmentation)

        # Сбросить к обычной обработке
        reset_btn = QPushButton("Вернуться к обычной обработке")
        reset_btn.clicked.connect(self.reset_to_normal)

        buttons_layout.addWidget(preview_btn)
        # buttons_layout.addWidget(apply_btn)
        buttons_layout.addWidget(reset_btn)

        # Добавляем все в основной layout
        layout.addWidget(info_group)
        layout.addWidget(model_group)
        layout.addWidget(params_group)
        layout.addLayout(buttons_layout)
        layout.addStretch()

    def preview_segmentation(self):
        if hasattr(self.parent, 'preview_semantic_segmentation'):
            self.parent.preview_semantic_segmentation()

    def apply_semantic_segmentation(self):
        if hasattr(self.parent, 'apply_semantic_segmentation'):
            self.parent.apply_semantic_segmentation()

    def reset_to_normal(self):
        if hasattr(self.parent, 'reset_to_normal_processing'):
            self.parent.reset_to_normal_processing()

    def get_parameters(self):
        """Возвращает параметры для семантической сегментации"""
        return {
            'model': self.model_combo.currentText(),
            'max_classes': self.classes_spin.value(),
            'adaptive_params': self.adaptive_params_check.isChecked(),
            'face_quality': self.face_quality_check.isChecked()
        }