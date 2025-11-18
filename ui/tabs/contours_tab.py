from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QGroupBox, QPushButton,
                             QComboBox, QSpinBox, QCheckBox)


class ContoursTab(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.parent = parent
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout(self)

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

    def apply_contours(self):
        if hasattr(self.parent, 'apply_contours'):
            self.parent.apply_contours()