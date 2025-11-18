from PyQt5.QtWidgets import QWidget, QVBoxLayout, QPushButton


class ExportTab(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.parent = parent
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout(self)

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

    def save_image(self, image_type):
        if hasattr(self.parent, 'save_image'):
            self.parent.save_image(image_type)