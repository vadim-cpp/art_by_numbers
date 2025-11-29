from PyQt5.QtCore import QThread, pyqtSignal


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

            # Новые операции для семантической сегментации
            elif self.operation == "semantic_full":
                # Полная семантическая обработка
                semantic_mask = self.processor.semantic_segment(self.processor.work)
                result = self.processor.semantic_aware_quantize(
                    self.processor.work,
                    n_colors=self.kwargs.get('n_colors', 12),
                    adaptive_params=self.kwargs.get('adaptive_params', True),
                    face_quality=self.kwargs.get('face_quality', True)
                )
            elif self.operation == "semantic_segment":
                result = self.processor.semantic_segment(self.processor.work)
            else:
                result = None

            self.finished.emit(result)
        except Exception as e:
            self.finished.emit(e)