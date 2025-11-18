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
            else:
                result = None

            self.finished.emit(result)
        except Exception as e:
            self.finished.emit(e)