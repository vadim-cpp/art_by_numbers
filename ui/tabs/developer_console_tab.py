from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QGroupBox, QTextEdit,
                             QPushButton, QCheckBox, QHBoxLayout, QLabel)
from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui import QFont, QTextCursor
import datetime


class DeveloperConsoleTab(QWidget):
    clear_signal = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.parent = parent
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout(self)

        # Заголовок
        title_label = QLabel("Консоль разработчика - Метрики кластеризации")
        title_label.setFont(QFont("Arial", 12, QFont.Bold))
        layout.addWidget(title_label)

        # Группа управления
        control_group = QGroupBox("Управление логированием")
        control_layout = QHBoxLayout(control_group)

        self.auto_scroll_check = QCheckBox("Автопрокрутка")
        self.auto_scroll_check.setChecked(True)

        self.timestamp_check = QCheckBox("Показывать время")
        self.timestamp_check.setChecked(True)

        clear_btn = QPushButton("Очистить консоль")
        clear_btn.clicked.connect(self.clear_console)

        export_btn = QPushButton("Экспорт логов")
        export_btn.clicked.connect(self.export_logs)

        control_layout.addWidget(self.auto_scroll_check)
        control_layout.addWidget(self.timestamp_check)
        control_layout.addWidget(clear_btn)
        control_layout.addWidget(export_btn)
        control_layout.addStretch()

        # Консоль вывода
        console_group = QGroupBox("Метрики и логи")
        console_layout = QVBoxLayout(console_group)

        self.console_output = QTextEdit()
        self.console_output.setReadOnly(True)
        self.console_output.setFont(QFont("Consolas", 9))
        self.console_output.setStyleSheet("""
            QTextEdit {
                background-color: #1e1e1e;
                color: #d4d4d4;
                border: 1px solid #444;
            }
        """)

        console_layout.addWidget(self.console_output)

        layout.addWidget(control_group)
        layout.addWidget(console_group)

    def log_message(self, message, message_type="info"):
        """Добавляет сообщение в консоль"""
        timestamp = ""
        if self.timestamp_check.isChecked():
            timestamp = f"[{datetime.datetime.now().strftime('%H:%M:%S')}] "

        # Цвета для разных типов сообщений
        colors = {
            "info": "#d4d4d4",
            "success": "#4EC9B0",
            "warning": "#FFD700",
            "error": "#F44747",
            "metric": "#569CD6",
            "header": "#CE9178"
        }

        color = colors.get(message_type, "#d4d4d4")
        html_message = f'<span style="color: {color}">{timestamp}{message}</span><br>'

        self.console_output.moveCursor(QTextCursor.End)
        self.console_output.insertHtml(html_message)

        if self.auto_scroll_check.isChecked():
            self.console_output.moveCursor(QTextCursor.End)

    def clear_console(self):
        """Очищает консоль"""
        self.console_output.clear()

    def export_logs(self):
        """Экспортирует логи в файл"""
        from PyQt5.QtWidgets import QFileDialog
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Экспорт логов", "", "Text Files (*.txt)")

        if file_path:
            try:
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(self.console_output.toPlainText())
                self.log_message(f"Логи экспортированы в: {file_path}", "success")
            except Exception as e:
                self.log_message(f"Ошибка экспорта: {str(e)}", "error")