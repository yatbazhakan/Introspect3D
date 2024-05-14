import os
import numpy as np
import pandas as pd
from PyQt5.QtWidgets import QApplication, QWidget, QPushButton, QVBoxLayout, QLabel, QFileDialog, QMessageBox, QSlider
from PyQt5.QtCore import Qt

class ErrorAnnotationApp(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()
        self.directory = None
        self.npy_files = []
        self.current_file_index = 0
        self.df = pd.DataFrame(columns=['file_name', 'error_annotation', 'filtered'])
        self.checkpoint_path = None

    def initUI(self):
        self.setWindowTitle('Error Annotation Tool')
        layout = QVBoxLayout()

        self.label = QLabel('Select a directory to start', self)
        layout.addWidget(self.label)

        dir_button = QPushButton('Choose Directory', self)
        dir_button.clicked.connect(self.load_directory)
        layout.addWidget(dir_button)

        load_button = QPushButton('Load CSV', self)
        load_button.clicked.connect(self.load_from_csv)
        layout.addWidget(load_button)

        self.file_label = QLabel('Current File: None', self)
        layout.addWidget(self.file_label)

        self.slider = QSlider(Qt.Horizontal)
        self.slider.setTickPosition(QSlider.TicksBelow)
        self.slider.setTickInterval(1)
        self.slider.setMinimum(0)
        self.slider.valueChanged.connect(self.slider_changed)
        layout.addWidget(self.slider)

        self.error_button = QPushButton('Annotate as Error', self)
        self.error_button.clicked.connect(lambda: self.annotate('error', False))
        self.error_button.setEnabled(False)
        layout.addWidget(self.error_button)

        self.no_error_button = QPushButton('Annotate as No Error', self)
        self.no_error_button.clicked.connect(lambda: self.annotate('no error', False))
        self.no_error_button.setEnabled(False)
        layout.addWidget(self.no_error_button)

        self.filtered_error_button = QPushButton('Annotate as Error (Filtered)', self)
        self.filtered_error_button.clicked.connect(lambda: self.annotate('error', True))
        self.filtered_error_button.setEnabled(False)
        layout.addWidget(self.filtered_error_button)

        self.filtered_no_error_button = QPushButton('Annotate as No Error (Filtered)', self)
        self.filtered_no_error_button.clicked.connect(lambda: self.annotate('no error', True))
        self.filtered_no_error_button.setEnabled(False)
        layout.addWidget(self.filtered_no_error_button)

        save_button = QPushButton('Save to CSV', self)
        save_button.clicked.connect(self.save_to_csv)
        layout.addWidget(save_button)

        save_checkpoint_button = QPushButton('Save Checkpoint', self)
        save_checkpoint_button.clicked.connect(self.save_checkpoint)
        layout.addWidget(save_checkpoint_button)

        self.setLayout(layout)

    def load_directory(self):
        self.directory = QFileDialog.getExistingDirectory(self, "Select Directory")
        if self.directory:
            innermost_folder = os.path.basename(self.directory.rstrip('/'))
            self.checkpoint_path = f"{innermost_folder}_annotations_checkpoint.csv"
            self.npy_files = sorted([f for f in os.listdir(self.directory) if f.endswith('.npy')])
            self.slider.setMaximum(len(self.npy_files) - 1)
            self.slider.setValue(0)
            self.slider.setEnabled(True)
            self.update_annotation_buttons(True)
            self.update_current_file()

    def load_from_csv(self):
        load_path, _ = QFileDialog.getOpenFileName(self, "Load File", "", "CSV files (*.csv)")
        if load_path:
            self.df = pd.read_csv(load_path)
            self.update_current_file()

    def update_current_file(self):
        if self.npy_files:
            current_file = self.npy_files[self.current_file_index]
            self.file_label.setText(f'Current File: {current_file}')

    def slider_changed(self, value):
        self.current_file_index = value
        self.update_current_file()

    def update_annotation_buttons(self, enable):
        self.error_button.setEnabled(enable)
        self.no_error_button.setEnabled(enable)
        self.filtered_error_button.setEnabled(enable)
        self.filtered_no_error_button.setEnabled(enable)

    def annotate(self, annotation, filtered):
        if self.current_file_index < len(self.npy_files):
            file_name = self.npy_files[self.current_file_index]
            self.df = self.df.append({'file_name': file_name, 'error_annotation': annotation, 'filtered': filtered}, ignore_index=True)
            QMessageBox.information(self, 'Annotation', f'File "{file_name}" annotated as "{annotation}" (Filtered: {filtered})')
            self.slider.setValue(self.current_file_index + 1)  # Automatically move to the next file

    def save_to_csv(self):
        if self.directory:
            innermost_folder = os.path.basename(self.directory.rstrip('/'))
            save_path = f"{innermost_folder}_annotations.csv"
            self.df.to_csv(save_path, index=False)
            QMessageBox.information(self, 'File Saved', f'Annotations saved to {save_path}')

    def save_checkpoint(self):
        if self.checkpoint_path:
            self.df.to_csv(self.checkpoint_path, index=False)
            QMessageBox.information(self, 'Checkpoint Saved', f'Annotations checkpoint saved to {self.checkpoint_path}')

if __name__ == '__main__':
    app = QApplication([])
    ex = ErrorAnnotationApp()
    ex.show()
    app.exec_()
