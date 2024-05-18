import os
import numpy as np
import pandas as pd
from PyQt5.QtWidgets import QApplication, QWidget, QPushButton, QVBoxLayout, QLabel, QFileDialog, QMessageBox, QSlider, QCheckBox
from PyQt5.QtCore import Qt

class ErrorAnnotationApp(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()
        self.directory = None
        self.npy_files = []
        self.current_file_index = 0
        self.df = pd.DataFrame(columns=['file_name', 'error_annotation', 'no_error_annotation', 'error_annotation_filtered', 'no_error_annotation_filtered'])
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

        self.counter_label = QLabel('Annotated: 0 / 0', self)
        layout.addWidget(self.counter_label)

        self.slider = QSlider(Qt.Horizontal)
        self.slider.setTickPosition(QSlider.TicksBelow)
        self.slider.setTickInterval(1)
        self.slider.setMinimum(0)
        self.slider.valueChanged.connect(self.slider_changed)
        layout.addWidget(self.slider)

        # Checkboxes for annotations
        self.error_checkbox = QCheckBox('Annotate as Error', self)
        layout.addWidget(self.error_checkbox)

        self.no_error_checkbox = QCheckBox('Annotate as No Error', self)
        layout.addWidget(self.no_error_checkbox)

        self.filtered_error_checkbox = QCheckBox('Annotate as Error (Filtered)', self)
        layout.addWidget(self.filtered_error_checkbox)

        self.filtered_no_error_checkbox = QCheckBox('Annotate as No Error (Filtered)', self)
        layout.addWidget(self.filtered_no_error_checkbox)

        # Annotate all button
        annotate_button = QPushButton('Annotate Selected', self)
        annotate_button.clicked.connect(self.annotate_selected)
        annotate_button.setEnabled(False)
        layout.addWidget(annotate_button)
        self.annotate_button = annotate_button

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
            parent_of_innermost = os.path.basename(os.path.dirname(self.directory.rstrip('/')))
            print(parent_of_innermost)
            self.checkpoint_path = f"{parent_of_innermost}_annotations_checkpoint.csv"
            self.npy_files = sorted([f for f in os.listdir(self.directory) if f.endswith('.npy')])
            self.slider.setMaximum(len(self.npy_files) - 1)
            self.slider.setValue(0)
            self.slider.setEnabled(True)
            self.update_annotation_buttons(True)
            self.update_current_file()
            self.update_counter_label()

    def load_from_csv(self):
        load_path, _ = QFileDialog.getOpenFileName(self, "Load File", "", "CSV files (*.csv)")
        if load_path:
            self.df = pd.read_csv(load_path)
            self.update_current_file()
            self.update_counter_label()

    def update_current_file(self):
        if self.npy_files:
            current_file = self.npy_files[self.current_file_index]
            self.file_label.setText(f'Current File: {current_file}')

    def slider_changed(self, value):
        self.current_file_index = value
        self.update_current_file()

    def update_annotation_buttons(self, enable):
        self.error_checkbox.setEnabled(enable)
        self.no_error_checkbox.setEnabled(enable)
        self.filtered_error_checkbox.setEnabled(enable)
        self.filtered_no_error_checkbox.setEnabled(enable)
        self.annotate_button.setEnabled(enable)

    def annotate(self, file_name, error, no_error, error_filtered, no_error_filtered):
        temp_df = pd.DataFrame([{
            'file_name': file_name,
            'error_annotation': error,
            'no_error_annotation': no_error,
            'error_annotation_filtered': error_filtered,
            'no_error_annotation_filtered': no_error_filtered
        }])
        self.df = pd.concat([self.df, temp_df], ignore_index=True)
        QMessageBox.information(self, 'Annotation', f'File "{file_name}" annotated')
        self.update_counter_label()

    def annotate_selected(self):
        if self.current_file_index < len(self.npy_files):
            file_name = self.npy_files[self.current_file_index]
            error = self.error_checkbox.isChecked()
            no_error = self.no_error_checkbox.isChecked()
            error_filtered = self.filtered_error_checkbox.isChecked()
            no_error_filtered = self.filtered_no_error_checkbox.isChecked()

            self.annotate(file_name, error, no_error, error_filtered, no_error_filtered)
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

    def update_counter_label(self):
        annotated_count = self.df['file_name'].nunique()
        total_count = len(self.npy_files)
        self.counter_label.setText(f'Annotated: {annotated_count} / {total_count}')

if __name__ == '__main__':
    app = QApplication([])
    ex = ErrorAnnotationApp()
    ex.show()
    app.exec_()
