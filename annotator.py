import os
import numpy as np
import pandas as pd
from PyQt5.QtWidgets import QApplication, QWidget, QPushButton, QVBoxLayout, QLabel, QFileDialog, QMessageBox, QSlider, QCheckBox, QHBoxLayout
from PyQt5.QtCore import Qt,QSize
from PyQt5.QtGui import QPixmap, QImage

class ErrorAnnotationApp(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()
        self.directory = None
        self.front_camera_directory = None
        self.rear_camera_directory = None
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

        self.image_layout = QHBoxLayout()
        self.front_image_label = QLabel(self)
        self.rear_image_label = QLabel(self)
        self.image_layout.addWidget(self.front_image_label)
        self.image_layout.addWidget(self.rear_image_label)
        layout.addLayout(self.image_layout)

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
            self.checkpoint_path = f"{parent_of_innermost}_annotations_checkpoint.csv"

            # Load .npy files from the selected directory
            self.npy_files = sorted([f for f in os.listdir(self.directory) if f.endswith('.npy')])

            # Set paths for front and rear camera directories
            base_dir = os.path.dirname(self.directory)
            self.front_camera_directory = os.path.join(base_dir, 'front_camera')
            self.rear_camera_directory = os.path.join(base_dir, 'rear_camera')

            if not (os.path.exists(self.front_camera_directory) and os.path.exists(self.rear_camera_directory)):
                QMessageBox.warning(self, 'Error', 'Corresponding front_camera and rear_camera directories not found.')
                return

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
            self.display_npy_content()

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

    def display_npy_content(self):
        if self.current_file_index < len(self.npy_files):
            current_file = self.npy_files[self.current_file_index]
            file_name = os.path.splitext(current_file)[0]
            front_image_path = os.path.join(self.front_camera_directory, f"{file_name}.npy")
            rear_image_path = os.path.join(self.rear_camera_directory, f"{file_name}.npy")
            print(front_image_path,"\n", rear_image_path)
            try:
                # Load front camera image
                if os.path.exists(front_image_path):
                    front_image = np.load(front_image_path)
                    front_pixmap = self.numpy_array_to_qpixmap(front_image, scale=0.5)  # Scale to half size
                    self.front_image_label.setPixmap(front_pixmap)
                else:
                    self.front_image_label.setText("Front camera image not found.")

                # Load rear camera image
                if os.path.exists(rear_image_path):
                    rear_image = np.load(rear_image_path)
                    rear_pixmap = self.numpy_array_to_qpixmap(rear_image, scale=0.5)  # Scale to half size
                    self.rear_image_label.setPixmap(rear_pixmap)
                else:
                    self.rear_image_label.setText("Rear camera image not found.")

            except Exception as e:
                self.front_image_label.setText(f'Error loading images: {e}')
                self.rear_image_label.setText(f'Error loading images: {e}')

    def numpy_array_to_qpixmap(self, np_array, scale=0.5):
        height, width, channels = np_array.shape
        bytes_per_line = channels * width
        qimage = QImage(np_array.data, width, height, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qimage)
        if scale != 1.0:
            size = QSize(int(width * scale), int(height * scale))
            pixmap = pixmap.scaled(size)
        return pixmap

if __name__ == '__main__':
    app = QApplication([])
    ex = ErrorAnnotationApp()
    ex.show()
    app.exec_()

# import os
# import numpy as np
# import pandas as pd
# from PyQt5.QtWidgets import QApplication, QWidget, QPushButton, QVBoxLayout, QLabel, QFileDialog, QMessageBox, QSlider, QCheckBox
# from PyQt5.QtCore import Qt
# from PyQt5.QtGui import QPixmap, QImage

# class ErrorAnnotationApp(QWidget):
#     def __init__(self):
#         super().__init__()
#         self.initUI()
#         self.directory = None
#         self.npy_files = []
#         self.current_file_index = 0
#         self.df = pd.DataFrame(columns=['file_name', 'error_annotation', 'no_error_annotation', 'error_annotation_filtered', 'no_error_annotation_filtered'])
#         self.checkpoint_path = None

#     def initUI(self):
#         self.setWindowTitle('Error Annotation Tool')
#         layout = QVBoxLayout()

#         self.label = QLabel('Select a directory to start', self)
#         layout.addWidget(self.label)

#         dir_button = QPushButton('Choose Directory', self)
#         dir_button.clicked.connect(self.load_directory)
#         layout.addWidget(dir_button)

#         load_button = QPushButton('Load CSV', self)
#         load_button.clicked.connect(self.load_from_csv)
#         layout.addWidget(load_button)

#         self.file_label = QLabel('Current File: None', self)
#         layout.addWidget(self.file_label)

#         self.image_label = QLabel(self)
#         layout.addWidget(self.image_label)

#         self.counter_label = QLabel('Annotated: 0 / 0', self)
#         layout.addWidget(self.counter_label)

#         self.slider = QSlider(Qt.Horizontal)
#         self.slider.setTickPosition(QSlider.TicksBelow)
#         self.slider.setTickInterval(1)
#         self.slider.setMinimum(0)
#         self.slider.valueChanged.connect(self.slider_changed)
#         layout.addWidget(self.slider)

#         # Checkboxes for annotations
#         self.error_checkbox = QCheckBox('Annotate as Error', self)
#         layout.addWidget(self.error_checkbox)

#         self.no_error_checkbox = QCheckBox('Annotate as No Error', self)
#         layout.addWidget(self.no_error_checkbox)

#         self.filtered_error_checkbox = QCheckBox('Annotate as Error (Filtered)', self)
#         layout.addWidget(self.filtered_error_checkbox)

#         self.filtered_no_error_checkbox = QCheckBox('Annotate as No Error (Filtered)', self)
#         layout.addWidget(self.filtered_no_error_checkbox)

#         # Annotate all button
#         annotate_button = QPushButton('Annotate Selected', self)
#         annotate_button.clicked.connect(self.annotate_selected)
#         annotate_button.setEnabled(False)
#         layout.addWidget(annotate_button)
#         self.annotate_button = annotate_button

#         save_button = QPushButton('Save to CSV', self)
#         save_button.clicked.connect(self.save_to_csv)
#         layout.addWidget(save_button)

#         save_checkpoint_button = QPushButton('Save Checkpoint', self)
#         save_checkpoint_button.clicked.connect(self.save_checkpoint)
#         layout.addWidget(save_checkpoint_button)

#         self.setLayout(layout)

#     def load_directory(self):
#         self.directory = QFileDialog.getExistingDirectory(self, "Select Directory")
#         if self.directory:
#             innermost_folder = os.path.basename(self.directory.rstrip('/'))
#             parent_of_innermost = os.path.basename(os.path.dirname(self.directory.rstrip('/')))
#             self.checkpoint_path = f"{parent_of_innermost}_annotations_checkpoint.csv"

#             # Load .npy files from the selected directory
#             npy_files_selected_dir = [f for f in os.listdir(self.directory) if f.endswith('.npy')]

#             # Load .npy files from the ../camera directory
#             print(os.path.dirname(self.directory))
#             camera_directory = os.path.join(os.path.dirname(self.directory),'..', 'camera')
#             npy_files_camera_dir = []
#             if os.path.exists(camera_directory):
#                 npy_files_camera_dir = [os.path.join('camera', f) for f in os.listdir(camera_directory) if f.endswith('.npy')]

#             self.npy_files = sorted(npy_files_selected_dir + npy_files_camera_dir)

#             self.slider.setMaximum(len(self.npy_files) - 1)
#             self.slider.setValue(0)
#             self.slider.setEnabled(True)
#             self.update_annotation_buttons(True)
#             self.update_current_file()
#             self.update_counter_label()

#     def load_from_csv(self):
#         load_path, _ = QFileDialog.getOpenFileName(self, "Load File", "", "CSV files (*.csv)")
#         if load_path:
#             self.df = pd.read_csv(load_path)
#             self.update_current_file()
#             self.update_counter_label()

#     def update_current_file(self):
#         if self.npy_files:
#             current_file = self.npy_files[self.current_file_index]
#             self.file_label.setText(f'Current File: {current_file}')
#             self.display_npy_content()

#     def slider_changed(self, value):
#         self.current_file_index = value
#         self.update_current_file()

#     def update_annotation_buttons(self, enable):
#         self.error_checkbox.setEnabled(enable)
#         self.no_error_checkbox.setEnabled(enable)
#         self.filtered_error_checkbox.setEnabled(enable)
#         self.filtered_no_error_checkbox.setEnabled(enable)
#         self.annotate_button.setEnabled(enable)

#     def annotate(self, file_name, error, no_error, error_filtered, no_error_filtered):
#         temp_df = pd.DataFrame([{
#             'file_name': file_name,
#             'error_annotation': error,
#             'no_error_annotation': no_error,
#             'error_annotation_filtered': error_filtered,
#             'no_error_annotation_filtered': no_error_filtered
#         }])
#         self.df = pd.concat([self.df, temp_df], ignore_index=True)
#         QMessageBox.information(self, 'Annotation', f'File "{file_name}" annotated')
#         self.update_counter_label()

#     def annotate_selected(self):
#         if self.current_file_index < len(self.npy_files):
#             file_name = self.npy_files[self.current_file_index]
#             error = self.error_checkbox.isChecked()
#             no_error = self.no_error_checkbox.isChecked()
#             error_filtered = self.filtered_error_checkbox.isChecked()
#             no_error_filtered = self.filtered_no_error_checkbox.isChecked()

#             self.annotate(file_name, error, no_error, error_filtered, no_error_filtered)
#             self.slider.setValue(self.current_file_index + 1)  # Automatically move to the next file

#     def save_to_csv(self):
#         if self.directory:
#             innermost_folder = os.path.basename(self.directory.rstrip('/'))
#             save_path = f"{innermost_folder}_annotations.csv"
#             self.df.to_csv(save_path, index=False)
#             QMessageBox.information(self, 'File Saved', f'Annotations saved to {save_path}')

#     def save_checkpoint(self):
#         if self.checkpoint_path:
#             self.df.to_csv(self.checkpoint_path, index=False)
#             QMessageBox.information(self, 'Checkpoint Saved', f'Annotations checkpoint saved to {self.checkpoint_path}')

#     def update_counter_label(self):
#         annotated_count = self.df['file_name'].nunique()
#         total_count = len(self.npy_files)
#         self.counter_label.setText(f'Annotated: {annotated_count} / {total_count}')

#     def display_npy_content(self):
#         if self.current_file_index < len(self.npy_files):
#             current_file = self.npy_files[self.current_file_index]
#             file_path = os.path.join(self.directory, current_file) if 'camera' not in current_file else os.path.join(os.path.dirname(self.directory), current_file)
#             try:
#                 npy_content = np.load(file_path)
#                 print(npy_content.shape)
#                 # npy_content = npy_content.astype(np.uint8)  # Ensure the content is in the correct format for an image
#                 # height, width = npy_content.shape[1:2]

#                 # bytes_per_line = width * npy_content.shape[2] if npy_content.ndim == 3 else width
#                 # qimage = QImage(npy_content.data, width, height, bytes_per_line, QImage.Format_RGB888 if npy_content.ndim == 3 else QImage.Format_Grayscale8)
#                 # pixmap = QPixmap.fromImage(qimage)

#                 # self.image_label.setPixmap(pixmap)
#             except Exception as e:
#                 self.image_label.setText(f'Error loading image: {e}')

# if __name__ == '__main__':
#     app = QApplication([])
#     ex = ErrorAnnotationApp()
#     ex.show()
#     app.exec_()
 