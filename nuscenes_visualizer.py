import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel, QPushButton, QLineEdit, QVBoxLayout, QWidget, QFileDialog, QListWidget, QMessageBox, QComboBox, QSizePolicy
import pandas as pd
import subprocess
import os
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("CSV Reader")
        self.setGeometry(100, 100, 400, 300)

        self.df = pd.DataFrame()

        self.initUI()

    def initUI(self):
        # Load CSV button
        load_button = QPushButton("Load CSV", self)
        load_button.clicked.connect(self.load_csv)
        
        # Scene selection
        self.scene_combobox = QComboBox(self)
        self.scene_combobox.setSizeAdjustPolicy(QComboBox.AdjustToContents)
        self.scene_combobox.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)

        self.scene_combobox.setMaxVisibleItems(10)  # Set maximum visible items
        self.scene_combobox.activated.connect(self.select_scene)

        # Token entry
        self.token_entry = QLineEdit(self)

        # Search token
        search_token_button = QPushButton("Search Token", self)
        search_token_button.clicked.connect(self.search_token)

        # Search name
        self.search_name_entry = QLineEdit(self)
        search_name_button = QPushButton("Search Name", self)
        search_name_button.clicked.connect(self.search_name)

        # Scene info
        self.scene_name_label = QLabel(self)
        self.scene_desc_text = QLabel(self)

        # Token list
        token_list_label = QLabel("Token List:", self)
        self.token_listbox = QListWidget(self)
        add_button = QPushButton("Add to List", self)
        add_button.clicked.connect(self.add_to_list)
        clear_button = QPushButton("Clear List", self)
        clear_button.clicked.connect(self.clear_list)

        # Run button
        run_button = QPushButton("Run", self)
        run_button.clicked.connect(self.run_scripts)

        # Layout
        layout = QVBoxLayout()
        layout.addWidget(load_button)
        layout.addWidget(self.scene_combobox)
        layout.addWidget(self.token_entry)
        layout.addWidget(search_token_button)
        layout.addWidget(self.search_name_entry)
        layout.addWidget(search_name_button)
        layout.addWidget(self.scene_name_label)
        layout.addWidget(self.scene_desc_text)
        layout.addWidget(token_list_label)
        layout.addWidget(self.token_listbox)
        layout.addWidget(add_button)
        layout.addWidget(clear_button)
        layout.addWidget(run_button)

        widget = QWidget()
        widget.setLayout(layout)
        self.setCentralWidget(widget)

    def load_csv(self):
        #If we have other csvs
        # file_path, _ = QFileDialog.getOpenFileName(self, "Open CSV File", "", "CSV Files (*.csv)")

        file_path= "nuscenes_scene_tokens.csv"
        if file_path:
            self.df = pd.read_csv(file_path)
            scene_names = self.df['name'].unique().tolist()
            self.scene_combobox.addItems(scene_names)
            QMessageBox.information(self, "Notification", "CSV file loaded successfully.")

    def select_scene(self):
        scene_name = self.scene_combobox.currentText()
        if scene_name:
            result = self.df[self.df['name'] == scene_name]
            if not result.empty:
                self.token_entry.setText(result.iloc[0]['scene_token'])
                self.scene_name_label.setText(result.iloc[0]['name'])
                self.scene_desc_text.setText(result.iloc[0]['description'])
            else:
                self.scene_name_label.setText('Scene not found')
                self.scene_desc_text.setText('')
        else:
            self.scene_name_label.setText('')
            self.scene_desc_text.setText('')

    def search_token(self):
        token = self.token_entry.text()
        if token:
            result = self.df[self.df['scene_token'] == token]
            if not result.empty:
                self.scene_name_label.setText(result.iloc[0]['name'])
                self.scene_desc_text.setText(result.iloc[0]['description'])
            else:
                self.scene_name_label.setText('Token not found')
                self.scene_desc_text.setText('')
        else:
            self.scene_name_label.setText('')
            self.scene_desc_text.setText('')

    def search_name(self):
        name = self.search_name_entry.text()
        if name:
            result = self.df[self.df['name'] == name]
            if not result.empty:
                self.token_entry.setText(result.iloc[0]['scene_token'])
                self.scene_name_label.setText(result.iloc[0]['name'])
                self.scene_desc_text.setText(result.iloc[0]['description'])
            else:
                self.scene_name_label.setText('Name not found')
                self.scene_desc_text.setText('')
        else:
            self.scene_name_label.setText('')
            self.scene_desc_text.setText('')

    def add_to_list(self):
        token = self.token_entry.text()
        if token:
            if token not in [self.token_listbox.item(i).text() for i in range(self.token_listbox.count())]:
                self.token_listbox.addItem(token)

    def clear_list(self):
        self.token_listbox.clear()
    def build_python_script(self, tokens):
        # Construct command to run python script within the Conda environment
        base_str = "conda run -n openmmlab2 --no-capture-output python3 /mnt/ssd2/Introspect3D/open3d_vis.py -n "
        script = base_str + ' '.join(map(lambda token: f'"{token}"', tokens))
        return script

    def open_terminal(self, script):
        # Start the script using gnome-terminal
        terminal = subprocess.Popen(['gnome-terminal', '--', 'bash', '-c', script], stdin=subprocess.PIPE)
        return terminal

    def run_scripts(self):
        # Collect tokens and execute the script in a new terminal
        tokens = [self.token_listbox.item(i).text() for i in range(self.token_listbox.count())]
        script = self.build_python_script(tokens)
        terminal = self.open_terminal(script)

        # subprocess.Popen(["gnome-terminal", "--", "python3", "script.py", "-p"] + [token for token in tokens])
        # subprocess.Popen(["gnome-terminal", "--", "bash", "-c", f"python3 script.py -n {token}], shell=True)
if __name__ == '__main__':
    app = QApplication(sys.argv)
    mainWindow = MainWindow()
    mainWindow.show()
    sys.exit(app.exec_())
