import sys
import yaml
from PyQt5.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QHBoxLayout, QWidget, QLabel, QLineEdit, QCheckBox, QPushButton, QComboBox, QSpinBox, QDoubleSpinBox

class YamlGenerator(QMainWindow):
    def __init__(self):
        super().__init__()
        self.inputs = {}
        self.initUI()

    def initUI(self):
        self.setWindowTitle('YAML Generator')
        self.setGeometry(100, 100, 1000, 800)

        # Main Layout
        layout = QVBoxLayout()

        # Experiment Name
        self.createInputField(layout, "Experiment Name", "introspection/experiment_name", QLineEdit)

        # WandB Configurations
        wandb_layout = QVBoxLayout()
        self.createInputField(wandb_layout, "WandB - Is Sweep", "introspection/wandb/is_sweep", QCheckBox)
        self.createInputField(wandb_layout, "WandB - Project", "introspection/wandb/project", QLineEdit)
        self.createInputField(wandb_layout, "WandB - Entity", "introspection/wandb/entity", QLineEdit)
        self.createInputField(wandb_layout, "WandB - Mode", "introspection/wandb/mode", QLineEdit)
        self.createInputField(wandb_layout, "WandB - Name", "introspection/wandb/name", QLineEdit)
        self.createSection(layout, "WandB Configuration", wandb_layout)

        # ... Additional fields in a similar manner ...

        # Save Button
        btn = QPushButton('Generate YAML', self)
        btn.clicked.connect(self.generateYaml)
        layout.addWidget(btn)

        # Set main layout
        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)

    def createInputField(self, layout, label, key, widgetType):
        # Create a horizontal layout
        h_layout = QHBoxLayout()

        # Label
        lbl = QLabel(label)
        h_layout.addWidget(lbl)

        # Input widget
        if widgetType == QLineEdit:
            widget = QLineEdit(self)
        elif widgetType == QCheckBox:
            widget = QCheckBox(self)
            widget.stateChanged.connect(lambda state, key=key: self.updateInput(key, state == Qt.Checked))
        elif widgetType == QComboBox:
            widget = QComboBox(self)
            # Add relevant options to the combo box
        elif widgetType == QSpinBox or widgetType == QDoubleSpinBox:
            widget = widgetType(self)
            widget.valueChanged.connect(lambda value, key=key: self.updateInput(key, value))
        else:
            raise ValueError("Unsupported widget type")

        if widgetType != QCheckBox:
            widget.textChanged.connect(lambda text, key=key: self.updateInput(key, text))

        h_layout.addWidget(widget)

        # Add to parent layout
        layout.addLayout(h_layout)
        self.inputs[key] = widget

    def createSection(self, parentLayout, title, sectionLayout):
        # Section Label
        section_label = QLabel(f"<b>{title}</b>")
        parentLayout.addWidget(section_label)
        parentLayout.addLayout(sectionLayout)

    def updateInput(self, key, value):
        keys = key.split('/')
        data = self.inputs
        for k in keys[:-1]:
            data = data.setdefault(k, {})
        data[keys[-1]] = value

    def generateYaml(self):
        try:
            with open('output.yaml', 'w') as file:
                yaml.dump(self.inputs, file, sort_keys=False, default_flow_style=False)
            print("YAML file generated successfully.")
        except Exception as e:
            print(f"Error generating YAML: {e}")

if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = YamlGenerator()
    ex.show()
    sys.exit(app.exec_())
