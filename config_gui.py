import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import yaml
from tkinter import filedialog, messagebox, ttk, simpledialog
from tkinter.font import Font
from definitions import CONFIG_DIR
class YAMLGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("YAML Multi-level Editor with Collapse Functionality")
        self.font_size = 10
        self.bold_font = Font(family="Helvetica", size=self.font_size, weight="bold")
        self.style = ttk.Style()
        self.style.configure("Bold.Treeview", font=self.bold_font)

        self.tree_frame = tk.Frame(root)
        self.tree_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        self.tree = ttk.Treeview(self.tree_frame, columns=('Value',), style="Bold.Treeview")

        self.tree.column('#0', width=150)
        self.tree.column('value', width=400, anchor='w')
        self.tree.heading('#0', text='Key')
        self.tree.heading('value', text='Value')
        self.tree.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        self.tree.bind('<Double-1>', self.edit_tree_item)
        self.root.bind('<Configure>', self.on_resize)
        self.collapse_button = tk.Button(self.tree_frame, text="Collapse All", command=self.collapse_all)
        self.collapse_button.pack(side=tk.BOTTOM)
        self.tree.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        self.open_button = tk.Button(self.tree_frame, text="Open All", command=self.open_all)
        self.open_button.pack(side=tk.BOTTOM)

        # self.editor = tk.Text(root, wrap="none")
        # self.editor.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.menu = tk.Menu(self.root)
        self.root.config(menu=self.menu)
        self.file_menu = tk.Menu(self.menu, tearoff=0)
        self.menu.add_cascade(label="File", menu=self.file_menu)
        self.file_menu.add_command(label="Open", command=self.open_file)
        self.file_menu.add_command(label="Save", command=self.save_file)
        self.file_menu.add_command(label="Export As", command=self.export_file_as)
        self.file_menu.add_separator()
        self.file_menu.add_command(label="Exit", command=root.quit)

        self.yaml_data = None
        self.current_selection = None
        self.root.bind('<Configure>', self.on_resize)

    def on_resize(self, event):
        width, height = event.width, event.height
        new_font_size = max(8, min(int(width / 80), 20))  # Example calculation, adjust as needed
        if new_font_size != self.font_size:
            self.font_size = new_font_size
            self.bold_font.configure(size=self.font_size)
            self.style.configure("Bold.Treeview", font=self.bold_font)

    def edit_tree_item(self, event):
        column = self.tree.identify_column(event.x)
        item = self.tree.identify_row(event.y)

        if column == '#1':  # We allow editing only the 'value' column
            path = self.get_path(item)
            current_value = self.get_nested_value(self.yaml_data, path)

            if not isinstance(current_value, dict):  # We don't allow editing dict values directly
                new_value = simpledialog.askstring("Edit Value", "Edit the value:", initialvalue=current_value)
                if new_value is not None:
                    self.set_nested_value(self.yaml_data, path, new_value)
                    self.tree.set(item, column=column, value=new_value)

    def open_file(self):
        file_path = filedialog.askopenfilename(initialdir=CONFIG_DIR,filetypes=[("YAML files", "*.yaml")])
        if file_path:
            with open(file_path, 'r') as file:
                self.yaml_data = yaml.safe_load(file)
                self.populate_tree()

    def populate_tree(self, node='', value=None):
        if value is None:
            value = self.yaml_data

        if isinstance(value, dict):
            for key, val in value.items():
                node_id = self.tree.insert(node, 'end', text=key, values=[str(val) if not isinstance(val, dict) else ''])
                if isinstance(val, dict):
                    self.populate_tree(node_id, val)

    def get_path(self, item_id):
        path = []
        while item_id:
            path.insert(0, self.tree.item(item_id, 'text'))
            item_id = self.tree.parent(item_id)
        return path

    def on_tree_select(self, event):
        selected_item = self.tree.selection()[0]
        self.current_selection = selected_item
        # self.editor.delete(1.0, tk.END)
        path = self.get_path(selected_item)

        # Retrieve data from the YAML structure using the constructed path
        data = self.yaml_data
        for key in path:
            if isinstance(data, dict):
                data = data.get(key, {})

        # Check if the data is not the whole dictionary
        if path and data != self.yaml_data:
            self.editor.insert(tk.END, yaml.dump({path[-1]: data}, sort_keys=False))

    def get_nested_value(self, data, path):
        for key in path:
            data = data.get(key, {})
        return data

    # def save_file(self):
    #     if self.yaml_data and self.current_selection:
    #         path = self.get_path(self.current_selection)
    #         new_data = yaml.safe_load(self.editor.get("1.0", tk.END))
    #         key, value = list(new_data.items())[0]
    #         self.set_nested_value(self.yaml_data, path[:-1] + [key], value)

    def set_nested_value(self, data, path, value):
        for key in path[:-1]:
            data = data.setdefault(key, {})
        data[path[-1]] = value

    # def export_file(self):
    #     file_path = filedialog.asksaveasfilename(defaultextension=".yaml", filetypes=[("YAML files", "*.yaml")])
    #     if file_path and self.yaml_data:
    #         with open(file_path, 'w') as file:
    #             yaml.dump(self.yaml_data, file, sort_keys=False)
    #         messagebox.showinfo("Info", "File exported successfully.")
    def save_file(self):
        if self.yaml_data and hasattr(self, 'current_file'):
            with open(self.current_file, 'w') as file:
                yaml.dump(self.yaml_data, file, sort_keys=False)
            messagebox.showinfo("Info", "File saved successfully.")
        else:
            self.export_file_as()  # Prompt for a file name if the current data isn't associated with a file

    def export_file_as(self):
        file_path = filedialog.asksaveasfilename(defaultextension=".yaml", filetypes=[("YAML files", "*.yaml")])
        if file_path and self.yaml_data:
            with open(file_path, 'w') as file:
                yaml.dump(self.yaml_data, file, sort_keys=False)
            messagebox.showinfo("Info", "File exported successfully.")

    def collapse_all(self):
        for item in self.tree.get_children():
            self.tree.item(item, open=False)
    def open_all(self):
        for item in self.tree.get_children():
            self.tree.item(item, open=True)
    # def export_file_as(self):
    #     file_path = filedialog.asksaveasfilename(defaultextension=".yaml", filetypes=[("YAML files", "*.yaml")])
    #     if file_path and self.yaml_data:
    #         with open(file_path, 'w') as file:
    #             yaml.dump(self.yaml_data, file, sort_keys=False)
    #         messagebox.showinfo("Info", "File exported successfully.")

if __name__ == "__main__":
    root = tk.Tk()
    gui = YAMLGUI(root)
    root.mainloop()
