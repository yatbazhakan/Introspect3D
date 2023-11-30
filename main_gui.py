import tkinter as tk
from tkinter import filedialog, simpledialog
import subprocess
from definitions import CONFIG_DIR
def run_in_tmux(config_path, operator_type, session_name):
    # Check if the session already exists
    existing_session = subprocess.run(["tmux", "has-session", "-t", session_name], stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    if existing_session.returncode == 0:
        # If session exists, split the current window
        subprocess.run(["tmux", "split-window", "-v", "-t", session_name])
    else:
        # If session does not exist, create a new session
        subprocess.run(["tmux", "new-session", "-d", "-s", session_name])

    # Set up environment and directory
    setup_commands = [
        "conda activate openmmlab2",
        "export PATH=/usr/local/cuda-11.7/bin${PATH:+:${PATH}}",
        "export LD_LIBRARY_PATH=/usr/local/cuda-11.7/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}",
        "cd /mnt/ssd2/Introspect3D"
    ]

    for cmd in setup_commands:
        subprocess.run(["tmux", "send-keys", "-t", session_name, cmd, "C-m"])

    # Run the Python script
    python_cmd = f"python main.py -c {config_path} -o {operator_type}"
    subprocess.run(["tmux", "send-keys", "-t", session_name, python_cmd, "C-m"])


def get_session_name():
    return simpledialog.askstring("Session Name", "Enter the tmux session name:")

def run():
    session_name = session_name_entry.get()
    if session_name:
        run_in_tmux(config_path_entry.get(), operator_var.get(), session_name)

def browse_file():
    filename = filedialog.askopenfilename(initialdir=CONFIG_DIR, title="Select Config File",
                                          filetypes=(("YAML files", "*.yaml"), ("all files", "*.*")))
    config_path_entry.delete(0, tk.END)
    config_path_entry.insert(0, filename)

app = tk.Tk()
app.title("Experiment Runner")
app.bind('<Return>', (lambda event: run()))
app.bind('<Escape>', (lambda event: app.destroy()))
# Config file selection
tk.Label(app, text="Config File:").grid(row=0, column=0)
config_path_entry = tk.Entry(app, width=50)
config_path_entry.grid(row=0, column=1)
tk.Button(app, text="Browse", command=browse_file).grid(row=0, column=2)

# Operator selection
tk.Label(app, text="Operator:").grid(row=1, column=0)
operator_var = tk.StringVar(app)
operator_var.set("1")  # default value
operator_menu = tk.OptionMenu(app, operator_var, "0", "1", "2")
operator_menu.grid(row=1, column=1)
# Session name entry
tk.Label(app, text="Session Name:").grid(row=2, column=0)
session_name_entry = tk.Entry(app, width=20)
session_name_entry.grid(row=2, column=1)
session_name_entry.insert(0, "default_session")  
# Run button
run_button = tk.Button(app, text="Run", command=run)
run_button.grid(row=3, columnspan=3)

app.mainloop()