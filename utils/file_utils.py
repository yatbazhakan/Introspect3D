import os
def check_and_create_folder(path: str):
    """Checks if a folder exists and creates it if not."""
    if not os.path.exists(path):
        os.makedirs(path)