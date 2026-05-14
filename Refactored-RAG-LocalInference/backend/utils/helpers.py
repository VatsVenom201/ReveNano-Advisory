import os
import hashlib
import gc

def get_file_hash(file_path):
    """Generate MD5 hash for a file to track changes/uniqueness."""
    hasher = hashlib.md5()
    with open(file_path, 'rb') as f:
        buf = f.read()
        hasher.update(buf)
    return hasher.hexdigest()

def clear_memory():
    """Force garbage collection to free up RAM."""
    gc.collect()

def get_supported_files(directory):
    """Walk directory and return list of supported file paths."""
    supported_extensions = {".pdf", ".docx", ".txt", ".xls", ".xlsx"}
    file_paths = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if os.path.splitext(file.lower())[1] in supported_extensions:
                file_paths.append(os.path.join(root, file))
    return file_paths
