from ._file import store_object


def store_png(filepath: str, filekey: str):
    """Store png file."""
    if not filepath.endswith("png"):
        raise ValueError()
    store_object(filepath, filekey)
