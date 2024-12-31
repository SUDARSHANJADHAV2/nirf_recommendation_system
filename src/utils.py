import joblib

def save_object(obj, file_path):
    """Save an object to a file."""
    joblib.dump(obj, file_path)

def load_object(file_path):
    """Load an object from a file."""
    return joblib.load(file_path)