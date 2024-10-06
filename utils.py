import os

def get_leaf_directories(directory):
    leaf_directories = []

    for root, dirs, files in os.walk(directory):
        if not dirs:
            leaf_directories.append(root)

    return leaf_directories

def create_output_directory(original_directory, base_output_dir):
    relative_path = os.path.relpath(original_directory, start='tipster')
    output_dir = os.path.join(base_output_dir, relative_path)
    os.makedirs(output_dir, exist_ok=True)
    return output_dir
