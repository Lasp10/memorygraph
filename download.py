import kagglehub

# Download latest version
path = kagglehub.dataset_download("anku5hk/5-faces-dataset")

print("Path to dataset files:", path)