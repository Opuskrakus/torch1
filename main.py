import torch

# Check if CUDA (GPU support) is available
print("CUDA available:", torch.cuda.is_available())

# Print the current device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# Optional: Get the name of the GPU (if available)
if torch.cuda.is_available():
    print("GPU name:", torch.cuda.get_device_name(0))