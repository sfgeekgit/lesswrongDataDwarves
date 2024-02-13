import torch

# Check if CUDA is available
if torch.cuda.is_available():
    print("CUDA is available. PyTorch will use the GPU:", torch.cuda.get_device_name(0))
else:
    print("CUDA is not available. PyTorch will use the CPU.")
