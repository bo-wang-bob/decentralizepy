import torch

# 检查 CUDA 是否可用
if torch.cuda.is_available():
    print(f"CUDA is available. Device: {torch.cuda.get_device_name(0)}")
else:
    print("CUDA is not available. Using CPU.")
