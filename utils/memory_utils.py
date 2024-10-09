import torch
import gc

def clear_memory():
    # Clear GPU memory if using PyTorch
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Garbage collection to free up RAM
    gc.collect()