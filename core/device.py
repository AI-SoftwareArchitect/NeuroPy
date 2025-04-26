# File: pyneuro/device.py
import torch

def get_device():
    """Get the device to use (CPU or GPU)"""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")