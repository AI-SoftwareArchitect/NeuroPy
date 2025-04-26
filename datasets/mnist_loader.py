# File: pyneuro/datasets/mnist_loader.py
import torch
from torchvision import datasets, transforms

def load_mnist(root='./data', train=True, download=True, batch_size=32):
    """
    Load the MNIST dataset
    
    Args:
        root: Root directory for storing the dataset
        train: Whether to load the training set (True) or test set (False)
        download: Whether to download the dataset if not found
        batch_size: Batch size for data loader
    
    Returns:
        DataLoader for the MNIST dataset
    """
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    dataset = datasets.MNIST(
        root=root,
        train=train,
        download=download,
        transform=transform
    )
    
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=train
    )
    
    return loader