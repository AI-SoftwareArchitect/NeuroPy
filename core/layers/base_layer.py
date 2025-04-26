import torch
import torch.nn as nn

class BaseLayer(nn.Module):
    """Base class for all layers in PyNeuro"""
    
    def __init__(self):
        super(BaseLayer, self).__init__()
    
    def forward(self, input):
        """Forward pass through the layer"""
        raise NotImplementedError
    
    def parameters(self):
        """Return the parameters of the layer"""
        return super().parameters()