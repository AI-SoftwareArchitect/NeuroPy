import torch
import torch.nn as nn
import torch.nn.functional as F

class GrowLinear(nn.Module):
    """
    A linear layer that can grow in size during training
    """
    def __init__(self, in_features, out_features):
        super(GrowLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.randn(out_features, in_features) * 0.01)
        self.bias = nn.Parameter(torch.zeros(out_features))
    
    def forward(self, x):
        return F.linear(x, self.weight, self.bias)
    
    def grow(self, new_neurons):
        """
        Add new_neurons to the output dimension of the layer
        """
        old_out, in_features = self.weight.shape
        device = self.weight.device
        
        # Create new weights and bias for the additional neurons
        new_weights = torch.randn(new_neurons, in_features, device=device) * 0.01
        new_bias = torch.zeros(new_neurons, device=device)
        
        # Concatenate with existing parameters
        self.weight = nn.Parameter(torch.cat([self.weight, new_weights], dim=0))
        self.bias = nn.Parameter(torch.cat([self.bias, new_bias], dim=0))
        
        # Update the out_features count
        self.out_features += new_neurons
        
        return self.out_features