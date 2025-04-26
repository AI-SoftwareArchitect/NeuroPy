import torch
import torch.nn as nn
import torch.nn.functional as F
from pyneuro.core.layers.grow_linear import GrowLinear

class GrowNet(nn.Module):
    """
    A neural network that can grow in size during training
    """
    def __init__(self, input_size, hidden_size, output_size):
        super(GrowNet, self).__init__()
        self.fc1 = GrowLinear(input_size, hidden_size)
        self.fc2 = GrowLinear(hidden_size, output_size)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    
    def grow_brain_cells(self, new_neurons):
        """
        Add new_neurons to the hidden layer of the network
        """
        # Grow the first layer (add output neurons)
        self.fc1.grow(new_neurons)
        
        # For the second layer, we need to add input connections
        # for the new neurons added to the first layer
        old_out, old_in = self.fc2.weight.shape
        device = self.fc2.weight.device
        
        # Create new weights for the additional inputs
        new_inputs = torch.randn(old_out, new_neurons, device=device) * 0.01
        
        # Concatenate with existing weights
        self.fc2.weight = nn.Parameter(torch.cat([self.fc2.weight, new_inputs], dim=1))
        
        # Update the in_features count for the second layer
        self.fc2.in_features += new_neurons
        
        return self.fc1.out_features