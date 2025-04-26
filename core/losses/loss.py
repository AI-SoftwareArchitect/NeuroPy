
# File: pyneuro/core/losses/loss.py
import torch
import torch.nn.functional as F

class CrossEntropyLoss:
    """
    Cross entropy loss implementation
    """
    def __init__(self):
        self.softmax_preds = None
        self.targets = None
    
    def forward(self, predictions, targets):
        """
        Forward pass for cross entropy loss
        
        Args:
            predictions: Model predictions (logits)
            targets: Target labels
        
        Returns:
            Loss value
        """
        # Apply softmax
        exp_preds = torch.exp(predictions - torch.max(predictions, dim=1, keepdim=True)[0])
        self.softmax_preds = exp_preds / torch.sum(exp_preds, dim=1, keepdim=True)
        
        # Convert targets to one-hot if needed
        if targets.dim() == 1:
            self.targets = torch.zeros_like(self.softmax_preds).scatter_(
                1, targets.unsqueeze(1), 1.0
            )
        else:
            self.targets = targets
        
        # Calculate cross entropy loss
        log_likelihood = -torch.log(self.softmax_preds + 1e-15)  # numerical stability
        loss = torch.sum(self.targets * log_likelihood) / predictions.size(0)
        return loss
    
    def __call__(self, predictions, targets):
        """
        Calculate loss
        
        Args:
            predictions: Model predictions (logits)
            targets: Target labels
        
        Returns:
            Loss value
        """
        return self.forward(predictions, targets)