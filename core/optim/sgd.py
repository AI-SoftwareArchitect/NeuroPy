import torch
from pyneuro.core.optim.optimizer import Optimizer

class SGD(Optimizer):
    """
    Stochastic Gradient Descent optimizer
    """
    def __init__(self, parameters, lr=0.01):
        self.parameters = list(parameters)
        self.lr = lr
    
    def step(self):
        """Perform a single optimization step"""
        for param in self.parameters:
            if param.grad is not None:
                param.data -= self.lr * param.grad
    
    def zero_grad(self):
        """Zero out all gradients"""
        for param in self.parameters:
            if param.grad is not None:
                param.grad.zero_()
