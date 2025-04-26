# File: pyneuro/core/optim/optimizer.py
class Optimizer:
    """Base class for all optimizers in PyNeuro"""
    
    def step(self):
        """Perform a single optimization step"""
        raise NotImplementedError
    
    def zero_grad(self):
        """Zero out all gradients"""
        pass
