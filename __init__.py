"""
PyNeuro: A lightweight neural network framework with dynamic growth capabilities
"""

__version__ = '0.1.0'

# Import key components to make them available at the top level
from pyneuro.core.models.grow_net import GrowNet
from pyneuro.core.layers.grow_linear import GrowLinear
from pyneuro.core.trainers.trainer import Trainer
from pyneuro.core.losses.loss import CrossEntropyLoss
