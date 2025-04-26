# File: pyneuro/config.py
"""
Configuration values for PyNeuro
"""

# Default model parameters
INPUT_SIZE = 784  # MNIST image size: 28x28
HIDDEN_SIZE = 64
OUTPUT_SIZE = 10  # 10 digits

# Training parameters
LEARNING_RATE = 0.01
BATCH_SIZE = 32
EPOCHS = 10

# Growth parameters
GROW_EVERY = 2  # Grow the model every N epochs
GROW_AMOUNT = 8  # Number of neurons to add each time