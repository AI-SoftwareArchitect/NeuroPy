# File: pyneuro/core/trainers/trainer.py
import torch
import torch.optim as optim
import torch.nn.functional as F
from tqdm import tqdm

class Trainer:
    """
    Class for training neural network models
    """
    def __init__(self, model, loss_fn, lr=0.01):
        """
        Initialize the trainer
        
        Args:
            model: The neural network model to train
            loss_fn: The loss function to use
            lr: Learning rate for the optimizer
        """
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optim.Adam(model.parameters(), lr=lr)
    
    def fit(self, X_train, y_train, X_test=None, y_test=None, epochs=10, 
            batch_size=64, grow_every=None, grow_amount=0):
        """
        Train the model
        
        Args:
            X_train: Training data
            y_train: Training labels
            X_test: Test data (optional)
            y_test: Test labels (optional)
            epochs: Number of epochs to train for
            batch_size: Batch size for training
            grow_every: Grow the model every N epochs (None = no growth)
            grow_amount: Number of neurons to add when growing
        """
        # Training loop
        num_samples = len(X_train)
        
        for epoch in range(epochs):
            self.model.train()
            running_loss = 0.0
            
            # Create batch indices
            indices = torch.randperm(num_samples)
            
            for start_idx in range(0, num_samples, batch_size):
                # Get mini-batch
                batch_indices = indices[start_idx:start_idx + batch_size]
                X_batch = X_train[batch_indices]
                y_batch = y_train[batch_indices]
                
                # Forward pass
                self.optimizer.zero_grad()
                outputs = self.model(X_batch)
                
                # Calculate loss
                if hasattr(self.loss_fn, '__call__'):
                    loss = self.loss_fn(outputs, y_batch)
                else:
                    loss = F.cross_entropy(outputs, y_batch)
                
                # Backward pass and optimize
                loss.backward()
                self.optimizer.step()
                
                running_loss += loss.item()
            
            # Calculate average loss for the epoch
            epoch_loss = running_loss / (num_samples / batch_size)
            
            # Evaluate on test set if provided
            test_loss = None
            if X_test is not None and y_test is not None:
                test_loss = self.evaluate(X_test, y_test)
                print(f"Epoch [{epoch+1}/{epochs}] - Train Loss: {epoch_loss:.4f} - Test Loss: {test_loss:.4f}")
            else:
                print(f"Epoch [{epoch+1}/{epochs}] - Train Loss: {epoch_loss:.4f}")
            
            # Grow the model if specified
            if grow_every is not None and epoch > 0 and (epoch + 1) % grow_every == 0 and grow_amount > 0:
                if hasattr(self.model, 'grow_brain_cells'):
                    self.model.grow_brain_cells(grow_amount)
                    print(f"🧠 +{grow_amount} neurons added! New hidden layer size: {self.model.fc1.out_features}")
    
    def evaluate(self, X, y, batch_size=64):
        """
        Evaluate the model on test data
        
        Args:
            X: Test data
            y: Test labels
            batch_size: Batch size for evaluation
        
        Returns:
            Average loss on the test data
        """
        self.model.eval()
        total_loss = 0.0
        num_samples = len(X)
        
        with torch.no_grad():
            for start_idx in range(0, num_samples, batch_size):
                end_idx = min(start_idx + batch_size, num_samples)
                X_batch = X[start_idx:end_idx]
                y_batch = y[start_idx:end_idx]
                
                outputs = self.model(X_batch)
                
                if hasattr(self.loss_fn, '__call__'):
                    loss = self.loss_fn(outputs, y_batch)
                else:
                    loss = F.cross_entropy(outputs, y_batch)
                
                total_loss += loss.item() * (end_idx - start_idx)
        
        return total_loss / num_samples