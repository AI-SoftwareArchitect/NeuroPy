
# File: pyneuro/visualization/metrics_plotter.py
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import torch
from sklearn.metrics import confusion_matrix

def plot_confusion_matrix(true_labels, pred_labels, class_names=None, normalize=True):
    """
    Plot confusion matrix
    
    Args:
        true_labels: True labels
        pred_labels: Predicted labels
        class_names: Names of classes (optional)
        normalize: Whether to normalize the confusion matrix (default: True)
    """
    # Convert tensors to numpy arrays if needed
    if isinstance(true_labels, torch.Tensor):
        true_labels = true_labels.cpu().numpy()
    if isinstance(pred_labels, torch.Tensor):
        pred_labels = pred_labels.cpu().numpy()
    
    # Compute confusion matrix
    cm = confusion_matrix(true_labels, pred_labels)
    
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        fmt = '.2f'
    else:
        fmt = 'd'
    
    # Plot
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt=fmt, cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.show()