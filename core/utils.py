import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split

def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)

def cross_entropy_loss(predictions, targets):
    m = targets.shape[0]
    p = predictions
    log_likelihood = -np.log(p[range(m), targets] + 1e-15)
    loss = np.sum(log_likelihood) / m
    return loss

def cross_entropy_grad(predictions, targets):
    m = targets.shape[0]
    grad = predictions.copy()
    grad[range(m), targets] -= 1
    grad = grad / m
    return grad

def load_data():
    # MNIST veri setini yüklemek
    mnist = fetch_openml("mnist_784", version=1)
    X = mnist["data"].values
    y = mnist["target"].astype(int)

    # Veriyi normalize et
    X = X / 255.0

    # Eğitim ve test verisine ayır
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    return X_train, y_train, X_test, y_test