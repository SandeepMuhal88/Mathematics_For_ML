import numpy as np
import matplotlib.pyplot as plt

# Generate synthetic data
np.random.seed(42)
X = 2 * np.random.rand(100, 1)  # Feature
y = 4 + 3 * X + np.random.randn(100, 1)  # Target with some noise

# Initialize parameters
w = np.random.randn(1, 1)
b = np.random.randn(1)
alpha = 0.1  # Learning rate
epochs = 1000  # Number of iterations

# Gradient Descent
m = len(X)
for epoch in range(epochs):
    y_pred = X.dot(w) + b
    error = y_pred - y
    w_grad = (2/m) * X.T.dot(error)
    b_grad = (2/m) * np.sum(error)
    
    # Update weights
    w -= alpha * w_grad
    b -= alpha * b_grad

    # Logging
    if epoch % 100 == 0:
        loss = np.mean(error**2)
        print(f"Epoch {epoch}: Loss = {loss}")

print(f"Final parameters: w = {w[0][0]}, b = {b[0]}")

# SGD Implementation
epochs = 1000
for epoch in range(epochs):
    for i in range(m):  # Iterate over each sample
        idx = np.random.randint(m)  # Pick a random sample
        xi = X[idx:idx+1]
        yi = y[idx:idx+1]
        
        y_pred = xi.dot(w) + b
        error = y_pred - yi
        w_grad = 2 * xi.T.dot(error)
        b_grad = 2 * error
        
        # Update weights
        w -= alpha * w_grad
        b -= alpha * b_grad

print(f"Final parameters after SGD: w = {w[0][0]}, b = {b[0]}")
