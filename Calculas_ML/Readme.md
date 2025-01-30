# Mathematics for Machine Learning: Calculus, Gradient Descent, and Stochastic Gradient Descent

This guide covers the essential mathematical concepts used in Machine Learning, including:

1. **Calculus Fundamentals**
2. **Gradient Descent Algorithm**
3. **Stochastic Gradient Descent (SGD)**
4. **Python Implementation**

---

## 1. 📌 Calculus Fundamentals for ML
Calculus helps optimize functions, particularly in minimizing loss functions in ML.

### 🔹 Key Concepts
- **Derivatives**: Measures the rate of change of a function.
  
  \[ f'(x) = \lim_{\Delta x \to 0} \frac{f(x+\Delta x) - f(x)}{\Delta x} \]

- **Partial Derivatives**: Used for functions with multiple variables, keeping others constant.
  
  \[ \frac{\partial f}{\partial x} \]

- **Gradient**: A vector of partial derivatives indicating the direction of steepest ascent.
  
  \[ \nabla f = \left( \frac{\partial f}{\partial x_1}, \frac{\partial f}{\partial x_2}, ..., \frac{\partial f}{\partial x_n} \right) \]

---

## 2. 🚀 Gradient Descent Algorithm
Gradient Descent is an optimization algorithm used to minimize loss functions by adjusting weights iteratively.

### **Steps**
1. Initialize weights \( w \) randomly.
2. Compute the loss function \( J(w) \).
3. Compute the gradient \( \nabla J(w) \).
4. Update weights using:
   
   \[ w := w - \alpha \nabla J(w) \]
   
   where \( \alpha \) is the learning rate.
5. Repeat until convergence.

---

## 3. 🔥 Stochastic Gradient Descent (SGD)
SGD updates weights after each data sample, making it efficient for large datasets.

### **Key Differences: GD vs. SGD**
| Feature | Gradient Descent (GD) | Stochastic Gradient Descent (SGD) |
|---------|-----------------------|----------------------------------|
| Update Frequency | After full dataset | After each sample |
| Convergence | Smooth | Noisy, but faster |
| Computational Cost | High for large datasets | Lower for large datasets |

### **SGD Update Rule**
\[ w := w - \alpha \nabla J(w, x_i) \]
where \( x_i \) is a single training example.

---

## 4. 📝 Implementing Gradient Descent & SGD in Python

### **Gradient Descent (Linear Regression Example)**
```python
import numpy as np
import matplotlib.pyplot as plt

# Generate synthetic data
np.random.seed(42)
X = 2 * np.random.rand(100, 1)
y = 4 + 3 * X + np.random.randn(100, 1)

# Initialize parameters
w = np.random.randn(1, 1)
b = np.random.randn(1)
alpha = 0.1  # Learning rate
epochs = 1000

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

    if epoch % 100 == 0:
        loss = np.mean(error**2)
        print(f"Epoch {epoch}: Loss = {loss}")

print(f"Final parameters: w = {w[0][0]}, b = {b[0]}")
```

### **Stochastic Gradient Descent (SGD)**
```python
# SGD Implementation
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
```

---

## 📌 Summary
- **Calculus** helps find gradients, crucial for optimization.
- **Gradient Descent** minimizes loss functions efficiently.
- **SGD** is faster and scalable for large datasets.
- **Python implementation** demonstrates these concepts in action.
