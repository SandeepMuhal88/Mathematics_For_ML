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
