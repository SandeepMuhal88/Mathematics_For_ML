# import numpy as np
# # from sklearn.linear_model import Ridge

# # Generate synthetic data
# X = np.random.randn(100, 2)
# y = 3 * X[:, 0] + 2 * X[:, 1] + np.random.randn(100)

# # Solve using Ridge Regression (convex optimization with L2 penalty)
# # model = Ridge(alpha=1.0)
# model.fit(X, y)
# print(f"Weights: {model.coef_}")