import numpy as np
from scipy.stats import norm
from sympy import symbols, diff

# 1. Linear Algebra: Matrix Operations
A = np.array([[2, 3], [4, 5]])
B = np.array([[1, 2], [3, 4]])

# Matrix Addition
A_plus_B = A + B

# Matrix Subtraction
A_minus_B = A - B

# Matrix Multiplication
A_mul_B = np.dot(A, B)

# Inverse of A
A_inv = np.linalg.inv(A)

# 2. Probability: Conditional Probability
red_balls = 3
blue_balls = 5
total_balls = red_balls + blue_balls

# P(R1) = Probability of drawing a red ball first
P_R1 = red_balls / total_balls

# P(R2|R1) = Probability of drawing a red ball second given first was red
P_R2_given_R1 = (red_balls - 1) / (total_balls - 1)

# Final probability
P_R2_if_R1 = P_R1 * P_R2_given_R1

# 3. Statistics: Mean, Variance, and Standard Deviation
X = np.array([4, 8, 15, 16, 23, 42])
mean_X = np.mean(X)
variance_X = np.var(X, ddof=0)
std_dev_X = np.std(X, ddof=0)

# 4. Calculus: Derivative Calculation
x = symbols('x')
f = 3*x**2 + 5*x + 2
f_derivative = diff(f, x)
value_at_2 = f_derivative.subs(x, 2)

# Display results
results = {
    "Matrix Addition": A_plus_B,
    "Matrix Subtraction": A_minus_B,
    "Matrix Multiplication": A_mul_B,
    "Inverse of A": A_inv,
    "Conditional Probability": P_R2_if_R1,
    "Mean": mean_X,
    "Variance": variance_X,
    "Standard Deviation": std_dev_X,
    "Derivative at x=2": value_at_2
}

for key, value in results.items():
    print(f"{key}:\n{value}\n")
