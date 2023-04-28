import math

import numba
import numpy as np
import matplotlib.pyplot as plt

# Project code
from bayesian_optimizer import BayesianOptimizer, AcquisitionFunc


@numba.jit
def test_function(x, sigma=0.1):
    # Add some noise to make this function hard to optimize
    noise = np.random.normal(loc=0, scale=sigma)
    # return x**2
    return -1 * (x**2 * np.sin(5 * np.pi * x)**6.0)


# Run the optimizer
bounds = [(0, 1)]
optimizer = BayesianOptimizer(
    lambda x1: test_function(x1), 30, 7, bounds, AcquisitionFunc.EI, debug=True)
optimal = optimizer.minimize()

# grid-based sample of the domain [0,1]
X = np.linspace(0, 1, 500)
# sample the domain without noise
y = [test_function(x) for x in X]
# sample the domain with noise
ynoise = [test_function(x) for x in X]
# Find best result
ix = np.argmin(y)

# Plot the function and data
fig = plt.figure(figsize=(6, 6))
# Plot the optimum
plt.plot(X[ix], y[ix], "*r", label="Optimal")
# plot the points without noise
plt.plot(X, y, label="Truth")
# Plot the optimzer's initialization points
opt_x_init = optimizer.x_init.reshape(-1, 1)
plt.plot(opt_x_init, test_function(opt_x_init), "ok", ms=4, alpha=0.7, label="Initialization")
# Plot the optimizer's evaluation points
opt_x = optimizer.x_eval.reshape(-1, 1)
plt.plot(opt_x, test_function(opt_x), "or", ms=4, alpha=1, label="Evaluation")
plt.legend()
plt.show()

print(opt_x)
print(optimizer.x_eval)
