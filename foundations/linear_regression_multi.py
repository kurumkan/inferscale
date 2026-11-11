import numpy as np
import matplotlib.pyplot as plt
import csv

SIZE = 1000
ITERATIONS = 5000
rng = np.random.default_rng(42)
x = rng.random((SIZE, 2))
eps = rng.normal(loc=0.0, scale=0.1, size=SIZE)
ones_column = np.ones((SIZE, 1))
params_true = np.array([
    [1],
    [2],
    [-3]
])
y = params_true[1, 0] * x[:, 0].flatten() + params_true[2, 0] * x[:, 1].flatten() + params_true[0, 0] * ones_column.flatten() + eps

params = np.ones((3, 1))
params[0] = 0
learning_rate = 0.01
x_matrix = np.hstack([ones_column, x])

loss_history = []
for i in range(ITERATIONS):
    y_predicted = x_matrix @ params
    diff = (y_predicted.flatten() - y).reshape(-1, 1)

    grad_w = 2 * x.T @ diff / len(x)
    grad_b = np.mean(2 * diff)

    params[0] = params[0] - learning_rate * grad_b
    params[1:] = params[1: ] - learning_rate * grad_w

    mse = np.mean(diff ** 2) 
    loss_history.append(mse)
    
    if i % 100 == 0:
        print("[", i , "] mse =", mse)

print("results:")
print(params)
print("mse", mse)

# Visualisation
fig, ax = plt.subplots()
ax.plot(np.arange(ITERATIONS), np.array(loss_history), color="blue")
ax.set_xlabel("iteration")
ax.set_ylabel("MSE")
ax.set_title("MSE progress")
fig.savefig("benchmarks/linear_regression_loss.png")
plt.show()

assert loss_history[0] > loss_history[-1]
assert np.allclose(params, params_true, atol=0.05), f"Wrong weights: {params.flatten()}"
assert np.all(np.isfinite(loss_history)), "Infinite loss value found"

with open("benchmarks/linear_regression_loss.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["step", "mse"])
    writer.writerows(enumerate(loss_history))