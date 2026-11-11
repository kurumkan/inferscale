import numpy as np

SIZE = 1000
ITERATIONS = 5000
rng = np.random.default_rng(42)
x = rng.random((SIZE, 2))
eps = rng.normal(loc=0.0, scale=0.1, size=SIZE)
ones_column = np.ones((SIZE, 1))

y = 2 * x[:, 0].flatten() - 3 * x[:, 1].flatten() + ones_column.flatten() + eps

params = np.ones((3, 1))
params[0] = 0
learning_rate = 0.01
x_matrix = np.hstack([ones_column, x])

for i in range(ITERATIONS):
    y_predicted = x_matrix @ params
    diff = (y_predicted.flatten() - y).reshape(-1, 1)

    grad_w = 2 * x.T @ diff / len(x)
    grad_b = np.mean(2 * diff)

    params[0] = params[0] - learning_rate * grad_b
    params[1:] = params[1: ] - learning_rate * grad_w

    mse = np.mean(diff ** 2) 
    
    if i % 100 == 0:
        print("[", i , "] mse =", mse)

print("results:")
print(params)
print("mse", mse)

