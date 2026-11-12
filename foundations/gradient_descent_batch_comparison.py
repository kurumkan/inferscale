import numpy as np
import time

SIZE = 1000

rng = np.random.default_rng(42)
x = rng.random((SIZE, 2))
eps = rng.normal(loc=0.0, scale=0.1, size=SIZE)
ones_column = np.ones((SIZE, 1))
params_true = np.array([
    [1],
    [2],
    [-3]
])

# (1000,)
y = params_true[1, 0] * x[:, 0].flatten() + params_true[2, 0] * x[:, 1].flatten() + params_true[0, 0] * ones_column.flatten() + eps

def train(x, y, batch_size, epochs, learning_rate):
    ones_column = np.ones((len(x), 1))
    params = np.ones((3, 1)) # (3, 1)
    params[0] = 0
    x_matrix = np.hstack([ones_column, x]) # (1000, 3)

    loss_history = []
    for epoch in range(epochs):         # 30 эпох
        indices = rng.permutation(len(x))   # перемешиваем индексы  

        for start in range(0, len(y), batch_size):  # 32 итерации за эпоху
            batch_indices = indices[start:start + batch_size]
            x_batch = x[batch_indices]
            y_batch = y[batch_indices]

            y_predicted = x_matrix[batch_indices] @ params # (32, 1)
            diff = (y_predicted.flatten() - y_batch).reshape(-1, 1) # (32, 1)
    
            grad_w = 2 * x_batch.T @ diff / len(diff)
            grad_b = np.mean(2 * diff)
    
            params[0] = params[0] - learning_rate * grad_b
            params[1:] = params[1: ] - learning_rate * grad_w

        y_predicted = x_matrix @ params
        diff = (y_predicted.flatten() - y).reshape(-1, 1)
        mse = np.mean(diff ** 2) 
        loss_history.append(float(mse))

    return params, loss_history
           
for batch_size in [1000, 32, 1]:
    for learning_rate in [0.01, 0.1, 0.5, 1.0]:
        start = time.perf_counter()

        weights, history = train(x, y, batch_size, 1000, learning_rate)

        elapsed = time.perf_counter() - start

        print(
            "batch_size:", batch_size,
            "final MSE:", history[-1],
            "seconds:", elapsed
        )
