import numpy as np

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

# (N,)
y = params_true[1, 0] * x[:, 0].flatten() + params_true[2, 0] * x[:, 1].flatten() + params_true[0, 0] * ones_column.flatten() + eps

def train(x_full, y_full, batch_size, epochs, learning_rate, mse_limit, train_ratio, l2_lambda):
    rng = np.random.default_rng(42)
    indices = rng.permutation(len(x_full))   # перемешиваем индексы  
    x_shuffled = x_full[indices]
    y_shuffled = y_full[indices]

    train_size = int(len(x_shuffled) * train_ratio)

    x = x_shuffled[:train_size]
    y = y_shuffled[:train_size]

    x_val = x_shuffled[train_size:]
    y_val = y_shuffled[train_size:]

    params = np.ones((3, 1)) # (3, 1)
    params[0] = 0
    x_matrix = np.hstack([np.ones((len(x), 1)), x]) # (N, 3)
    x_matrix_val = np.hstack([np.ones((len(x_val), 1)), x_val]) # (N, 3)
    updates = 0   

    loss_history = []
    val_loss_history = []
    for epoch in range(epochs):      
        indices = rng.permutation(len(x))   # перемешиваем индексы  

        for start in range(0, len(y), batch_size):  # 32 итерации за эпоху
            batch_indices = indices[start:start + batch_size]
            x_batch = x[batch_indices]
            y_batch = y[batch_indices]

            y_predicted = x_matrix[batch_indices] @ params # (32, 1)
            diff = (y_predicted.flatten() - y_batch).reshape(-1, 1) # (32, 1)

            penalty = 2 * l2_lambda * params[1:] # (2, 1)
            grad_w = 2 * x_batch.T @ diff / len(diff) + penalty (2, 1)
            grad_b = np.mean(2 * diff)
    
            params[0] = params[0] - learning_rate * grad_b
            params[1:] = params[1: ] - learning_rate * grad_w
            updates += 1

        y_predicted = x_matrix @ params
        diff = (y_predicted.flatten() - y).reshape(-1, 1)
        train_mse = np.mean(diff ** 2) 
        loss_history.append(float(train_mse))

        y_val_predicted = x_matrix_val @ params
        diff_val = (y_val_predicted.flatten() - y_val).reshape(-1, 1)
        val_mse = np.mean(diff_val ** 2) 
        val_loss_history.append(float(val_mse))

    return loss_history, val_loss_history
           
for lambda_l2 in [0, 0.1]:
    batch_size = 32
    learning_rate = 0.1
    
    history, val_loss_history = train(x, y, batch_size, 200, learning_rate, 0.0105, 0.8, lambda_l2)

    print(
        "MSE first", history[0],
        "MSE last:", history[-1],
        "val MSE first", val_loss_history[0],
        "val MSE last:", val_loss_history[-1]
    )
        