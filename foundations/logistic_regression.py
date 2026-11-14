import numpy as np


def sigmoid(z): # turn valuation into probability
    return 1.0 / (1.0 + np.exp(-z))

def binary_cross_entropy(y, p): # loss function
    loss = -(y @ np.log(p) + (1 - y) @ np.log(1 - p))
    return loss / len(p)


learning_rate = 0.1

y = np.array([1., 0., 1.])
p = np.array([0.9, 0.2, 0.7])
result = binary_cross_entropy(y, p)

rng = np.random.default_rng(42)
x = rng.standard_normal((1000, 2))
y = (2 * x[:, 0] - x[:, 1] + 0.5 > 0).astype(float) # (1000,)
y = y.reshape(-1, 1) # (1000, 1)
w = np.zeros((2, 1)) 
b = np.zeros((1, 1))

loss_history = []

for iteration in range(0, 1000):
    z = x @ w + b # (1000, 1)
    p = sigmoid(z) # (1000, 1)
    bce = -(y.T @ np.log(p) + (1 - y.T) @ np.log(1 - p)) / len(x) # average BCE
    grad_w = x.T @ (p - y) / len(x) # average grad (2, 1)
    grad_b = np.average(p - y) # average grad scalar
    w = w - learning_rate * grad_w
    b = b - learning_rate * grad_b
    loss_history.append(bce)

z = x @ w + b
p = sigmoid(z)
y_pred = (p >= 0.5).astype(float)
accuracy = np.mean(y == y_pred)

print("loss start", loss_history[0])
print("loss end", loss_history[-1])
print("final weight", w)
print("final bias", b)
print("accuracy", accuracy)

