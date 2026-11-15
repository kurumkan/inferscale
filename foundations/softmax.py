import numpy as np

# softmax algorithm
# задача - оперделить категорию

# Y [N]             ← 1 правильная категория для каждого кейса - содержит индексы правильных категорий для каждого кейса
# X [N, D]          ← исходные признаки - у каждого z - свой набор весов
#     ↓
# Z = X @ W + b     ← линейный слой - вычисляем z - оценка вероятности
#     ↓
# Z [N, C]          ← logits
#     ↓
# P = softmax(Z)    ← Softmax - переводим z - в вероятности
#     ↓
# P [N, C]          ← вероятности классов


def softmax(z): # finding probabilities    
    exp_z = np.exp(z)
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)

def stable_softmax(z):
    max_z = np.max(z, axis=1, keepdims=True) 
    exp_z = np.exp(z - max_z) # extract max to avoid float overflow
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)

def cross_entropy_batch(p, y):
    loss = np.sum(- np.log(p[np.arange(len(p)), y])) / len(p)
    return loss

def cross_entropy_from_logits(z, y):
    m = np.max(z, axis=1, keepdims=True) # max to avoid float overflow or zeroing issue
    return np.mean(m + (np.log(np.sum(np.exp(z - m), axis=1)) - z[np.arange(len(z)), y]).reshape(-1,1))

def softmax_cross_entropy_gradient(p, y):
    return (p - one_hot(y, p.shape[1])) / len(p)

def one_hot(m, number_of_classes):
    return np.eye(number_of_classes)[m]

def forward(x, w, b):
    z = x @ w + b
    p = stable_softmax(z)
    return z, p

def backward(x, p, y):
    # x: [N, D]
    # p: [N, C]
    # y: [N]

    grad_z = softmax_cross_entropy_gradient(p, y) # [N, C]
    grad_w = x.T @ grad_z # [D, C]
    grad_b = np.sum(p - one_hot(y, p.shape[1]), axis=0) / len(x) # [C]
    return grad_w, grad_b
 
p = np.array([
    [0.1, 0.2, 0.7],
    [0.8, 0.1, 0.1],
    [0.2, 0.6, 0.2]
])
y = np.array([2, 0, 1])

grads = softmax_cross_entropy_gradient(p, y)
result = cross_entropy_batch(p, y)

# testing stable soft max
z = np.array([
    [1.0, 2.0, 3.0],
    [1000.0, 1001.0, 1002.0],
    [-1000.0, -999.0, -998.0]
])
result = stable_softmax(z)
assert np.all(np.isfinite(result))
assert np.allclose(result.sum(axis=1), 1.0)
expected = np.exp(z[0]) / np.exp(z[0]).sum()
assert np.allclose(result[0], expected, atol=1e-12)


# check stable loss function
z = np.array([
    [1.0, 2.0, 3.0],
    [1000.0, 1001.0, 1002.0],
    [-1000.0, -999.0, -998.0]
])

y = np.array([2, 2, 2])
loss = cross_entropy_from_logits(z, y)
assert np.isfinite(loss)
assert np.isclose(loss, 0.4076059644, atol=1e-9)


# check forward
x = np.array([
    [1.0, 2.0],
    [2.0, 1.0],
    [-1.0, -2.0],
    [-2.0, -1.0]
])
w = np.array([
    [0.1, 0.2, -0.1],
    [0.3, -0.2, 0.1]
])
b = np.array([0.1, 0.0, -0.1])
z, p = forward(x, w, b)

# check backward
y = np.array([0, 0, 2, 2])
dw, db = backward(x, p, y)
assert dw.shape == w.shape
assert db.shape == b.shape
assert np.all(np.isfinite(dw))
assert np.all(np.isfinite(db))