import numpy as np


def relu(m):
    return np.maximum(0, m)    

# x -> z1 -> h1 -> z2 -> loss

x = np.array([[1.0, 2.0]])

w1 = np.array([
    [1.0, -1.0],
    [2.0,  1.0]
])

b1 = np.array([0.0, 0.0])

w2 = np.array([
    [2.0],
    [-1.0]
])

b2 = 1.0

y = 6.0

# forward
z1 = x @ w1 + b1
h1 = relu(z1)
z2 = h1 @ w2 + b2
loss = (z2 - y) ** 2



