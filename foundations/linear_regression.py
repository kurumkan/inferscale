import numpy as np

x = np.array([1., 2., 3.])
y = np.array([3., 5., 7.])
x_matrix = np.stack([np.ones_like(x), x], axis = 1)

w = 1.0
b = 0.0

learning_rate = 0.01


for i in range(1000):
    params = np.array([[b], [w]])
    y_predicted = x_matrix @ params

    diff = y_predicted.flatten() - y
    mse = np.mean(diff ** 2) 

    grad_w = np.mean(2 * x * diff ) 
    grad_b = np.mean(2 * diff)

    w = w - learning_rate * grad_w
    b = b - learning_rate * grad_b

    if i % 100 == 0:
        print("[", i, "] mse", mse)


print("results:")
print("w", w)
print("b", b)
print("mse", mse)
