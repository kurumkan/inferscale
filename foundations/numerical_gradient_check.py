import numpy as np

def compute_loss(x, y, params):
    size = len(y)
    x_matrix = np.hstack([np.ones((size, 1)), x])
    y_predicted = x_matrix @ params
    diff = (y_predicted.flatten() - y).reshape(-1, 1)
    return np.mean(diff ** 2) 

def gradient_by_index(x, y, params, h, index):
    params_plus = params.copy()
    params_minus = params.copy()

    params_plus[index, 0] += h
    params_minus[index, 0] -= h

    loss_plus = compute_loss(x, y, params_plus)
    loss_minus = compute_loss(x, y, params_minus)

    return (loss_plus - loss_minus) / (2 * h)

def numerical_gradient(x, y, params, h):
    return np.array([
        [gradient_by_index(x, y, params, h, 0)],
        [gradient_by_index(x, y, params, h, 1)],
        [gradient_by_index(x, y, params, h, 2)]
    ])

def analytical_gradient(x, y, params):
    ones_column = np.ones((len(y), 1))
    x_matrix = np.hstack([ones_column, x])
    y_predicted = x_matrix @ params
    diff = (y_predicted.flatten() - y).reshape(-1, 1)

    grad_w = 2 * x.T @ diff / len(x)
    grad_b = np.mean(2 * diff)
    return np.array([
        [grad_b],
        [grad_w[0, 0]],
        [grad_w[1, 0]]
    ])

rng = np.random.default_rng(42)
x = rng.random((1000, 2))
y = rng.normal(size=1000)

# important to have float64, otherwise precision errors
params = np.array([
    [0.5],
    [1.5],
    [-2.0]
])
h = 1e-5

grad_analytical = analytical_gradient(x, y, params)
grad_numerical = numerical_gradient(x, y, params, h)

np.testing.assert_allclose(grad_analytical, grad_numerical, rtol=1e-4, atol=1e-6)