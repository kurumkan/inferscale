import numpy as np

a = np.arange(1000, dtype=np.float64)
b = np.arange(1000, dtype=np.float64) * 2


# native python loop
result_loop = np.empty(1000)

for i, _ in enumerate(a):
    result_loop[i] = a[i] + b[i]


# np sum
result_vectorized = a + b

np.testing.assert_allclose(result_loop, result_vectorized)