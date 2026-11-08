import numpy as np
print(np.__version__)

e = np.array([
    [
        [1, 2, 3], [4, 5, 6]
    ],
    [
        [7, 8, 9], [10, 11, 12]
    ]
])

f = np.array([[100], [200]])

print("properties of E")
print(e.shape)
print(e.ndim)
print(e.dtype)


print("properties of F")
print(f.shape)
print(f.ndim)
print(f.dtype)


print("sum")
g = e + f
print(g)

desired = np.array([
    [
        [101, 102, 103], [204, 205, 206]
    ],
    [
        [107, 108, 109],  [210, 211, 212]
    ]
])

np.testing.assert_array_equal(g, desired)


a = np.array([1, 2, 3, 4])
b = np.array([[1], [2], [3], [4]])
c = np.array([[1, 2, 3, 4]])

print("a (4, )")
print(a.shape)
print(a.ndim)

print("b (4, 1)")
print(b.shape)
print(b.ndim)

print("c (1, 4)")
print(c.shape)
print(c.ndim)
