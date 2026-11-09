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

# reshape test
a = np.array([
    [1, 2, 3],
    [4, 5, 6]
], dtype=np.int64)

b = a.reshape(3, 2)

print("a shape and strides")
print(a.shape)
print(a.strides)

print("b shape and strides")
print(b.shape)
print(b.strides)

print("a and b share memory?")
print(np.shares_memory(a, b))
print(a.nbytes, b.nbytes)

# mutate array b
b[0, 0] = 1000
print("a[0, 0]")
print(a[0, 0])

# views vs copies
print("test views vs copies")
a = np.array([10, 20, 30, 40, 50, 60], dtype=np.int64)
b = a[::2]
c = a[[0, 2, 4]]

print("a, b shares memory?")
print(np.shares_memory(a, b))
print("a, c shares memory?")
print(np.shares_memory(a, c))

b[0] = 1000
c[0] = 2000

print("arrays after value change")
print(a)
print(b)
print(c)

print("strides")
print(a.strides)
print(b.strides)
print(c.strides)