import numpy as np


def relu(m):
    return np.maximum(0, m)    

def d_relu(m):
    return (m > 0).astype(float)

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

# forward pass
z1 = x @ w1 + b1     # (1, 2)
h1 = relu(z1)        # (1, 2)   
z2 = h1 @ w2 + b2    # (1, 1)
loss = (z2 - y) ** 2 # (1, 1)

# backward pass
dl_dz2 = 2 * (z2 - y)   # (1, 1)
dl_dw2 = h1.T @ dl_dz2  # (2, 1)
dl_db2 = dl_dz2         # (1, 1) 

dl_dh1 = dl_dz2 @ w2.T  # (1, 2)
dl_dz1 = dl_dh1 * d_relu(z1) # element to element multiplication (1, 2)

dl_dw1 = x.T @ dl_dz1 # (2, 2)
dl_db1 = dl_dz1       # (1, 2)

assert dl_dw1.shape == w1.shape
assert np.allclose(dl_dw1, [[16, -8], [32, -16]])
assert np.allclose(dl_db1, [[16, -8]])

# numerical check of gradients: w2
eps = 0.00001
original = w2[0, 0]
w2[0, 0] += eps
z1 = x @ w1 + b1     
h1 = relu(z1)        
z2 = h1 @ w2 + b2    
loss_plus = (z2 - y) ** 2 
w2[0, 0] = original

w2[0, 0] -= eps
z1 = x @ w1 + b1     
h1 = relu(z1)        
z2 = h1 @ w2 + b2    
loss_minus = (z2 - y) ** 2 
w2[0, 0] = original

numerical_dl_dw2 = (loss_plus - loss_minus) / 2 / eps

assert np.allclose(
    numerical_dl_dw2[0, 0],
    dl_dw2[0, 0],
    atol=1e-4,
    rtol=0
)

# numerical check of gradients: w1
original = w1[0, 0]
w1[0, 0] += eps
z1 = x @ w1 + b1     
h1 = relu(z1)        
z2 = h1 @ w2 + b2    
loss_plus = (z2 - y) ** 2 
w1[0, 0] = original

w1[0, 0] -= eps
z1 = x @ w1 + b1     
h1 = relu(z1)        
z2 = h1 @ w2 + b2    
loss_minus = (z2 - y) ** 2 
w1[0, 0] = original

numerical_dl_dw1 = (loss_plus - loss_minus) / 2 / eps

assert np.allclose(
    numerical_dl_dw1[0, 0],
    dl_dw1[0, 0],
    atol=1e-4,
    rtol=0
)
