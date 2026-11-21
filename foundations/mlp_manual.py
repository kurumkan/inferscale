import numpy as np

def relu(m):
    return np.maximum(0, m)    

def d_relu(m):
    return (m > 0).astype(float)

# get loss value
def forward_loss(x, w1, b1, w2, b2, y): 
    z1 = x @ w1 + b1     
    h1 = relu(z1)        
    z2 = h1 @ w2 + b2    
    return (z2 - y) ** 2 

# calculate gradients
def backward_pass(x, w1, b1, w2, b2, y):
    z1 = x @ w1 + b1     # (1, 2)
    h1 = relu(z1)        # (1, 2)   
    z2 = h1 @ w2 + b2    # (1, 1)

    dl_dz2 = 2 * (z2 - y)   # (1, 1)
    dl_dw2 = h1.T @ dl_dz2  # (2, 1)
    dl_db2 = dl_dz2         # (1, 1) 

    dl_dh1 = dl_dz2 @ w2.T  # (1, 2)
    dl_dz1 = dl_dh1 * d_relu(z1) # element to element multiplication (1, 2)

    dl_dw1 = x.T @ dl_dz1 # (2, 2)
    dl_db1 = dl_dz1       # (1, 2)

    return (dl_dw1, dl_db1.flatten(), dl_dw2, dl_db2.item())


# x -> z1 -> h1 -> z2 -> loss
eps = 0.00001
learning_rate = 0.001

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

loss = forward_loss(x, w1, b1, w2, b2, y)
dl_dw1, dl_db1, dl_dw2, dl_db2 = backward_pass(x, w1, b1, w2, b2, y)

assert dl_dw1.shape == w1.shape
assert np.allclose(dl_dw1, [[16, -8], [32, -16]])
assert np.allclose(dl_db1, [[16, -8]])

# numerical check of gradients: w2
for idx in np.ndindex(w2.shape):
    original = w2[idx]

    w2[idx] += eps
    loss_plus = forward_loss(x, w1, b1, w2, b2, y)
    w2[idx] = original

    w2[idx] -= eps
    loss_minus = forward_loss(x, w1, b1, w2, b2, y)
    w2[idx] = original

    numerical_dl_dw2 = (loss_plus - loss_minus) / 2 / eps
    
    assert np.allclose(
        numerical_dl_dw2.item(),
        dl_dw2[idx],
        atol=1e-4,
        rtol=0
    )

# numerical check of gradients: b2
original = b2

b2 += eps
loss_plus = forward_loss(x, w1, b1, w2, b2, y)
b2 = original

b2 -= eps
loss_minus = forward_loss(x, w1, b1, w2, b2, y)
b2 = original

numerical_dl_db2 = (loss_plus - loss_minus) / 2 / eps

assert np.allclose(
    numerical_dl_db2.item(),
    dl_db2,
    atol=1e-4,
    rtol=0
)    

# numerical check of gradients: w1
for idx in np.ndindex(w1.shape):
    original = w1[idx]

    w1[idx] += eps
    loss_plus = forward_loss(x, w1, b1, w2, b2, y)
    w1[idx] = original

    w1[idx] -= eps
    loss_minus = forward_loss(x, w1, b1, w2, b2, y)
    w1[idx] = original

    numerical_dl_dw1 = (loss_plus - loss_minus) / 2 / eps

    assert np.allclose(
        numerical_dl_dw1.item(),
        dl_dw1[idx],
        atol=1e-4,
        rtol=0
    )

# numerical check of gradients: b1
for idx in np.ndindex(b1.shape):
    original = b1[idx]

    b1[idx] += eps
    loss_plus = forward_loss(x, w1, b1, w2, b2, y)
    b1[idx] = original

    b1[idx] -= eps
    loss_minus = forward_loss(x, w1, b1, w2, b2, y)
    b1[idx] = original

    numerical_dl_db1 = (loss_plus - loss_minus) / 2 / eps

    assert np.allclose(
        numerical_dl_db1.item(),
        dl_db1[idx],
        atol=1e-4,
        rtol=0
    )

# train
for i in range(100):
    dl_dw1, dl_db1, dl_dw2, dl_db2 = backward_pass(x, w1, b1, w2, b2, y)

    w1 -= learning_rate * dl_dw1
    b1 -= learning_rate * dl_db1
    w2 -= learning_rate * dl_dw2
    b2 -= learning_rate * dl_db2

    loss = forward_loss(x, w1, b1, w2, b2, y)

    if i % 10 == 0:
        print(loss)    

final_loss = forward_loss(x, w1, b1, w2, b2, y)
assert final_loss < 0.01
for item in [w1, w2, b1, b2]:
    assert np.all(np.isfinite(item)), "Infinite value found"
    