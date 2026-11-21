import numpy as np

def relu(m):
    return np.maximum(0, m)    

def d_relu(m):
    return (m > 0).astype(float)

# get outputs from layers
def get_layer_outputs(x, w1, b1, w2, b2): 
    z1 = x @ w1 + b1     
    h1 = relu(z1)        
    z2 = h1 @ w2 + b2    
    return z2, h1, z1

# get prediction 
def get_prediction(x, w1, b1, w2, b2): 
    z2, _, _ = get_layer_outputs(x, w1, b1, w2, b2)
    return z2



# get loss value
def forward_loss(x, w1, b1, w2, b2, y): 
    z2 = get_prediction(x, w1, b1, w2, b2)
    return (z2 - y) ** 2 

# calculate gradients
def backward_pass(x, w1, b1, w2, b2, y):
    z2, h1, z1 = get_layer_outputs(x, w1, b1, w2, b2)

    dl_dz2 = 2 * (z2 - y) / len(y)
    dl_dw2 = h1.T @ dl_dz2  
    dl_db2 = np.sum(dl_dz2) 

    dl_dh1 = dl_dz2 @ w2.T 
    dl_dz1 = dl_dh1 * d_relu(z1) 
    dl_dw1 = x.T @ dl_dz1
    dl_db1 = np.sum(dl_dz1, axis=0) 

    return (dl_dw1, dl_db1.flatten(), dl_dw2, dl_db2.item())

eps = 1e-5

x = np.array([
    [0.0, 0.0],
    [0.0, 1.0],
    [1.0, 0.0],
    [1.0, 1.0]
])
y = np.array([
    [0.0],
    [1.0],
    [1.0],
    [0.0]
])

w1 = np.array([
    [1.0, -1.0],
    [2.0,  1.0]
])
b1 = np.array([0.1, 0.2])
w2 = np.array([
    [2.0],
    [-1.0]
])
b2 = 1.0

loss = forward_loss(x, w1, b1, w2, b2, y)
average_loss = np.mean(loss)

dl_dw1, dl_db1, dl_dw2, dl_db2 = backward_pass(x, w1, b1, w2, b2, y)

# check grads numerically: w2[0, 0]
original = w2[0, 0]
w2[0, 0] += eps
loss_plus = forward_loss(x, w1, b1, w2, b2, y)
w2[0, 0] = original

w2[0, 0] -= eps
loss_minus = forward_loss(x, w1, b1, w2, b2, y)
w2[0, 0] = original

numerical_dl_dw2 = np.mean((loss_plus - loss_minus) / 2 / eps)

assert np.allclose(
    numerical_dl_dw2,
    dl_dw2[0, 0],
    atol=eps,
    rtol=0
)

# check grads numerically: w2
for idx in np.ndindex(w2.shape):
    original = w2[idx]
    w2[idx] += eps
    loss_plus = forward_loss(x, w1, b1, w2, b2, y)
    w2[idx] = original

    w2[idx] -= eps
    loss_minus = forward_loss(x, w1, b1, w2, b2, y)
    w2[idx] = original

    numerical_dl_dw2 = np.mean((loss_plus - loss_minus) / 2 / eps)

    assert np.allclose(
        numerical_dl_dw2,
        dl_dw2[idx],
        atol=eps,
        rtol=0
    )


# check grads numerically: b2
original = b2

b2 += eps
loss_plus = forward_loss(x, w1, b1, w2, b2, y)
b2 = original

b2 -= eps
loss_minus = forward_loss(x, w1, b1, w2, b2, y)
b2 = original

numerical_dl_db2 = np.mean((loss_plus - loss_minus) / 2 / eps)

assert np.allclose(
    numerical_dl_db2,
    dl_db2,
    atol=eps,
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

    numerical_dl_dw1 = np.mean((loss_plus - loss_minus) / 2 / eps)

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

    numerical_dl_db1 = np.mean((loss_plus - loss_minus) / 2 / eps)
    
    assert np.allclose(
        numerical_dl_db1.item(),
        dl_db1[idx],
        atol=1e-4,
        rtol=0
    )


# train
learning_rate = 0.01
epochs = 2000

loss = 1000
for i in range(epochs):
    dl_dw1, dl_db1, dl_dw2, dl_db2 = backward_pass(x, w1, b1, w2, b2, y)
    
    w1 -= learning_rate * dl_dw1
    b1 -= learning_rate * dl_db1
    w2 -= learning_rate * dl_dw2
    b2 -= learning_rate * dl_db2

    loss = np.mean(forward_loss(x, w1, b1, w2, b2, y))

    if i % 200 == 0:
        print(i, "average loss", loss)

# get final prediction:
final_prediction = get_prediction(x, w1, b1, w2, b2)
print("final prediction:", final_prediction)
print("total average loss:", loss)        

for item in [w1, w2, b1, b2]:
    assert np.all(np.isfinite(item)), "Infinite value found"
    
