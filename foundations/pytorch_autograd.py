from micrograd import Value, MLP
import torch
import torch.nn.functional as F
import numpy as np
import random

random.seed(42)

# micrograd MLP
model_micro = MLP(1, [3, 1])
x = Value(2.0)
y = Value(5.0)
prediction_micro = model_micro([x])
print("micrograd MLP prediction")
print(prediction_micro)
loss_micro = (prediction_micro[0] - y) ** 2
loss_micro.backward()

# pytorch version of MLP
model = torch.nn.Sequential(
    torch.nn.Linear(1, 3),
    torch.nn.ReLU(),
    torch.nn.Linear(3, 1)
).double()

with torch.no_grad():
  # Layer 0: Linear(1, 3) -> weight shape is (out_features, in_features) = (3, 1)
    weight_data = [[w.data for w in n.weights] for n in model_micro.layers[0].neurons]
    bias_data = [n.bias.data for n in model_micro.layers[0].neurons]
    model[0].weight.copy_(torch.tensor(weight_data, dtype=torch.float64))
    model[0].bias.copy_(torch.tensor(bias_data, dtype=torch.float64))

   # Layer 2: Linear(3, 1) -> weight shape is (out_features, in_features) = (1, 3)
    model[2].weight.copy_(torch.tensor([[w.data for w in model_micro.layers[1].neurons[0].weights]], dtype=torch.float64))
    model[2].bias.copy_(torch.tensor([model_micro.layers[1].neurons[0].bias.data], dtype=torch.float64))


x = torch.tensor([[2.0]], dtype=torch.float64)
prediction = model(x)

print("pytorch MLP prediction")
print(prediction.item())

target = torch.tensor([[5.0]], dtype=torch.float64)
loss = F.mse_loss(prediction, target)

# backward
model.zero_grad()

print(loss.grad_fn)
print(loss.grad_fn.next_functions)
print(prediction.grad_fn)
print(prediction.grad_fn.next_functions)
print([
    name for name in dir(prediction.grad_fn)
    if name.startswith("_saved_")
])
print(prediction.grad_fn._saved_mat1)
print(prediction.grad_fn._saved_mat2)

loss.backward()

# compare
wd_micro0 = np.array([[w.grad for w in n.weights] for n in model_micro.layers[0].neurons])
bd_micro0= np.array([n.bias.grad for n in model_micro.layers[0].neurons])
wd_micro2 = np.array([[w.grad for w in model_micro.layers[1].neurons[0].weights]])
bd_micro2= np.array([model_micro.layers[1].neurons[0].bias.grad])

with torch.no_grad():
  # Layer 0: Linear(1, 3) -> weight shape is (out_features, in_features) = (3, 1)    
    weights0 = model[0].weight.grad.numpy()
    bias0= model[0].bias.grad.numpy()

   # Layer 2: Linear(3, 1) -> weight shape is (out_features, in_features) = (1, 3)
    weights2 = model[2].weight.grad.numpy()
    bias2 = model[2].bias.grad.numpy()

    np.testing.assert_allclose(weights0, wd_micro0, rtol=1e-9, atol=1e-10)
    np.testing.assert_allclose(bias0, bd_micro0, rtol=1e-9, atol=1e-10)
    np.testing.assert_allclose(weights2, wd_micro2, rtol=1e-9, atol=1e-10)
    np.testing.assert_allclose(bias2, bd_micro2, rtol=1e-9, atol=1e-10)

    np.testing.assert_allclose(loss.numpy(), loss_micro.data, rtol=1e-9, atol=1e-10)
    np.testing.assert_allclose(prediction.numpy(), np.array([[prediction_micro[0].data]]), rtol=1e-9, atol=1e-10)
