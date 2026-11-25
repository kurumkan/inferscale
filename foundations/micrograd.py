import random
import math

# autograd reverse engineered
class Value:
    def __init__(self, data, parents=()):
        self._backward = lambda: None
        self.data = data
        self.parents = parents
        self.grad = 0 # store gradient

    def __add__(self, other):
        def _backward():
            self.grad += out.grad * 1 # dl_da += dl_dout * dout_da
            other.grad += out.grad * 1
        
        parents = (self, other)
        out = Value(self.data + other.data, parents)
        out._backward = _backward    

        return out

    def __mul__(self, other):
        def _backward():
            self.grad += out.grad * other.data # dl_da += dl_dout * dout_da
            other.grad += out.grad * self.data
        
        parents = (self, other)
        out = Value(self.data * other.data, parents)
        out._backward = _backward    

        return out

    def __pow__(self, exponent):
        def _backward():
            self.grad += out.grad * exponent * self.data ** (exponent - 1)
        
        parents = (self,)
        out = Value(self.data ** exponent, parents)
        out._backward = _backward    

        return out

    def relu(self):
        def _backward():
            self.grad += out.grad * float(self.data > 0)
        
        parents = (self,)
        out = Value(max(0, self.data), parents)
        out._backward = _backward    

        return out

    def backward(self):
        self.zero_grad()

        topo = self.build_topo()
        self.grad = 1

        for node in reversed(topo):
            node._backward()

    def zero_grad(self):
        topo = self.build_topo()                

        for node in reversed(topo):
            node.grad = 0

    def build_topo(self):
        topo = []
        visited = set()

        def visit(node):
            if node in visited:
                return

            visited.add(node)
            for p in node.parents:
                visit(p)
            topo.append(node)

        visit(self)
        return topo
                    
    def __repr__(self):
        return "Value(data=" + str(self.data) + ")"


def numerical_grad(f, x, eps=1e-4):
    return (f(x + eps) - f(x - eps)) / 2 / eps    

class Neuron:
    def __init__(self, nin):
        # xavier init - cannot be used, because of relu
        # limit = math.sqrt(6 / (nin + nout))
        # self.weights = [Value(random.uniform(-limit, limit)) for _ in range(nin)]

        # use he uniform, since Relu zeroing negative vals, breaking activation distribution
        limit = math.sqrt(6 / nin)
        self.weights = [
            Value(random.uniform(-limit, limit))
            for _ in range(nin)
        ]

        self.bias = Value(0.0)

    def __call__(self, inputs):
        result = Value(0)
        for i, x in enumerate(inputs):
            result += x * self.weights[i]

        result += self.bias            

        return result.relu()


class Layer:
    def __init__(self, nin, nout):
        self.neurons = [Neuron(nin) for _ in range(nout)]

    def __call__(self, inputs):
        result = []
        for n in self.neurons:
            result.append(n(inputs))

        return result



layer = Layer(2, 3)

params = [
    ([3, 4], 1),
    ([1, -2], 0),
    ([-1, 2], 0),
]

for neuron, (weights, bias) in zip(layer.neurons, params):
    for w, value in zip(neuron.weights, weights):
        w.data = value
    neuron.bias.data = bias

x1 = Value(2)
x2 = Value(-1)

outputs = layer([x1, x2])
loss = outputs[0] + outputs[1] + outputs[2]

loss.backward()

assert [o.data for o in outputs] == [3, 4, 0]
assert loss.data == 7

assert x1.grad == 4
assert x2.grad == 2

assert [w.grad for w in layer.neurons[0].weights] == [2, -1]
assert [w.grad for w in layer.neurons[1].weights] == [2, -1]
assert [w.grad for w in layer.neurons[2].weights] == [0, 0]

assert [n.bias.grad for n in layer.neurons] == [1, 1, 0]

print("Layer backward PASS")