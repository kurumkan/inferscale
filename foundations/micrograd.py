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
        # use He uniform, since Relu zeroing negative vals, breaking activation distribution
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


class MLP:
    def __init__(self, nin, nouts):
        self.layers = []

        input_count = nin
        for output_count in nouts:
            self.layers.append(Layer(input_count, output_count))
            input_count = output_count

    def __call__(self, inputs):
        data = inputs
        result = []
        for layer in self.layers:
            result = layer(data)    
            data = result

        return result


random.seed(42)
model = MLP(2, [3, 2, 1])

assert len(model.layers) == 3
assert [len(layer.neurons) for layer in model.layers] == [3, 2, 1]
assert [len(layer.neurons[0].weights) for layer in model.layers] == [2, 3, 2]
outputs = model([Value(2), Value(-1)])

assert len(outputs) == 1
assert isinstance(outputs[0], Value)

print("MLP structure PASS")

