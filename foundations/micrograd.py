import random
import math

# autograd reverse engineered
class Value:
    def __init__(self, data, parents=(), op=""):
        self._backward = lambda: None
        self.data = data
        self.parents = parents
        self.grad = 0
        self.op = op

    def __add__(self, other):
        def _backward():
            self.grad += out.grad * 1 # dl_da += dl_dout * dout_da
            other.grad += out.grad * 1
        
        parents = (self, other)
        out = Value(self.data + other.data, parents, op="+")
        out._backward = _backward    

        return out

    def __sub__(self, other):  # self - other
        return self + (-other)
    
    def __neg__(self):  # -self
        def _backward():
            self.grad += out.grad * -1

        out = Value(-self.data, (self,), op="neg")
        out._backward = _backward
    
        return out
        
    def __mul__(self, other):
        def _backward():
            self.grad += out.grad * other.data # dl_da += dl_dout * dout_da
            other.grad += out.grad * self.data
        
        parents = (self, other)
        out = Value(self.data * other.data, parents, op="*")
        out._backward = _backward    

        return out

    def __pow__(self, exponent):
        def _backward():
            self.grad += out.grad * exponent * self.data ** (exponent - 1)
        
        parents = (self,)
        out = Value(self.data ** exponent, parents, op=f"**{exponent}")
        out._backward = _backward    

        return out

    def relu(self):
        def _backward():
            self.grad += out.grad * float(self.data > 0)
        
        parents = (self,)
        out = Value(max(0, self.data), parents, op="ReLU")
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
    def __init__(self, nin, activation = "relu"):        
        # use He uniform, since Relu zeroing negative vals, breaking activation distribution
        limit = math.sqrt(6 / nin)
        self.weights = [
            Value(random.uniform(-limit, limit))
            for _ in range(nin)
        ]

        self.bias = Value(0.0)
        self.activation = activation

    def __call__(self, inputs):
        result = Value(0)
        for i, x in enumerate(inputs):
            result += x * self.weights[i]

        result += self.bias            

        if self.activation == "relu":
            return result.relu()
        elif self.activation == "linear":
            return result
        else:
            raise ValueError(f"Unknown activation: {self.activation}")

class Layer:
    def __init__(self, nin, nout, activation = "relu"):
        self.neurons = [Neuron(nin, activation) for _ in range(nout)]

    def __call__(self, inputs):
        result = []
        for n in self.neurons:
            result.append(n(inputs))

        return result


class MLP:
    def __init__(self, nin, nouts):
        self.layers = []

        input_count = nin
        for i, output_count in enumerate(nouts):
            is_last = i == len(nouts) - 1
            if is_last:
                self.layers.append(Layer(input_count, output_count, activation="linear"))
            else:
                self.layers.append(Layer(input_count, output_count, activation="relu"))                    
            input_count = output_count

    def __call__(self, inputs):
        data = inputs
        result = []
        for layer in self.layers:
            result = layer(data)    
            data = result

        return result

if __name__ == "__main__":
    random.seed(42)

    model = MLP(1, [3, 1])

    dataset = [
        (-1, -1),
        (0, 1),
        (1, 3),
        (2, 5),
    ]

    learning_rate = 0.02
    epochs = 500

    initial_loss = None

    for epoch in range(epochs):
        loss = Value(0)

        for x_scalar, y_scalar in dataset:
            x, y = (Value(x_scalar), Value(y_scalar))
            prediction = model([x])
            loss = loss + (y + prediction[0] * Value(-1)) ** 2 

        if epoch == 0:
            initial_loss = loss.data

        loss.backward()

        for layer in model.layers:
            for neuron in layer.neurons:
                for w in neuron.weights:
                    w.data -= learning_rate * w.grad

                neuron.bias.data -= learning_rate * neuron.bias.grad    

        if epoch % 100 == 0:
            print(f"Epoch {epoch}: loss={loss.data:.6f}")

    # check results
    final_loss = 0.0

    for x, target in dataset:
        prediction = model([Value(x)])[0].data
        final_loss += (prediction - target) ** 2
        print(f"x={x:2}, target={target:2}, prediction={prediction:.4f}")

    print("Initial loss:", initial_loss)
    print("Final loss:", final_loss)

    assert final_loss < initial_loss
    assert final_loss < 0.01
