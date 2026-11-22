# foundations of computation graph
class Value:
    def __init__(self, data, parents=()):
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

    def __repr__(self):
        return "Value(data=" + str(self.data) + ")"

    
a = Value(2)
b = Value(3)

c = a + b
c.grad = 1
c._backward()

assert a.grad == 1
assert b.grad == 1