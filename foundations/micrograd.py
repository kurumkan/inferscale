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

    def backward(self):
        topo = self.build_topo()
        self.grad = 1

        for node in reversed(topo):
            node._backward()

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



a = Value(2)
b = Value(3)

p = a * b
q = p + a
loss = q * q

loss.backward()

assert loss.grad == 1
assert q.grad == 16
assert p.grad == 16
assert a.grad == 64
assert b.grad == 32