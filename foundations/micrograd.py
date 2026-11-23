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

def build_topo(root):
    topo = []
    visited = set()

    def visit(node):
        if node in visited:
            return

        for p in node.parents:
            visit(p)
        topo.append(node)
        visited.add(node)

    visit(root)
    return topo
    
a = Value(2)
b = Value(3)
p = a * b
q = p + a
loss = q * q # loss = a * b + a

topo = build_topo(loss)
assert len(topo) == 5
assert topo.index(a) < topo.index(p)
assert topo.index(b) < topo.index(p)
assert topo.index(p) < topo.index(q)
assert topo.index(a) < topo.index(q)
assert topo.index(q) < topo.index(loss)