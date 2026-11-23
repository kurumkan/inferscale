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
            print("mul backward")
            print(self.grad)
            print(out.grad)
            print(out.grad)
            print(self.data)
        
        parents = (self, other)
        out = Value(self.data + other.data, parents)
        out._backward = _backward    

        return out

    def __mul__(self, other):
        def _backward():
            self.grad += out.grad * other.data # dl_da += dl_dout * dout_da
            other.grad += out.grad * self.data
            print("mul backward")
            print(self.grad)
            print(out.grad)
            print(out.grad)
            print(self.data)
        
        parents = (self, other)
        out = Value(self.data * other.data, parents)
        out._backward = _backward    

        return out

    def __pow__(self, exponent):
        def _backward():
            print("before pow backward", self.grad)
            self.grad += out.grad * exponent * self.data ** (exponent - 1)
            print("pow backward", exponent)
            print(self.grad)
            print(out.grad)
            print(self.data)
        
        parents = (self,)
        out = Value(self.data ** exponent, parents)
        out._backward = _backward    

        return out

    def relu(self):
        def _backward():
            self.grad += out.grad * float(self.data > 0)
            print("relu backward")
            print(self.grad)
            print(out.grad)
            print(self.data)
        
        parents = (self,)
        out = Value(max(0, self.data), parents)
        out._backward = _backward    

        return out

    def backward(self):
        self.zero_grad()

        topo = self.build_topo()
        self.grad = 1

        print("backward")
        print(self)
        print(topo)

        for node in reversed(topo):
            node._backward()

    def zero_grad(self):
        topo = self.build_topo()
        self.grad = 1
                
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
        print("topology")
        print(self)
        print(topo)
        return topo
                    
    def __repr__(self):
        return "Value(data=" + str(self.data) + ")"


x = Value(3)
h = x.relu()
loss = h ** 2

loss.backward()

assert x.grad == 6
assert h.grad == 6

loss.zero_grad()

assert x.grad == 0
assert h.grad == 0
assert loss.grad == 0

loss.backward()

assert x.grad == 6
assert h.grad == 6