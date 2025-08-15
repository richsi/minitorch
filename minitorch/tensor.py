import numpy as np
from abc import abstractmethod
from typing import Optional, Tuple, Union

class Tensor:
    """
    A Tensor class that wraps a numpy array and supports automatic differentiation
    """

    def __init__(self, data, _children=(), _op=''):
        self.data = np.asarray(data)
        self.grad = np.zeros_like(self.data, dtype=np.float32)

        # graph building
        self._backward = lambda: None # stores function call to compute gradients
        self._prev = set(_children) 
        self._op = _op # operation that created this Tensor

    def __repr__(self):
        return f"Tensor(data={self.data}, grad={self.grad}"

    def __add__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)

        out = Tensor(self.data + other.data, (self, other), '+')

        # defining backwards function for +
        def _backward():
            self.grad += out.grad 
            other.grad += out.grad 
        
        out._backward = _backward
        return out

    def __mul__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)

        out = Tensor(self.data * other.data, (self, other), '*')

        # defining backwards function for +
        def _backward():
            self.grad += other.data * out.grad 
            other.grad += self.data * out.grad 
        
        out._backward = _backward
        return out

    def backward(self):
        topo = []
        visited = set()

        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)
        
        build_topo(self)
        self.grad = np.ones_like(self.data, dtype=np.float32)

        for node in reversed(topo):
            node._backward()