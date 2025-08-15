import numpy as np
from typing import Optional, Tuple, Union

class Tensor:
    """
    A Tensor class that wraps a numpy array and supports automatic differentiation
    """

    def __init__(
        self, data: Union[np.ndarray, list, int, float], requires_grad: bool = False, dtype=np.float32
    ):

        if isinstance(data, np.ndarray):
            self.data = data.astype(dtype) 
        else:
            self.data = np.array(data, dtype=dtype)

        self.requires_grad = requires_grad

    @property
    def shape(self) -> Tuple[int, ...]:
        return self.data.shape

    @property 
    def dtype(self):
        return self.data.dtype

    def __repr__(self) -> str:
        return f"tensor({self.data}, requires_grad={self.requires_grad}, dtype={self.dtype})"

    def zero_grad(self):
        self.grad = None

    def backward(self):
        raise NotImplementedError()