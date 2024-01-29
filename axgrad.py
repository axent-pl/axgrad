import unittest
from typing import Union, Tuple


class Operation:

    def __init__(self, left: 'Value', right: 'Value') -> None:
        self.left = left
        self.right = right

    def __call__(self) -> float:
        pass

    def parameters(self) -> Tuple['Value', 'Value']:
        return self.left, self.right

    def gradients(self, value: float, h: float) -> Tuple[float, float]:
        pass


class AddOperation(Operation):

    def __call__(self) -> float:
        return self.left.value + self.right.value

    def gradients(self, value: float, h: float) -> Tuple[float, float]:
        return 1.0, 1.0


class SubOperation(Operation):

    def __call__(self) -> float:
        return self.left.value - self.right.value

    def gradients(self, value: float, h: float) -> Tuple[float, float]:
        return 1.0, -1.0


class MulOperation(Operation):

    def __call__(self) -> float:
        return self.left.value * self.right.value

    def gradients(self, value: float, h: float) -> Tuple[float, float]:
        return self.right.value, self.left.value


class PowOperation(Operation):

    def __call__(self) -> float:
        return self.left.value ** self.right.value

    def gradients(self, value: float, h: float) -> Tuple[float, float]:
        left_grad = self.right.value * \
            self.left.value ** (self.right.value - 1.0)
        right_grad = (self.left.value ** (self.right.value + h) - value) / h
        return left_grad, right_grad


class Value:

    def __init__(self, value: float, label: str = None, func: Operation = None):
        self.value: float = value
        self.label: str = label
        self.func = func
        self.grad: float = 0.0

    def _ensure_value(self, other: Union['Value', float, int]):
        return other if isinstance(other, Value) else Value(float(other))

    def __add__(self, other: Union['Value', float, int]) -> 'Value':
        other = self._ensure_value(other)
        func = AddOperation(self, other)
        return Value(value=func(), func=func)

    def __radd__(self, other: Union['Value', float, int]) -> 'Value':
        other = self._ensure_value(other)
        func = AddOperation(other, self)
        return Value(value=func(), func=func)

    def __sub__(self, other: Union['Value', float, int]) -> 'Value':
        other = self._ensure_value(other)
        func = SubOperation(self, other)
        return Value(value=func(), func=func)

    def __rsub__(self, other: Union['Value', float, int]) -> 'Value':
        other = self._ensure_value(other)
        func = SubOperation(other, self)
        return Value(value=func(), func=func)

    def __mul__(self, other: Union['Value', float, int]) -> 'Value':
        other = self._ensure_value(other)
        func = MulOperation(self, other)
        return Value(value=func(), func=func)

    def __rmul__(self, other: Union['Value', float, int]) -> 'Value':
        other = self._ensure_value(other)
        func = MulOperation(other, self)
        return Value(value=func(), func=func)

    def __pow__(self, other: Union['Value', float, int]) -> 'Value':
        other = self._ensure_value(other)
        func = PowOperation(self, other)
        return Value(value=func(), func=func)

    def __rpow__(self, other: Union['Value', float, int]) -> 'Value':
        other = self._ensure_value(other)
        func = PowOperation(other, self)
        return Value(value=func(), func=func)

    def __str__(self) -> str:
        return f'{self.value}' if not self.label else f'{self.label}={self.value}'

    def backward(self, h: float) -> None:
        self.grad = 1.0
        self._backward(h)

    def _backward(self, h: float) -> None:
        if self.func is not None:
            parameters = self.func.parameters()
            gradients = self.func.gradients(self.value, h)
            for param, grad in zip(parameters, gradients):
                param.grad = param.grad + self.grad * grad
                param._backward(h)


class TestValue(unittest.TestCase):

    def test_poly0(self):
        # y = x**2 + 2*x + 7
        for i in range(1, 10):
            # calculate derivative with backward pass
            x = Value(float(i), label='x')
            y = x**2 + 2*x + 7
            y.backward(0.00001)

            # analytical derivative for comparison
            xx = x.value
            dydx = 2.0*xx + 2

            self.assertAlmostEqual(x.grad, dydx)


    def test_poly1(self):
        # y = x**2 + 2*x + 7
        for i in range(1, 10):
            # calculate derivative with backward pass
            x = Value(float(i), label='x')
            y = x**3*x**(-1.0) + 2*x + 7
            y.backward(0.00001)

            # analytical derivative for comparison
            xx = x.value
            dydx = 2.0*xx + 2

            self.assertAlmostEqual(x.grad, dydx)


if __name__ == '__main__':
    unittest.main()
