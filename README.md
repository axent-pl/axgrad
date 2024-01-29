# axgrad
Backpropagation engine for scalar values.

This code has been created just to train myself in how backpropagation works (get the proper intuition).

## Example usage

```python
from axgrad.axgrad import Value

for i in range(1, 10):
    x = Value(i)
    y = x**3 + 2*x**2 + x - 7
    y.backward(0.001)
    print(x, x.grad)
```

## Run tests

```shell
python -m unittest tests/test_axgrad.py
```