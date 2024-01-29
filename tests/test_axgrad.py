import unittest
from axgrad.axgrad import Value

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

    def test_poly2(self):
        # y = x**3 + 2*x**2 + x - 7
        for i in range(1, 10):
            # calculate derivative with backward pass
            x = Value(float(i), label='x')
            y = x**3 + 2*x**2 + x - 7
            y.backward(0.00001)

            # analytical derivative for comparison
            xx = x.value
            dydx = 3.0*xx**2 + 4*xx + 1

            self.assertAlmostEqual(x.grad, dydx)
