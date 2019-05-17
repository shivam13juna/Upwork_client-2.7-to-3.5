
import numpy
from numpy import add, subtract, multiply, divide, negative, exp, square, log, sqrt

numpy.seterr(all=None)


class ActivationFunction(object):
    def get_random_weights(self, number_of_weights):
        r = 1.0 / numpy.sqrt(number_of_weights[1])
        return numpy.random.uniform(-r, r, number_of_weights), numpy.random.uniform(-r, r, number_of_weights[0])

    def apply(self, x):
        raise NotImplementedError

    def derivative(self, x=None, f_of_x=None, out=None):
        raise NotImplementedError

    def inverse(self, y):
        raise NotImplementedError


class Identity(ActivationFunction):
    def apply(self, x, out=None):
        return x

    def derivative(self, x=None, f_of_x=None, out=None):
        return 1.0

    def inverse(self, y):
        return y


class Sum(ActivationFunction):
    def apply(self, value, out=None):
        return value

    def derivative(self, x=None, f_of_x=None, out=None):
        return 1.0


class UnipolarSigmoid(ActivationFunction):
    def apply(self, x, out=None):
        if out is not None:
            return divide(1.0, add(1.0, exp(negative(x, out), out), out), out)

        return 1.0 / (1.0 + (numpy.e ** (-x)))

    def derivative(self, x=None, f_of_x=None, out=None):
        if x is not None:
            f_of_x = self.apply(x)
        return (1.0 - f_of_x) * f_of_x

    def inverse(self, y):
        return numpy.log(y / (1 - y))


class BipolarSigmoid(ActivationFunction):
    def apply(self, x, out=None):
        return -1.0 + (2.0 / (1.0 + numpy.e ** (-x)))

    def derivative(self, x=None, f_of_x=None, out=None):
        if x is not None:
            f_of_x = self.apply(x)
        return numpy.sqrt((1.0 + f_of_x) * (1.0 - f_of_x))

    def inverse(self, y):
        return numpy.log((y + 1.0) / (1.0 - y))


class HyperbolicTangent(ActivationFunction):
    def apply(self, x, out=None):
        if out is not None:
            out = exp(multiply(2.0, x, out), out)
            return divide((out - 1.0), (out + 1.0), out)

        e_raised_to_the_two_times_x = exp(2.0 * x)
        return (e_raised_to_the_two_times_x - 1) / (e_raised_to_the_two_times_x + 1)

    def derivative(self, x=None, f_of_x=None, out=None):
        if x is not None:
            f_of_x = self.apply(x, out=out)

        if out is not None:
            return subtract(1.0, square(f_of_x, out), out)

        return 1.0 - square(f_of_x)

    def inverse(self, y, out=None):
        if out is not None:
            return sqrt(log(divide((1.0 + y), (1.0 - y), out), out), out)

        return sqrt(numpy.log((1.0 + y) / (1.0 - y)))
