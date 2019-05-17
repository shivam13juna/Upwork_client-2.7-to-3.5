

import numpy

from numpy import add, subtract, divide, multiply, mean, std


class Normalization(object):
    def normalize(self, data_set):
        raise NotImplementedError

    def de_normalize(self, data_set):
        raise NotImplementedError


class Statistical(Normalization):
    def __init__(self, data_set):
        self.means, self.standard_deviation = mean(data_set, axis=0), std(data_set, axis=0)

    def normalize(self, data_set):
        return divide(subtract(data_set, self.means),  self.standard_deviation)

    def de_normalize(self, data_set):
        return add(multiply(data_set, self.standard_deviation), self.means)


class MinMax(Normalization):
    def __init__(self, data_set, interval_range=(-1, 1)):
        self.min, self.max = numpy.min(data_set, axis=0), numpy.max(data_set, axis=0)
        self.interval_range = interval_range
        self.interval_length = float(self.interval_range[1] - self.interval_range[0])
        self.max_min_diff = self.max - self.min

    def normalize(self, data_set):
        return self.interval_length * ((data_set - self.min) / self.max_min_diff) + self.interval_range[0]

    def de_normalize(self, data_set):
        return (((data_set - self.interval_range[0]) / self.interval_length) * self.max_min_diff) + self.min


class Median(Normalization):
    def __init__(self, data_set):
        self.median = numpy.median(data_set, axis=0)

    def normalize(self, data_set):
        return data_set / self.median

    def de_normalize(self, data_set):
        return data_set * self.median


class Sigmoid(Normalization):
    def __init__(self, _, function_name='HyperbolicTangent'):
        self.function = globals()[function_name]()

    def normalize(self, data_set):
        return self.function.apply(data_set)

    def de_normalize(self, data_set):
        return self.function.inverse(data_set)


class StatisticalColumn(Normalization):
    def __init__(self, data_set, bias=.1):
        self.norms = numpy.linalg.norm(data_set, axis=0)
        self.bias = bias

    def normalize(self, data_set):
        return ((data_set - self.norms) / self.norms) * self.bias

    def de_normalize(self, data_set):
        return self.norms * ((data_set / self.bias) + 1.0)

