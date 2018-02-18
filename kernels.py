import numpy as np
import numpy.linalg as la


class Kernel(object):
    @staticmethod
    def linear():
        return lambda x, y: np.inner(x, y)

    @staticmethod
    def gaussian(sigma):
        return lambda x, y: \
            np.exp(-np.sqrt(la.norm(x-y) ** 2 / (2 * sigma ** 2)))

    @staticmethod
    def polynomial(dimension, offset):
        return lambda x, y: (offset + np.inner(x, y)) ** dimension