import numpy
from numpy import linalg


def linear(x, y):
    return numpy.inner(x, y)

# def linear():
#     def f(x, y):
#         return numpy.inner(x, y)
#     return f
    

# def gaussian(sigma):
#     def f(x, y):
#         return numpy.exp(-linalg.norm(x - y)** 2 / (2 * (sigma ** 2)))
#     return f
    

# def polynomial(dim, offset):
#     def f(x, y):
#         return (offset + numpy.dot(x, y)) ** dim
#     return f