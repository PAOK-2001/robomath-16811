import math

def constant(x, c = 1):
    return c

def linear(x, c = 1):
    return c * x

def quadratic(x, c = 1):
    return c * x**2

def cos_pi(x, c = 1):
    return math.cos(c * math.pi * x)

def sin_pi(x, c = 1):
    return math.sin(c* math.pi * x)
