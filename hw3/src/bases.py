import math

class Constant:
    def __init__(self, c=1):
        self.c = c
        self.print_txt = f"{c}"

    def __call__(self, x):
        return self.c

class Linear:
    def __init__(self, c=1):
        self.c = c
        self.print_txt = f"{c} * x"

    def __call__(self, x):
        return self.c * x

class Quadratic:
    def __init__(self, c=1):
        self.c = c
        self.print_txt = f"{c} * x^2"

    def __call__(self, x):
        return self.c * x**2

class CosPi:
    def __init__(self, c=1):
        self.c = c
        self.print_txt = f"cos({c} * pi * x)"

    def __call__(self, x):
        return math.cos(self.c * math.pi * x)

class SinPi:
    def __init__(self, c=1):
        self.c = c
        self.print_txt = f"sin({c} * pi * x)"

    def __call__(self, x):
        return math.sin(self.c * math.pi * x)
