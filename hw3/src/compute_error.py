import math
import numpy as np
from utils import poly_from_coeffs

def compute_l_inf(f:callable, coeff: list, interval: np.array):
    # Build polynomial from coefficients
    poly = poly_from_coeffs(coeff) # Lis of coefficients in increasing order [c, b, a]
    # Evaluate the polynomial at the range
    errors = []
    inter =  np.arange(interval[0], interval[1] + 1)
    for x in inter:
        y = f(x)
        e = abs(poly(x) - y)
        errors.append(e)
    return max(errors)

def compute_l_2(f:callable, coeff: list, interval: np.array):
    # Build polynomial from coefficients
    poly = poly_from_coeffs(coeff) # Lis of coefficients in increasing order [c, b, a]
    # Evaluate the polynomial at the range
    errors = []
    inter =  np.arange(interval[0], interval[1] + 1)
    for x in inter:
        y = f(x)
        e = (poly(x) - y)**2
        errors.append(e)
    return np.sqrt(sum(errors))

def plot_approximation(f:callable, coeff: list, interval: np.array):
    import matplotlib.pyplot as plt
    # Build polynomial from coefficients
    poly = poly_from_coeffs(coeff) # Lis of coefficients in increasing order [c, b, a]
    # Evaluate the polynomial at the range
    x = np.linspace(interval[0], interval[1], 100)
    y = [poly(xi) for xi in x]
    y_true = [f(xi) for xi in x]
    plt.plot(x, y, label='Approximation')
    plt.plot(x, y_true, label='True function')
    plt.legend()
    plt.savefig('results/approximation.png')
    

if __name__ == "__main__":
    def f(x):
        return math.sin(x) - 0.5
    coeffs = [-0.5, 0.7246, 0]
    interval = [-np.pi/2, np.pi/2]
    l_inf = compute_l_inf(f, coeffs, interval)
    l_2 = compute_l_2(f, coeffs, interval)
    print(f"L_inf: {l_inf}")
    print(f"L_2: {l_2}")
    plot_approximation(f, coeffs, interval)
