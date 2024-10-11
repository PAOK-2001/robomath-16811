import math
import numpy as np
from utils import poly_from_coeffs

def compute_l_inf(f:callable, coeff: list, interval: np.array, n=100):
    a , b = interval[0], interval[1]
    x_range = np.linspace(a, b, n)
    
    poly = poly_from_coeffs(coeff) # Lis of coefficients in increasing order [c, b, a]

    y_hat = [poly(xi) for xi in x_range]
    y = [f(xi) for xi in x_range]

    errors = [abs(y_hat[i] - y[i]) for i in range(len(y))]
    return max(errors)

def compute_l_2(f:callable, coeff: list, interval: np.array, n=100):
    # Build polynomial from coefficients
    a , b = interval[0], interval[1]
    x_range = np.linspace(a, b, n)
    
    poly = poly_from_coeffs(coeff) # Lis of coefficients in increasing order [c, b, a]

    y_hat = [poly(xi) for xi in x_range]
    y = [f(xi) for xi in x_range]
    
    errors = [(y_hat[i] - y[i])**2 for i in range(n)]
    integral_error = (b - a) / (n - 1) * (errors[0] + 2 * sum(errors[1:-1]) + errors[-1]) / 2
    
    return math.sqrt(integral_error)

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
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--problem", type=str, default="c")
    args = parser.parse_args()

    if args.problem == "c":
        print("Problem 1.c")
        def f(x):
            return math.sin(x) - 0.5
        coeffs = [-0.5, 0.7246, 0]
        interval = [-np.pi/2, np.pi/2]
        l_inf = compute_l_inf(f, coeffs, interval)
        l_2 = compute_l_2(f, coeffs, interval)
        print(f"L_inf: {l_inf}")
        print(f"L_2: {l_2}")
        plot_approximation(f, coeffs, interval)

    elif args.problem == "d":
        print("Problem 1.d")
        def f(x):
            return math.sin(x) - 0.5
        coeffs = [-0.5, 0.774, 0]
        interval = [-np.pi/2, np.pi/2]
        l_inf = compute_l_inf(f, coeffs, interval)
        l_2 = compute_l_2(f, coeffs, interval)
        print(f"L_inf: {l_inf}")
        print(f"L_2: {l_2}")
        plot_approximation(f, coeffs, interval)