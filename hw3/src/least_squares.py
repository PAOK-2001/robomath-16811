import random
import bases as B
import numpy as np
import matplotlib.pyplot as plt
from utils import poly_from_coeffs, print_poly

from typing import List
from utils import load_data_point

def solve_system_svd(A: np.array, B: np.array) -> np.array:
    U,S, Vh = np.linalg.svd(A, full_matrices=False)
    S_inv = np.diag(1/S)
    V = Vh.T
    # Proposed SVD solution
    x_bar = V@ S_inv @ U.T @ B
    return x_bar

def least_squares(f: callable, interval: np.array, bases_functions: List[callable]) -> np.array:
    inter = np.arange(interval[0], interval[1] + 1)
    A = np.zeros((len(inter), len(bases_functions)))
    b = np.zeros(len(inter))
    # Construct 
    for i, x in enumerate(inter):
        b[i] = f(x) # Construct observation vector from real function values
        # Iterate over the bases functions to construct the matrix A
        for j, base in enumerate(bases_functions):
            A[i, j] = base(x)
    c = solve_system_svd(A, b)
    return c

def least_squares_sampled(data_points, bases_functions: List[callable]) -> np.array:
    x = data_points[:, 0]
    y = data_points[:, 1]
    A = np.zeros((len(x), len(bases_functions)))
    b = y
    for i in range(len(x)):
        for j, base in enumerate(bases_functions):
            A[i, j] = base(x[i])
    c = solve_system_svd(A, b)
    return c

if __name__ == "__main__":
    data_points = load_data_point("data/problem2.txt")
    # Define the bases functions
    all_bases = [B.constant, B.linear, B.quadratic, B.cos_pi, B.sin_pi]
    # Use random N bases
    # N = 3
    # bases_functions = random.sample(all_bases, N)
    bases_functions = all_bases[:3]
    c = least_squares_sampled(data_points, bases_functions)
    # Plot data points and approximation
    x = data_points[:, 0]
    y = data_points[:, 1]
    plt.scatter(x, y, c = "orange", label='Data points')
    y_approx = np.zeros(len(x))
    function = poly_from_coeffs(c)
    print_poly(c)
    for i in range(len(x)):
        y_approx[i] = function(x[i])
    plt.plot(x, y_approx, label='Approximation')
    plt.legend()
    plt.savefig('results/least_square_approx_sampled.png', dpi = 300)
    

    

