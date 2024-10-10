import bases as B
import numpy as np
import matplotlib.pyplot as plt
import itertools

from typing import List
from utils import load_data_point

def solve_system_svd(A: np.array, B: np.array) -> np.array:
    U,S, Vh = np.linalg.svd(A, full_matrices=False)
    S_inv = np.diag(1/S)
    V = Vh.T
    # Proposed SVD solution
    x_bar = V@ S_inv @ U.T @ B
    return x_bar

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


if __name__ == "__main__":
    data_points = load_data_point("data/problem2.txt")
    # Define the bases functions
    all_bases = [B.Constant(), B.Linear(), B.Quadratic(), 
                 B.CosPi(), B.SinPi()]
    
    for ncos in range(1, 10):
        all_bases.append(B.CosPi(ncos))
    
    for nsin in range(1, 10):
        all_bases.append(B.SinPi(nsin))
    
    best_bases = None
    best_error = float('inf')
    best_c = None
    
    # Try all combinations of up to 3 bases functions
    for i in range(1, min(4, len(all_bases) + 1)):
        for bases_functions in itertools.combinations(all_bases, i):
            c = least_squares_sampled(data_points, bases_functions)
            # Calculate the approximation error
            x = data_points[:, 0]
            y = data_points[:, 1]
            y_approx = np.zeros(len(x))
            for j in range(len(x)):
                for k, base in enumerate(bases_functions):
                    y_approx[j] += c[k] * base(x[j])
            error = np.linalg.norm(y - y_approx, ord=2)
            if error < best_error:
                best_error = error
                best_bases = bases_functions
                best_c = c
    
    # Plot data points and best approximation
    x = data_points[:, 0]
    y = data_points[:, 1]
    plt.scatter(x, y, c="orange", s = 7.5, label='Data points')
    y_approx = np.zeros(len(x))

    for i in range(len(x)):
        for j, base in enumerate(best_bases):
            y_approx[i] += best_c[j] * base(x[i])

    for j, base in enumerate(best_bases):     
        print(f"{best_c[j]} * {base.print_txt} + ", end="")

    plt.plot(x, y_approx, c="purple", linewidth=2, label='Best Approximation', zorder= 0)
    plt.legend()
    plt.savefig('results/least_square_approx_best.png', dpi=400)
    

    

