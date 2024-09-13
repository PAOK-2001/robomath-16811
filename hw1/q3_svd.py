"""
q1_ldu_decomposition.py

Author: Pablo Ortega Kral (portegak)
"""

import numpy as np
import scipy
import scipy.linalg
np.set_printoptions(suppress=True)

def solve_system_svd(A: np.array, B: np.array) -> np.array:
    det = np.linalg.det(A) # Check whether A is invertible, assume it is square
    U,S, Vh = np.linalg.svd(A)
    row, col = U.shape
    S_inv = np.eye(row)
    # Obtain pseudo-inverse
    for i in range(row):
        if np.allclose(S[i], 0, atol= 1e-8):
            S_inv[i,i] = 0.0
        else:
            S_inv[i,i] = 1/S[i]
        
    V = Vh.T
    # Proposed SVD solution
    x_bar = V@ S_inv @ U.T @ B

    # Check whether system has multiple solution
    if det == 0:
        null_space = scipy.linalg.null_space(A)
        x = x_bar + null_space
        print("X_n")
        print(null_space)

        print("X_bar")
        print(x_bar)
        
        print("X")
        print(x)
        return x, x_bar

    else:
        np.testing.assert_allclose(A@x_bar, B)
        print("X")
        print(x_bar)
        return x_bar, None
    
if __name__ == "__main__":
    A = np.array([ 
        [1,1,1],
        [10,2,9],
        [8,0,7]
    ], dtype= np.float64)

    B = np.array([[3], [2], [2]])

    solve_system_svd(A, B)
    
