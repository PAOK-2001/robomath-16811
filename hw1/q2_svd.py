"""
q1_ldu_decomposition.py

Author: Pablo Ortega Kral (portegak)
"""

import numpy as np

def get_svd(A: np.array, print_result: bool = True):
    rows, cols = A.shape
    U,S, Vh = np.linalg.svd(A)
    S = np.diag(S)
    if len(S) < rows:
        S = np.pad(S, ((0, rows - S.shape[0]), (0, 0)), mode='constant', constant_values=0)

    np.testing.assert_allclose(A, U@S@Vh,atol=1e-10)
    if print_result:
        print("U")
        print(U)

        print("S")
        print(S)

        print("Vt")
        print(Vh)

    return U,S, Vh

if __name__ == "__main__":
    A = np.array([ 
        [1,1,1],
        [10,2,9],
        [8,0,7]
    ], dtype= np.float64)

    get_svd(A)