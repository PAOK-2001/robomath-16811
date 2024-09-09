
"""
ldu_decomposition.py

Author: Pablo Ortega Kral (portegak)
"""

import pprint
import numpy as np

from typing import Tuple

def get_pivot_matrix(A: np.array):
    # We can find the pivot matrix prior to any calculation using Doolittle's method
    rows, cols = A.shape
    P = np.eye(rows, dtype= np.float64)
    for i in range(rows):
        max_row = np.argmax(np.abs(A[i:, i])) + i
        if i != max_row:
             P[[i, max_row]] = P[[max_row, i]]
    return P

def ldu(A: np.array, pivot: bool = False) -> Tuple[np.array]:
    rows, cols = A.shape

    P = get_pivot_matrix(A)
    A_pivoted =  P@A # Apply transformation to re-index A
    L = np.eye(rows, dtype= np.float64)# Lower Triangular mxm
    U = np.zeros(shape=(rows,cols), dtype= np.float64)
    DU = A_pivoted.copy()

    # Iterate through matrix diagonal and perform gaussian elimination
    for j in range(cols):
        for i in range(j+1,rows):
            if DU[i][j] != 0:
                scaler = DU[i][j]/DU[j][j] 
                DU[i] = DU[i] - scaler* DU[j]
                L[i][j] = scaler

    # Extract Diagonal Elements and normalize row to obtain u
    D = np.diag(np.diag(DU))
    for i in range(rows):
        diag = D[i,i] 
        if diag != 0.0:
            U[i] = DU[i]/ diag
  
    return (P, L, D, U)

if __name__ == "__main__":
    A = np.array([ 
        [1,1,0],
        [1,1,2],
        [4,2,3]
    ], dtype= np.float64)

    P, L, D, U = ldu(A)

    assert (P@A == L@D@U).all(), "Factorization does not equal original matrix!"
    print("---LDU Decomposition---")

    print("P:")
    pprint.pprint(P)

    print("A:")
    pprint.pprint(A)

    print("L:")
    pprint.pprint(L)

    print("D:")
    pprint.pprint(D)

    print("U:")
    pprint.pprint(U)