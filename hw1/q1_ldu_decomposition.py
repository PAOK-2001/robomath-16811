
"""
q1_ldu_decomposition.py

Author: Pablo Ortega Kral (portegak)
"""

import pprint
import numpy as np

from typing import Tuple



def ldu(A: np.array) -> Tuple[np.array]:
    # Initialize
    rows, cols = A.shape
    P = np.eye(rows, dtype= np.float64)
    L = np.eye(rows, dtype= np.float64) 
    U = np.zeros(shape=(rows,cols), dtype= np.float64)
    DU = A.copy()

    # Iterate through matrix diagonal and perform gaussian elimination, with pivot chanding
    for j in range(cols):
            # Iterate rows below diagonal element.
        for i in range(j+1,rows):
            if DU[i][j] != 0:
                # If above diagonal is 0, cannot be used for elimination. Pivot by finding row with largest diagonal element.
                if DU[j][j] == 0.0:
                    max_row = np.argmax(np.abs(DU[j:, j])) + j
                    DU[[j, max_row]] = DU[[max_row, j]]
                    P[[j, max_row]] = P[[max_row, j]]
                    if j > 0:
                        L[[j, max_row], :j] = L[[max_row, j], :j]
                # Calculate factor to 0-out row, track this factor in L
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