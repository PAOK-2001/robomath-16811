
"""
q1_ldu_decomposition.py

Author: Pablo Ortega Kral (portegak)
"""

import numpy as np
import scipy

from typing import Tuple

def matrix_to_latex(matrix: np.array) -> str:
    rows, cols = matrix.shape
    latex = "\\begin{bmatrix}\n"
    for i in range(rows):
        for j in range(cols):
            if matrix[i][j] == 0.0:
                if j == cols - 1:
                    latex += f"{0} \\\\"
                else:
                    latex += f"{0} & "
            else:
                if j == cols - 1:
                    latex += f"{matrix[i][j]} \\\\"
                else:
                    latex += f"{matrix[i][j]} & "
        latex += "\n"
    latex += "\\end{bmatrix}"
    return latex


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
    A1 = np.array([ 
        [10,-10,0],
        [0,-4,2],
        [2,0,-5]
    ], dtype= np.float64)

    A2 =  np.array([
    [5, -5, 0, 0],
    [5, 5, 5, 0],
    [0, -1, 4, 1],
    [0, 4, -1, 2],
    [0, 0, 2, 1]
    ], dtype= np.float64)

    A3 = np.array([ 
        [1,1,1],
        [10,2,9],
        [8,0,7]
    ], dtype= np.float64)

    problem_set = [ A3]
    for A in problem_set:
        row, col = A.shape

        if row != col:
            # Use numpy to calculate LDU decomposition
            P, L, DU = scipy.linalg.lu(A)
            U = np.zeros(shape=(row,col), dtype= np.float64)
            D = np.diag(np.diag(DU))
            for i in range(row):
                diag = D[i,i] 
                if diag != 0.0:
                    U[i] = DU[i]/ diag
            print("---LDU Decomposition---")

        else:
            P, L, D, U = ldu(A)

        # assert (P@A == L@D@U).all(), "Factorization does not equal original matrix!"c
        print("---LDU Decomposition---")

        print("P:")
        print(matrix_to_latex(P))

        print("A:")
        print(matrix_to_latex(A))

        print("L:")
        print(matrix_to_latex(L))

        print("D:")
        print(matrix_to_latex(D))

        print("U:")
        print(matrix_to_latex(U))