"""
q1_test_ldu.py

Author: Pablo Ortega Kral (portegak)
"""

import unittest
import numpy as np

from q1_ldu_decomposition import ldu
np.set_printoptions(suppress=True)

def pmatrix(a):
    # Helper functio
    """Returns a LaTeX bmatrix

    :a: numpy array
    :returns: LaTeX bmatrix as a string
    """
    if len(a.shape) > 2:
        raise ValueError('bmatrix can at most display two dimensions')
    lines = str(a).replace('[', '').replace(']', '').splitlines()
    rv = [r'\begin{pmatrix}']
    rv += ['  ' + ' & '.join(l.split()) + r'\\' for l in lines]
    rv +=  [r'\end{pmatrix}']

    return '\n'.join(rv)


def get_random_mat() -> np.array:
    # Adapted from: https://stackoverflow.com/questions/73426718/generating-invertible-matrices-in-numpy-tensorflow
    # size = np.random.randint(3, 11)
    size = 5
    matrix = np.random.randint(low= -10, high= 10, size= (size, size)).astype(np.float64)
    mx = np.sum(np.abs(matrix), axis=1)
    np.fill_diagonal(matrix, mx)
    return matrix

def print_paldu(P,A,L,D,U):
    print("P:")
    print(pmatrix(P))

    print("A:")
    print(pmatrix(A))

    print("L:")
    print(pmatrix(L))

    print("D:")
    print(pmatrix(D))

    print("U:")
    print(pmatrix(U))

class TestLDU(unittest.TestCase):
    def __init__(self, methodName='runTest'):
        super().__init__(methodName)

        self.fixed_examples = [
             np.array([ 
                [1,-2,1],
                [1,2,2],
                [2,3,4]
            ], dtype= np.float64),
            np.array([ 
                [1,1,0],
                [1,1,2],
                [4,2,3]
            ], dtype= np.float64),
        ]

        self.random_cases = 2

    def test_fixed_examples(self):
        for A in self.fixed_examples:
            P, L, D, U = ldu(A)
            np.testing.assert_allclose(P@A, L@D@U)
            print_paldu(P,A,L,D,U)

    def test_random(self, print = True):
        for _ in range(self.random_cases):
            A = get_random_mat()
            P, L, D, U = ldu(A)
            np.testing.assert_allclose(P@A, L@D@U,atol=1e-7)
            if print: 
                print_paldu(P,A,L,D,U)


if __name__ == 'main':
    unittest.main()