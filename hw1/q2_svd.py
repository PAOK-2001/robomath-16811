import numpy as np

def pmatrix(a):
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


def get_svd(A: np.array, print_result: bool = True):
    rows, cols = A.shape
    U,S, Vh = np.linalg.svd(A)
    S = np.diag(S)
    if len(S) < rows:
        S = np.pad(S, ((0, rows - S.shape[0]), (0, 0)), mode='constant', constant_values=0)

    np.testing.assert_allclose(A, U@S@Vh,atol=1e-10)
    if print_result:
        print("U")
        print(pmatrix(U))

        print("S")
        print(pmatrix(S))

        print("Vt")
        print(pmatrix(Vh))

    return U,S, Vh

if __name__ == "__main__":
    A = np.array([ 
        [1,1,1],
        [10,2,9],
        [8,0,7]
    ], dtype= np.float64)
    # A = np.array([ 
    #     [5,-5,0,0],
    #     [5,5,5,0],
    #     [0,-1,4,1],
    #     [0,4,-1,2],
    #     [0,0,2,1]
    # ], dtype= np.float64)
    get_svd(A)