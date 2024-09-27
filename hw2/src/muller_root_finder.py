import numpy as np

def deflated_poly(coeffs: list, root: float) -> list:
    deflated_poly, _ = np.polynomial.polynomial.polydiv(coeffs, [-root, 1])
    return deflated_poly

def poly_from_coeffs(coeffs: list) -> callable:
    def poly(x: float) -> float:
        return np.polynomial.polynomial.polyval(x, coeffs)
    return poly

def mueller_root_finder(f: callable, guesses: float, max_iter: int = 10000, tol: float = 1e-8) -> float:
    iter = 0
    x0, x1, x2 = guesses
    while iter < max_iter:
        f2 = f(x2)
        f1 = f(x1)
        f0 = f(x0)
        # Calculate the step and function differences
        dx1, dx2 = x1 - x0, x2 - x1
        df1 = (f1 - f0) / dx1
        df2 = (f2 - f1) / dx2

        # Calculate the coefficients of the quadratic interpolation 
        c = f2
        a = (df2 - df1) / (dx2 + dx1)
        b = a * dx2 + df2

        # Calculate the denominator and pick value the maximize the denominator
        D = np.lib.scimath.sqrt(b**2 - 4 * a * c)
        if abs(b + D) > abs(b - D):
            denominator = b + D
        else:
            denominator = b - D
        # Check if denominator is zero, to avoid division by zero
        if denominator == 0:
            print("denominatorinator is zero")
            return None, iter

        root = 2 * c / denominator
        x3 = x2 - root

        if abs(root) < tol:
            return x3, iter
        
        x0, x1, x2 = x1, x2, x3
        iter += 1

    return x3, iter

def get_roots(coefficients, x0 = 1.0, x1 = 1.5, x2 = 2.0):
    # Perform muller root finding and deflation N times
    N = len(coefficients) - 1
    roots = []
    for n in range(N):
        root, _ = mueller_root_finder(poly_from_coeffs(coefficients), guesses= (x0, x1, x2))
        roots.append(root)
        print(f"Root {n}: {root}")
        coefficients = deflated_poly(coefficients, root)
    return roots

if __name__ == "__main__":
    x0, x1, x2 = 1.0, 1.5, 2.0
    coeffs = [-1,-1,0, 1]
    get_roots(coeffs)
    

        


