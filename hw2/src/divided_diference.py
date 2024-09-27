import numpy as np

def print_dd_polynomial(divided_diff: np.array, X: np.array) -> None:
    N = len(X)
    for n in range(N):
        print(f"{divided_diff[0, n]:.2f}", end="")
        for j in range(n):
            print(f"(x - {X[j]:.2f})", end="")
        if n < N-1:
            print(" + ", end="")
        else:
            print("\n")

def divided_diff(X: np.array, Y: np.array) -> np.array:
    # Helper array for storing the divided difference table
    N = len(X) 
    divided_diff = np.zeros((N, N))
    divided_diff[:, 0] = Y
    # Build the divided difference table, each column is the kth divided difference. We will have N-1 columns.
    for i in range(1, N):
        for j in range(N-i):
            divided_diff[j,i] = (divided_diff[j, i-1] - divided_diff[j+1, i-1])/(X[j] - X[i+j])
    return divided_diff
    
def get_interpolated_value(value: float, X: np.array, divided_diff: np.array) -> float:
    N = len(X)
    interpolated_value = 0
    for n in range(N):
        coeff = divided_diff[0, n]
        term = 1
        # Build newton's polynomial coeff_k(x-x_0)(x-x_1)...(x-x_k-1)
        for i in range(n):
            term *= (value - X[i])
        interpolated_value += coeff * term
    return interpolated_value

if __name__ == "__main__":
    X = np.array([0, 1, -1])
    Y = np.array([1,0,4])
    dd = divided_diff(X, Y)
    print("Divided Difference, polynomial:")
    print_dd_polynomial(dd, X)

    x_test = 0.5
    print(f"Interpolated value at x = {x_test}")
    print(get_interpolated_value(x_test, X, dd))
