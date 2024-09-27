
def get_newton_root(fuction: callable, derivative: callable, x0: float, tol: float, max_iter: int) -> tuple:
    x = x0
    for i in range(max_iter):
        if derivative(x) == 0:
            return -1, i, x
            print("Derivative is zero, can't continue")
        
        if abs(fuction(x)) < tol:
            print(f"Converged in {i} iterations")
            return 0, i, x

        x = x - fuction(x) / derivative(x) # Use newton's update rule

    print(f"Did not converge in {max_iter} iterations")
    return -1, i, x
