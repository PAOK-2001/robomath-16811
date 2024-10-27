import math
import numpy as np
import matplotlib.pyplot as plt

import src.divided_diference as DD
import src.newton_root_finder as NR
import src.muller_root_finder as MR
import src.common as C

def Q1():
    def f_b(x):
        return math.cos(math.pi * x)

    def f_c(x):
        return 2/(1+9*(x**2))

    def get_error(real: np.array, estimated: np.array) -> np.array:
        return np.max(np.abs(real - estimated))

    print("######################## Q1 ########################")
    # B)
    X = [0 ,1/8, 1/4, 3/8, 1/2]
    Y = [f_b(x) for x in X]
    dd_table = DD.divided_diff(np.array(X), np.array(Y))

    print("Found interpolating polynomial for cos(pi x):")
    DD.print_dd_polynomial(dd_table, X)

    val = DD.get_interpolated_value(3/10, X, dd_table)
    print(f"b) Interpolated value at x = 3/10. f(x) = {val}\n")

    # C)
    print(f"c) Real value for x = 0.07. {f_c(0.07)}")
    N = [2, 4, 40]
    for n in N:
        X = [(i*2/n) -1 for i in range(n+1)]
        Y = [f_c(x) for x in X]
        dd_table = DD.divided_diff(np.array(X), np.array(Y))

        val = DD.get_interpolated_value(0.07, X, dd_table)
        print(f"Interpolated value for x = 0.07 at n = {n}. \n f(x) = {val}")

    # D)
    errors = []
    N = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 40]
    for n in N:
        X = [(i*2/n) -1 for i in range(0,n+1)]
        Y = [f_c(x) for x in X]
        dd_table = DD.divided_diff(np.array(X), np.array(Y))
        eval_range = list(np.linspace(-1, 1, 10000)) 
        real, estimated = [], []
        # Evaluate for discretized range -1 to 1
        for value in eval_range:
            real.append(f_c(value))
            estimated.append(DD.get_interpolated_value(value, X, dd_table))

        
        error = get_error(real= np.array(real), estimated= np.array(estimated))
        errors.append(error)
            # plot function and points
        plt.cla()
        plt.plot(eval_range, real, c="green")
        plt.plot(eval_range, estimated, c="purple")
        plt.scatter(X, Y, c="orange")

        plt.title("Divided Diference Interpolation")
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.savefig(f"results/interpolated function_{n}.png")

    # Plot and save error vs n
    plt.cla()
    plt.plot(N, errors, c="purple")
    plt.scatter(N, errors, c="purple", marker="^")
    plt.xlabel("Data points (Degree of polynomial)") 
    plt.ylabel("Error")
    plt.title("Error Estimation")
    plt.savefig("results/error estimation.png")

    error_table = C.array_to_latex_table(N, errors, ("N", "Error"))
    print(error_table)
  
def Q2():
    print("######################## Q1 ########################")
    def f(x):
        return math.tan(x) - x
    
    def df(x):
        return -1 + 1/(math.cos(x)**2)

    lower_bound, upper_bound = C.initial_root_estimate(f, pivot= 15)
    print(f"Initial estimate for root: {lower_bound, upper_bound}")
    rc1, it1, lb = NR.get_newton_root(f, df, lower_bound, 1e-5, 1000)
    rc2, it2, up = NR.get_newton_root(f, df, upper_bound, 1e-5, 1000)
    print(f"Estimated roots: {lb, up}")

def Q3():
    # For p(x) = x**3 +x +1
    coeffs = [1, 1, 0, 1]
    f = MR.poly_from_coeffs(coeffs)
    x1 = 0
    x0, x2 = C.initial_root_estimate(f, pivot= 0)
    roots = MR.get_roots(coeffs, x0, x1, x2)
    print("Initial estimates for roots:", x0, x1, x2)
    print(f"Roots: {roots}")

def Q7(x_min = -3, x_max = 3, n_points = 1000):
    print("######################## Q7 ########################")
    import matplotlib.pyplot as plt
    
    x = np.linspace(x_min, x_max, n_points)
    y = np.linspace(x_min, x_max, n_points)

    X, Y = np.meshgrid(x, y)

    P = 2*X**2 + 2*Y**2 - 4*X -4*Y + 3
    Q = X**2 + Y**2 + 2*X*Y -5*X -3*Y +4


    plt.scatter(0.91594130, 0.29790, c="red")
    plt.scatter(1.70209, 1.0841, c="red")
    plt.contour(X, Y, Q, levels=[0], colors='orange')
    plt.contour(X, Y, P, levels=[0], colors='purple')
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title("Contour plot of P and Q")
    plt.savefig("results/contour_plot.png")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Solve questions for hw2")
    parser.add_argument("--question", type=int, help="Question number to solve", default=-1)
    args = parser.parse_args()
    if args.question == -1:
        print("Running all questions")
    else:
        print(f"Running question {args.question}")


    if args.question == 1:
        Q1()
    elif args.question == 2:
        Q2()
    elif args.question == 3:
        Q3()
    elif args.question == 7:
        Q7()
    
    else:
        Q1()
        Q2()
        Q3()
        Q7()
