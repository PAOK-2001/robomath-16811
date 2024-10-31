import pandas as pd
import numpy as np

def get_error(y: list, y_hat: list) -> float:
    y = np.array(y)
    y_hat = np.array(y_hat)
    return y - y_hat

def euler_solver(f:callable, x0: float, y0: float, step: float, n_iter: int) -> tuple:
    x = [x0]
    y = [y0]
    for i in range(1, n_iter+1):
        x.append(x0 + i*step)
        y.append(y[i-1] + step*f(x[i-1], y[i-1]))
    return x, y

def runge_kutta4_solver(f: callable, x0: float, y0: float, step: float, n_iter: int) -> tuple:
    x = [x0]
    y = [y0]
    for i in range(1, n_iter+1):
        x.append(x0 + i*step)
        # Calculate runge kutta constants
        k1 = step*f(x[i-1], y[i-1])
        k2 = step*f(x[i-1] + step/2, y[i-1] + k1/2)
        k3 = step*f(x[i-1] + step/2, y[i-1] + k2/2)
        k4 = step*f(x[i-1] + step, y[i-1] + k3)
        y_hat = y[i-1] + (k1 + 2*k2 + 2*k3 + k4)/6
        y.append(y_hat)
    return x, y

def adams_bashforth_solver(f: callable, x0: list, y0: list, step: float, n_iter: int) -> tuple:
    x = x0
    y = y0
    for i in range(4, n_iter+4):
        x.append(x0[0] + i*step)
        y_hat = y[i-1] + step/24*(55*f(x[i-1], y[i-1]) - 59*f(x[i-2], y[i-2]) + 37*f(x[i-3], y[i-3]) - 9*f(x[i-4], y[i-4]))
        y.append(y_hat)
    return x[3:], y[3:]

if __name__ == "__main__":
    def diff_eq(x, y):
        return 1/(3*y**2)
    
    def exact_sol(x):
        return x**(1/3)
    
    #----------------- Euler -----------------
    x0 = 1
    y0 = 1
    step = -0.05
    n_iter = int(1/-step)
    print(f"Performing Euler Solver with step size: {step} and n_iter: {n_iter}")
    x, y_euler = euler_solver(diff_eq, x0, y0, step, n_iter)
    fn_euler = [diff_eq(i, y) for i, y in zip(x, y_euler)]
    exact = [exact_sol(i) for i in x]

    euler_error = get_error(exact, y_euler)

    #----------------- Runge Kutta 4 -----------------
    x0 = 1
    y0 = 1
    step = -0.05
    n_iter = int(1/-step)
    print(f"Performing Runge Kutta 4 Solver with step size: {step} and n_iter: {n_iter}")
    _, y_runge_kutta = runge_kutta4_solver(diff_eq, x0, y0, step, n_iter)
    fn_runge_kutta = [diff_eq(i, y) for i, y in zip(x, y_runge_kutta)]
    runge_jutta_error = get_error(exact, y_runge_kutta)

    # #----------------- Adams Bashforth -----------------
    x0 = [1, 1.05, 1.1, 1.15]
    y0 = [1, 1.01639635681485, 1.03228011545637, 1.04768955317165]
    x0, y0 = x0[::-1], y0[::-1]
    step = -0.05
    n_iter = int(1/-step)
    print(f"Performing Adams Bashforth Solver with step size: {step} and n_iter: {n_iter}")
    x, y_adams = adams_bashforth_solver(diff_eq, x0, y0, step, n_iter)
    fn_adams = [diff_eq(i, y) for i, y in zip(x, y_adams)]
    adams_error = get_error(exact, y_adams)


    data = {
        # "x": x,
        "Euler": y_euler,
        "Euler f(x,y)": fn_euler,
        "Euler Error": euler_error,
        "Runge Kutta": y_runge_kutta,
        "Runge Kutta f(x,y)": fn_runge_kutta,
        "Runge Kutta Error": runge_jutta_error,
        "Adams Bashforth": y_adams,
        "Adams Bashforth f(x,y)": fn_adams,
        "Adams Bashforth Error": adams_error,
        "Exact Solution": exact
    }

    df = pd.DataFrame(data)
    # Create a multi-index for the columns
    df.columns = pd.MultiIndex.from_tuples([
        # ("", "x"),
        ("Euler", "y"),
        ("Euler", "f(x,y)"),
        ("Euler", "Error"),
        ("Runge Kutta", "y"),
        ("Runge Kutta", "f(x,y)"),
        ("Runge Kutta", "Error"),
        ("Adams Bashforth", "y"),
        ("Adams Bashforth", "f(x,y)"),
        ("Adams Bashforth", "Error"),
        ("Exact", "Solution")
    ])

    def custom_float_format(x):
        if isinstance(x, float):
            if abs(x) < 1e-5 or abs(x) >= 1e5:
                return f"{x:.2e}"
            return f"{x:.2f}"
        return x

    table = df.to_latex(index=False, multirow=True, float_format=custom_float_format)
    import matplotlib.pyplot as plt

    plt.figure(figsize=(10, 6))
    plt.plot(x, exact, label='Exact Solution', color='black', linestyle='--')
    plt.plot(x, y_euler, label='Euler Method')
    plt.plot(x, y_runge_kutta, label='Runge Kutta Method')
    plt.plot(x, y_adams, label='Adams Bashforth Method')

    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Comparison of Differential Equation Solvers')
    plt.legend()
    plt.grid(True)
    plt.savefig('results/solver_comparison.png')
    plt.show()
    with open('results/table.txt', 'w') as f:
        f.write(table)





