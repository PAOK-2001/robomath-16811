import numpy as np
import matplotlib.pyplot as plt

def print_poly(coeffs: list):
    # In ascending order
    n = len(coeffs)
    for i in range(n):
        print(f"{coeffs[i]}x^{i}", end="")
        if i < n-1:
            print(" + ", end="")

def poly_from_coeffs(coeffs: list) -> callable:
    def poly(x: float) -> float:
        return np.polynomial.polynomial.polyval(x, coeffs)
    return poly

def plot_plane(coeff: tuple):
    a, b, c, d = coeff
    # Create a meshgrid of points
    x = np.linspace(-1, 1, 10)
    y = np.linspace(-1, 1, 10)
    

def load_point_cloud(path: str) -> np.ndarray:
    """
    Load point cloud from text file.
    """

    # Each row of the text file is a point in the point cloud.
    # Each column of the row is the x, y, z coordinates of the point.
    point_cloud = np.loadtxt(path) # shape: (n, 3)
    return point_cloud

def plot_point_cloud(pc, tag = None, save=False):
    import matplotlib.pyplot as plt
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(pc[:, 0], pc[:, 1], pc[:, 2])
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    if save:
        plt.savefig(f'results/{tag}.png')
        return
    plt.show()


def load_data_point(path):
    y = np.loadtxt(path)
    # X is i/100 for each element in y
    x = np.array([i/100 for i in range(len(y))])
    return np.column_stack((x, y))

