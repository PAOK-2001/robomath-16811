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

def plot_plane(coeff: tuple, x, y, ax: plt.Axes):
    # Assume the plane is defined by ax + by + cz + d = 0
    a, b, c, d = coeff
    x = np.linspace(np.min(x), np.max(x), 10)
    y = np.linspace(np.min(y), np.max(y), 10)
    X, Y = np.meshgrid(x, y)
    Z = (-a*X - b*Y - d) / c
    ax.plot_surface(X, Y, Z, alpha=0.5)

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

def plot_pc_and_plane(pc, coeff, tag = None, save=False):
    import matplotlib.pyplot as plt
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(pc[:, 0], pc[:, 1], pc[:, 2])
    plot_plane(coeff, x = pc[:,0], y= pc[:,1] ,ax= ax)
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

