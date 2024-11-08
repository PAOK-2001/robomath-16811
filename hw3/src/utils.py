import numpy as np
import matplotlib.pyplot as plt

def print_poly(coeffs: list):
    n = len(coeffs)
    for i in range(n):
        print(f"{coeffs[i]}x^{i}", end="")
        if i < n-1:
            print(" + ", end="")

def poly_from_coeffs(coeffs: list) -> callable:
    def poly(x: float) -> float:
        return np.polynomial.polynomial.polyval(x, coeffs)
    return poly

def plot_plane(coeff: tuple, x, y, ax: plt.Axes, color='r', z_limit = None, step: float = 0.001):
    a, b, c, d = coeff
    x = np.arange(np.min(x), np.max(x), step)
    y = np.arange(np.min(y), np.max(y), step)
    X, Y = np.meshgrid(x, y)
    Z = (-a*X - b*Y - d) / c
    if z_limit:
        z_min, z_max = z_limit
        Z[Z < z_min] = np.nan
        Z[Z > z_max] = np.nan

    ax.plot_surface(X, Y, Z, alpha=0.5, antialiased=False, color=color)

def load_point_cloud(path: str) -> np.ndarray:
    """
    Load point cloud from text file.
    """
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

def plot_pc_and_plane(pc, coeff, tag = None, color = 'r', save=False, title_str: str = None):
    import matplotlib.pyplot as plt
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(pc[:, 0], pc[:, 1], pc[:, 2])
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    z_limit = (np.min(pc[:, 2]), np.max(pc[:, 2]))
    plot_plane(coeff, x = pc[:,0], y= pc[:,1], z_limit= z_limit, color=color, ax= ax)
    
    if title_str:
        ax.set_title(title_str)
    
    if save:
        plt.savefig(f'results/{tag}.png')
        plt.close()
        return
    plt.show()


def plot_many_pc_and_planes(pc_d, planes, tag, save=True, color: list = ['r', 'g', 'b', 'y', 'c', 'm']):
    import matplotlib.pyplot as plt
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(pc_d[:, 0], pc_d[:, 1], pc_d[:, 2])
    
    # Set limits to point cloud limits
    x_min, x_max = np.min(pc_d[:, 0]), np.max(pc_d[:, 0])
    y_min, y_max = np.min(pc_d[:, 1]), np.max(pc_d[:, 1])
    z_min, z_max = np.min(pc_d[:, 2]), np.max(pc_d[:, 2])
    limits = (x_min, x_max), (y_min, y_max), (z_min, z_max)
    ax.set_xlim([x_min, x_max])
    ax.set_ylim([y_min, y_max])
    ax.set_zlim([z_min, z_max])

    for i, plane in enumerate(planes):
        plot_plane(plane, x = pc_d[:,0], y= pc_d[:,1], z_limit= (z_min, z_max) ,ax= ax, color = color[i])
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    
    ax.set_ylim([y_min, y_max])
    ax.set_zlim([z_min, z_max])
    # Set aspect ratio to be equal
    # ax.set_box_aspect([np.ptp(pc_d[:, 0]), np.ptp(pc_d[:, 1]), np.ptp(pc_d[:, 2])])
    
    if save:
        plt.savefig(f'results/{tag}.png')
        plt.close()
        return
    plt.show()
 
def load_data_point(path):
    y = np.loadtxt(path)
    x = np.array([i/100 for i in range(len(y))])
    return np.column_stack((x, y))

