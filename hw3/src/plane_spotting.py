import numpy as np
from utils import load_point_cloud, plot_pc_and_plane

def get_plane_error(coeff, point_cloud):
    a, b, c, d = coeff
    x = point_cloud[:, 0]
    y = point_cloud[:, 1]
    z = point_cloud[:, 2]

    return np.abs(a*x + b*y + c*z + d) / np.sqrt(a**2 + b**2 + c**2)

def fit_plane(point_cloud):
    """
    Fit a plane to the given point cloud using least squares.
    """
    # Asume the plane is defined by ax + by + cz + d = 0 and
    d = 1

    x = point_cloud[:, 0]
    y = point_cloud[:, 1]
    z = point_cloud[:, 2]

    pc = np.column_stack((x, y, z))
    obs = -np.ones(pc.shape[0])

    coeff = np.linalg.pinv(pc.T @ pc) @ (pc.T @ obs)
    coeff = np.append(coeff, d)
    return tuple(coeff)

if __name__ == "__main__":
    # Data path
    paths = {
        "part_a": "data/clear_table.txt",
        "part_b": "data/cluttered_table.txt",
        "part_c": "data/cluttered_table.txt",
        "part_d": "data/clean_hallway.csv",
    }

    # Load point cloud
    pc_a = load_point_cloud(paths["part_a"])
    coeff = fit_plane(pc_a)
    plot_pc_and_plane(pc_a, coeff)
    print("Average error: ", np.mean(get_plane_error(coeff, pc_a)))
    # plot_point_cloud(pc_a)
