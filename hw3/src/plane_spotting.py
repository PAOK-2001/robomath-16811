import numpy as np
from utils import load_point_cloud, plot_point_cloud

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

    coeff = np.linalg.lstsq(pc, obs, rcond=None)[0]
    a, b, c = coeff
    print(f"Plane equation: {a}x + {b}y + {c}z + {d} = 0")

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
    plot_point_cloud(pc_a)
    fit_plane(pc_a)
