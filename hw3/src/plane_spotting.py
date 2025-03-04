import numpy as np
from utils import load_point_cloud, plot_pc_and_plane, plot_many_pc_and_planes, plot_point_cloud

def get_plane_error(coeff, point_cloud, limit: bool = False):
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

def fit_plane_robust(point_cloud, samples  = 5, max_iter=100, tol=0.03):
    best_inliers = 0
    best_coeff = None
    inliers = []

    for _ in range(max_iter):
        # Randomly sample point cloud
        idx = np.random.choice(point_cloud.shape[0], min(samples, len(point_cloud)), replace=False)
        sampled_pc = point_cloud[idx]
        # Fit plane to sampled point cloud
        curr_coeff = fit_plane(sampled_pc)
        # Calculate inliers
        curr_inliers = get_plane_error(curr_coeff, point_cloud) < tol
        inliers.append(curr_inliers)
        point_cloud = point_cloud[~curr_inliers]
        in_value = np.sum(curr_inliers)
        if in_value > best_inliers:
            best_inliers = in_value
            best_coeff = curr_coeff
    return best_coeff

def fit_multiple_planes(point_cloud, samples= 7, max_iter = 100, ransac_iter=1000, tol=0.01, min_points = 1000):
    iter = 0
    pc = point_cloud.copy()
    planes = []

    while iter < max_iter and pc.shape[0] > samples:
        coeff = fit_plane_robust(pc, samples=samples, max_iter=ransac_iter, tol=tol)
        curr_inliers = get_plane_error(coeff, pc) < tol
        if np.sum(curr_inliers) > min_points:
            planes.append(coeff)
            print(pc.shape)
            pc = pc[~curr_inliers]
        iter += 1
    print(f"Found {len(planes)} planes")
    return planes

def score_planes(planes, point_cloud, tol=0.01):
    scores = []
    for plane in planes:
        pc = point_cloud.copy()
        curr_inliers = get_plane_error(plane, pc) < tol
        pc = pc[~curr_inliers]
        score = np.mean(get_plane_error(plane, pc))
        scores.append(score)
    return scores

if __name__ == "__main__":
    # Data path
    paths = {
        "part_a": "data/clear_table.txt",
        "part_b": "data/cluttered_table.txt",
        "part_c": "data/cluttered_table.txt",
        "part_d": "data/clean_hallway.txt",
        "part_e": "data/cluttered_hallway.txt"
    }

    # Part A
    pc_a = load_point_cloud(paths["part_a"])
    coeff = fit_plane(pc_a)
    a, b, c, d = coeff
    print(f"Fitted plane coefficients: {a}x + {b}y + {c}z + {d} = 0")
    plot_pc_and_plane(pc_a, coeff, "part_a", save=True)
    print("Average error: ", np.mean(get_plane_error(coeff, pc_a)))
    
    # Part B
    pc_b = load_point_cloud(paths["part_b"])
    coeff = fit_plane(pc_b)
    a, b, c, d = coeff
    print(f"Fitted plane coefficients: {a}x + {b}y + {c}z + {d} = 0")
    plot_pc_and_plane(pc_b, coeff, "part_b", save=True)
    print("Average error: ", np.mean(get_plane_error(coeff, pc_b)))

    # Part C
    pc_c = load_point_cloud(paths["part_c"])
    coeff = fit_plane_robust(pc_c, samples=7, max_iter=1000, tol=0.03)
    a, b, c, d = coeff
    print(f"Fitted plane coefficients: {a}x + {b}y + {c}z + {d} = 0")
    plot_pc_and_plane(pc_c, coeff, "part_c", save=True)
    print("Average error: ", np.mean(get_plane_error(coeff, pc_c)))

    # Part D
    # pc_d = load_point_cloud(paths["part_d"])
    # planes = []
    # while len(planes) != 4:
    #     planes = fit_multiple_planes(pc_d, samples=9, max_iter=100, ransac_iter=100, tol=0.03, min_points=1500)
    
    # # Plot all planes in the same graph with color coding
    # colors = ['r', 'g', 'b', 'c', 'm', 'y', 'k']
    # for i, coeff in enumerate(planes):
    #     a, b, c, d = coeff
    #     print(f"Fitted plane coefficients {i}: {a}x + {b}y + {c}z + {d} = 0")
    #     plot_pc_and_plane(pc_d, coeff, f"part_d_{i}", save=True, color=colors[i])
    #     print("Average error: ", np.mean(get_plane_error(coeff, pc_d)))
    
    # # Plot all planes together
    # plot_many_pc_and_planes(pc_d, planes, "part_d_all_planes", save=False, color=colors)

    # Part E
    tol = 0.03
    pc_e = load_point_cloud(paths["part_e"])
    plot_point_cloud(pc_e, "part_e_pc", save=False)
    planes = []
   
    planes = fit_multiple_planes(pc_e, samples=9, max_iter=400, ransac_iter=400, tol=tol, min_points=500)
    scores = score_planes(planes, pc_e, tol=tol)
    print("Scores: ", scores)
    # Plot all planes in the same graph with color coding
    colors = ['r', 'g', 'b', 'c', 'm', 'y', 'k']
    for i, (coeff, score) in enumerate(zip(planes, scores)):
        a, b, c, d = coeff
        print(f"Fitted plane coefficients {i}: {a}x + {b}y + {c}z + {d} = 0")
        title = f"Score: {score:.4f}"
        plot_pc_and_plane(pc_e, coeff, tag=f"part_e_{i}", save=True, color=colors[i % len(colors)], title_str=title)
        print("Average error: ", np.mean(get_plane_error(coeff, pc_e)))

    # Plot all planes together
    # plot_many_pc_and_planes(pc_e, planes, "part_e_all_planes", save=False, color=colors)





