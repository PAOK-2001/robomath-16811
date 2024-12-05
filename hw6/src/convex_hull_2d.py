import os
import numpy as np
from enum import IntEnum
from tqdm import tqdm
import matplotlib.pyplot as plt

def plot_convex_hull(points: np.array, hull: np.array, ax: plt.Axes = None):
    if ax is None:
        fig, ax = plt.subplots()
    ax.scatter(points[:, 0], points[:, 1], zorder=0)
    # ax.plot(hull[:, 0], hull[:, 1], c='g', zorder=0)
    ax.fill(hull[:, 0], hull[:, 1], 'g', alpha=0.3, zorder=1, edgecolor='g', linewidth=2)
    return ax

def generate_random_points(n: int, seed: int = 0):
    np.random.seed(seed)
    points = np.random.uniform(low=-10, high=100, size=(n,2))
    return points

class Orientation(IntEnum):
    COLINEAR = 0
    CLOCKWISE = 1
    COUNTER_CLOCKWISE = 2

# Helper function to get the polar angle between two points
def get_polar_angle(p1: np.array, p2: np.array) -> float:
    '''
    Returns the polar angle between two points.
    '''
    angle = np.arctan2(p2[1] - p1[1], p2[0] - p1[0])
    return angle
# Helper function to compute orientation of three points
def get_triplet_orientation(p1, p2, p3):
    val = np.cross(p2 - p1, p3 - p1)
    if val == 0:
        return Orientation.COLINEAR
    elif val > 0:
        return Orientation.COUNTER_CLOCKWISE
    elif val < 0:
        return Orientation.CLOCKWISE

def convex_hull_2D(points: np.array) -> np.array:
    '''
    Computes the convex hull of a set of 2D points using Graham Scan.
    '''
    # Pick the point with the smallest y-coordinate as the starting point
    start_idx = np.argmin(points[:, 1])
    start_point = points[start_idx]
    # Sort the points by polar angle with respect to the starting point
    sorted_points = sorted(points, key=lambda x: get_polar_angle(start_point, x))
    sorted_points = np.array(sorted_points)
    # Initialize the hull stack
    hull_stack = [
        sorted_points[0], sorted_points[1], sorted_points[2]
    ]
    for p1 in sorted_points[3:]:
        while len(hull_stack) >= 2:
            p2, p3 = hull_stack[-2], hull_stack[-1]
            #  Pop points that make a clockwise turn
            if get_triplet_orientation(p1, p2, p3) is Orientation.CLOCKWISE:
                hull_stack.pop()
            elif get_triplet_orientation(p1, p2, p3) is Orientation.COUNTER_CLOCKWISE:
                break
            else:
                break
        hull_stack.append(p1)
    hull = np.array(hull_stack)
    return hull


if __name__ == "__main__":
    OUT_DIR = os.path.join('output', 'convex_hull')
    os.makedirs(OUT_DIR, exist_ok=True)

    n_cases = [5, 27, 31,  143, 200, 1000, 5000]
    for n in tqdm(n_cases, desc="Generating convex hulls"):
        points = generate_random_points(n, seed = 56)
        hull = convex_hull_2D(points)
        plot_convex_hull(points, hull)
        file_out = os.path.join(OUT_DIR, f'convex_hull_n_points{n}.png')
        plt.savefig(file_out)
        plt.close()
    