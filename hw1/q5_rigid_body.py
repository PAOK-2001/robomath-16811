"""q5_rigid_body.py

Author: Pablo Agustín Ortega Kral (portegak)
"""

import os
import numpy as np
import matplotlib.pyplot as plt


from typing import Tuple

def apply_transform(P, R, T):
    Q = (P @ R.T) + T
    return Q

def get_rotation_matrix3D(angle: float = 45):
    angle = np.radians(angle)
    return np.array([
        [np.cos(angle), -np.sin(angle), 0],
        [np.sin(angle), np.cos(angle), 0],
        [0, 0, 1]
    ])

def center_point(P_org):
    P_bar = np.mean(P_org, axis= 1)
    P = P_org - P_bar[:, np.newaxis]
    return P_bar, P

def get_transformation_params(P_original: np.array, Q_original: np.array) -> Tuple:
    # Get centroids and center points
    P_bar, P = center_point(P_original)
    Q_bar, Q = center_point(Q_original)

    C = P @ Q.T

    U, _ , Vh = np.linalg.svd(C)
    R = Vh.T @ U.T

    trans = Q_bar - (R @ P_bar)

    return R, trans


def get_test_set(n_points: int = 5, angle: float = 45, d: tuple = (1,2,3)):
    original_points = np.random.rand(n_points, 3)

    applied_rot = get_rotation_matrix3D(angle)
    applied_trans = np.array([d[0], d[1], d[2]])
    
    generated_points = apply_transform(original_points, applied_rot, applied_trans)

    return (original_points, generated_points, applied_rot, applied_trans)

def plot(P_original, Q_original, Q_prime, out = './results', tag = 'test'):
    os.makedirs(out, exist_ok =  True)
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(P_original[:, 0], P_original[:, 1], P_original[:, 2], c='purple', s = 30, marker = 'o', label='P')
    ax.scatter(Q_original[:, 0], Q_original[:, 1], Q_original[:, 2], c='green', s = 30, label='Q')
    ax.scatter(Q_prime[:, 0], Q_prime[:, 1], Q_prime[:, 2], c='orange',  marker = '^', s =40, label='Calculated transformation of P')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.legend()
    plt.savefig(os.path.join(out, f"rigid_transformation_{tag}.png"))


if __name__ == "__main__":
    P_original, Q_original, R_org, T_org = get_test_set(n_points= 20)
    R, T = get_transformation_params(P_original.T, Q_original.T)
    Q_calculated =  apply_transform(P_original, R, T)
    np.testing.assert_allclose(R, R_org, atol= 1e-8)
    np.testing.assert_allclose(T, T_org, atol= 1e-8)
    plot(P_original, Q_original, Q_prime= Q_calculated, tag= f"n_points{20}")
