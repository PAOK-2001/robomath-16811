import numpy as np
from scipy.ndimage.morphology import distance_transform_edt
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

BOUNDS = 0, 100


def zero_gradient(path, grad_values, tol = 1e-5):
    return np.allclose(grad_values, 0, atol=tol)

def get_path_grad(path):
    path_grad = np.empty(path.shape)
    for i in range(1, path.shape[0]):
        path_grad[i] = path[i] - path[i-1]
    return path_grad

def get_biliteral_path_grad(path):
    path_grad = np.empty(path.shape)
    for i in range(1, path.shape[0]-1):
        path_grad[i] = (path[i] - path[i-1])  + (path[i] - path[i+1])
    return path_grad

def optmize_path_gd(path: np.array, grad : tuple, convergence_criteria : callable, 
                    obstacle_cost: np.array, max_iter: int = 10000, learning_rate: float = 0.1):
    gx, gy = grad
    for i in range(max_iter):
        # Read gradient from precomputed values
        x, y = path.astype(int).T
        grad_values = np.array([gx[x, y], gy[x, y]]).T
        # Update path
        path[1:-1] = path[1:-1] - learning_rate * grad_values[1:-1] 
        path = np.clip(path, BOUNDS[0], BOUNDS[1])
        if i == 1:
            plot_scene(path, obstacle_cost, tag="first_iter_simple_gradient", title=f"Simple Gradient Descent (Iter {i})")
        if convergence_criteria(path, grad_values, tol = 1e-5):
            break
    print(f"Optimization finished after {i} iterations")
    return path, i

def optmize_path_smooth(path: np.array, grad : tuple, convergence_criteria : callable, 
                    obstacle_cost: np.array, max_iter: int = 10000, learning_rate: float = 0.1, smooth_factor: tuple = (0.8, 4)):
    gx, gy = grad
    for i in range(max_iter):
        # Read gradient from precomputed values
        x, y = path.astype(int).T
        grad_values = np.array([gx[x, y], gy[x, y]]).T
        path_grad = get_path_grad(path)
        # Update path
        path[1:-1] = path[1:-1] - learning_rate * (smooth_factor[0]*grad_values + smooth_factor[1]* path_grad)[1:-1]
        path = np.clip(path, BOUNDS[0], BOUNDS[1])
        if convergence_criteria(path, grad_values, tol = 1e-5):
            break

    print(f"Optimization finished after {i} iterations")
    return path, i

def optmize_path_smooth_v2(path: np.array, grad : tuple, convergence_criteria : callable, 
                    obstacle_cost: np.array, max_iter: int = 10000, learning_rate: float = 0.1, smooth_factor: tuple = (0.8, 4)):
    gx, gy = grad
    for i in range(max_iter):
        # Read gradient from precomputed values
        x, y = path.astype(int).T
        grad_values = np.array([gx[x, y], gy[x, y]]).T
        path_grad = get_biliteral_path_grad(path)
        # Update path
        path[1:-1] = path[1:-1] - learning_rate * (smooth_factor[0]*grad_values + smooth_factor[1]* path_grad)[1:-1]
        path = np.clip(path, BOUNDS[0], BOUNDS[1])
        if convergence_criteria(path, grad_values, tol = 1e-5):
            break

    print(f"Optimization finished after {i} iterations")
    return path, i
               
def main():
    obstacle_cost = generate_cost()
    gx, gy = np.gradient(obstacle_cost)

    start_point = np.array([10, 10])
    end_point = np.array([90, 90])
    vector = end_point - start_point

    num_pts = 300
    initial_path = start_point + \
        np.outer(np.linspace(0, 1, num_pts), vector)

    plot_scene(initial_path, obstacle_cost, tag= "initial_path")
    # FURTHER CODE HERE
    test_iter = [100, 200, 500]
    for i in test_iter:
        smooth_optimized_path, iterations = optmize_path_smooth(initial_path, (gx, gy), convergence_criteria = zero_gradient, obstacle_cost= obstacle_cost,  max_iter=i)
        plot_scene(smooth_optimized_path, obstacle_cost, tag= f"smooth_gradient_iter_{i}", title=f"Smooth Gradient Descent (Iter {iterations})")
    
    test_iter = [100, 200, 500, 1000, 5000]
    for i in test_iter:
        simple_optimized_path, iterations = optmize_path_gd(initial_path, (gx, gy), convergence_criteria = zero_gradient, obstacle_cost= obstacle_cost, max_iter=i)
        plot_scene(simple_optimized_path, obstacle_cost, tag= f"simple_gradient_{i}", title=f"Simple Gradient Descent (Iter {iterations})")

        smooth_optimized_path, iterations = optmize_path_smooth_v2(initial_path, (gx, gy), convergence_criteria = zero_gradient, obstacle_cost= obstacle_cost,  max_iter=i)
        plot_scene(smooth_optimized_path, obstacle_cost, tag= f"both_smooth_gradient_iter_{i}_v2", title=f"Both endpoint smoothing (Iter {iterations})")

def generate_cost():
    n = 101
    obstacles = np.array([[20, 30], [60, 40], [70, 85]])
    epsilon = np.array([[25], [20], [30]])
    obstacle_cost = np.zeros((n, n))
    for i in range(obstacles.shape[0]):
        t = np.ones((n, n))
        t[obstacles[i, 0], obstacles[i, 1]] = 0
        t_cost = distance_transform_edt(t)
        t_cost[t_cost > epsilon[i]] = epsilon[i]
        t_cost = (1 / (2 * epsilon[i])) * (t_cost - epsilon[i])**2
        obstacle_cost += + t_cost
    return obstacle_cost

def get_values(path, cost):
    x, y = path.astype(int).T
    return cost[x, y].reshape((path.shape[0], 1))

def plot_scene(path, cost, tag="plot", title="Paths", plot_3d=False):
    values = get_values(path, cost)

    # Plot 2D
    plt.imshow(cost.T)
    plt.plot(path[1:-1, 0], path[1:-1, 1], "ro", label="Path")
    plt.plot(path[0, 0], path[0, 1], "go", label="Start Point")
    plt.plot(path[-1, 0], path[-1, 1], "bo", label="End Point")
    plt.title(title)
    plt.legend()
    plt.savefig(f"results/{tag}_2d.png")
    plt.close()

    # Plot 3D
    if plot_3d:
        fig3d = plt.figure()
        ax3d = fig3d.add_subplot(111, projection="3d")
        xx, yy = np.meshgrid(range(cost.shape[1]), range(cost.shape[0]))
        ax3d.plot_surface(xx, yy, cost.T, cmap=plt.get_cmap("coolwarm"))
        ax3d.scatter(path[1:-1, 0], path[1:-1, 1], values[1:-1], s=20, c="r", label="Path")
        ax3d.scatter(path[0, 0], path[0, 1], values[0], s=50, c="g", label="Start Point")
        ax3d.scatter(path[-1, 0], path[-1, 1], values[-1], s=50, c="b", label="End Point")
        ax3d.set_title(title)
        ax3d.legend()
        plt.savefig(f"results/{tag}_3d.png")
        plt.close()


if __name__ == "__main__":
    main()