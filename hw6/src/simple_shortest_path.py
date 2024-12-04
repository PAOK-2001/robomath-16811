import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as MplPolygon
class Polygon:
    def __init__(self, vertices: np.array):
        self.vertices = vertices
        self.color = 'red'

    def set_color(self, color):
        self.color = color

    def get_color(self):
        return self.color
    
class Map:
    def __init__(self, bounds: tuple = None, goal: np.array = None, start: np.array = None):
        self.obstacles = []
        self.bounds = bounds
        self.goal = None
        self.start = None
        self.robot_pos = None

    def add_obstacle(self, polygon):
        self.obstacles.append(polygon)

    def init_random_map(
        self, 
        bounds: tuple, 
        obstacle_size: tuple,
        n_obstacles: int, 
        seed: int = 42,
        vertix_range: tuple = (4, 20)
    ):
        
        np.random.seed(seed)
        min_x, min_y, max_x, max_y = bounds
        min_size, max_size = obstacle_size

        self.start = np.random.uniform([min_x, min_y], [max_x, max_y])
        self.goal = np.random.uniform([min_x, min_y], [max_x, max_y])
        self.robot_pos = self.start

        for _ in range(n_obstacles):
            n_vertices = np.random.randint(vertix_range[0], vertix_range[1])  # Random number of vertices
            center_x = np.random.uniform(min_x, max_x)
            center_y = np.random.uniform(min_y, max_y)
            size = np.random.uniform(min_size, max_size)
            angles = np.sort(np.random.uniform(0, 2 * np.pi, n_vertices))
            vertices = np.array([
                [center_x + size * np.cos(angle), center_y + size * np.sin(angle)]
                for angle in angles
            ])
            obstacle = Polygon(vertices)
            obstacle.set_color('black')
            self.add_obstacle(obstacle)

    def visualize(self, ax: plt.Axes = None, show: bool =  True, save: bool = False, filename: str = 'map.png'):
        if ax is None:
            fig, ax = plt.subplots()
        # Plot obstacles
        for obstacle in self.obstacles:
            polygon = MplPolygon(obstacle.vertices, closed=True, edgecolor=obstacle.get_color(), facecolor=obstacle.get_color(), alpha=0.5)
            ax.add_patch(polygon)
        # Plot start and goal
        if self.start is not None and self.goal is not None:
            ax.plot(self.start[0], self.start[1], 'bo', markersize=10, color='#F6FADF', label='Start')
            ax.plot(self.goal[0], self.goal[1], 'go', markersize=10, color='#B9DDB3', label='Goal')

        if self.robot_pos is not None:
            ax.plot(self.robot_pos[0], self.robot_pos[1], 'rs', markersize=10, color='#5794A1', label='Robot')
        
        if self.bounds is not None:
            min_x, min_y, max_x, max_y = self.bounds
            ax.set_xlim(min_x, max_x)
            ax.set_ylim(min_y, max_y)
        else:
            ax.autoscale()
        ax.set_aspect('equal')
        ax.legend()
        plt.show()

if __name__ == '__main__':
    bounds = (-10, -10, 10, 10)
    map = Map(bounds=bounds)
    map.init_random_map(bounds=bounds, obstacle_size=(2, 2), n_obstacles=10)
    map.visualize()