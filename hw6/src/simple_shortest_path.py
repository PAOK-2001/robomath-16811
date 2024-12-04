import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as MplPolygon
from convex_hull_2d import get_triplet_orientation, Orientation

def on_segment(p1, p2, q):
    # Check if q lies on the line segment p1p2
    if (q[0] <= max(p1[0], p2[0]) and q[0] >= min(p1[0], p2[0]) and
        q[1] <= max(p1[1], p2[1]) and q[1] >= min(p1[1], p2[1])):
        return True


def do_lines_intersect(l1: np.array, l2: np.array) -> bool:
    '''
    Check if two lines intersect.
    '''
    p1, q1 = l1[0], l1[1]
    p2, q2 = l2[0], l2[1]

    o1 = get_triplet_orientation(p1, q1, p2)
    o2 = get_triplet_orientation(p1, q1, q2)
    o3 = get_triplet_orientation(p2, q2, p1)
    o4 = get_triplet_orientation(p2, q2, q1)

    # Base case for line intersection
    if o1 != o2 and o3 != o4:
        return True

    return False
    

class Polygon:
    def __init__(self, vertices: np.array):
        self.vertices = vertices
        self.color = 'red'

    def set_color(self, color):
        self.color = color

    def get_color(self):
        return self.color

    def contains_point(self, point: np.array):
        x, y = point[0], point[1]
        vertix = self.vertices[0]
        intersection_count = 0 

        # Loop through all vertices
        for i in range(1, len(self.vertices) + 1):
            next_vertix = self.vertices[i % len(self.vertices)]
            # Check if  point is within bounds
            if y > min(vertix[1], next_vertix[1]) and y <= max(vertix[1], next_vertix[1]):
                if  x <= max(vertix[0], next_vertix[0]):
                    intersect_point = (y - vertix[1]) * (next_vertix[0] - vertix[0]) / (next_vertix[1] - vertix[1]) + vertix[0]
                    if vertix[0] == next_vertix[0] or x <= intersect_point:
                        intersection_count += 1
            vertix = next_vertix

        return intersection_count % 2 == 1 # Odd number of intersections means point is inside polygon

class Map:
    def __init__(self, bounds: tuple = None, goal: np.array = None, start: np.array = None):
        self.obstacles = []
        self.bounds = bounds
        self.goal = None
        self.start = None
        self.robot_pos = None
        self.nodes = None
        self.edges = None

    def add_obstacle(self, polygon):
        self.obstacles.append(polygon)

    def init_random_map(
        self, 
        bounds: tuple, 
        obstacle_size: tuple,
        n_obstacles: int, 
        seed: int = 42,
        vertix_range: tuple = (4, 20),
        place_goal_in_obstacle: bool = False
    ):
        
        np.random.seed(seed)
        min_x, min_y, max_x, max_y = bounds
        min_size, max_size = obstacle_size

        
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

        self.start = np.random.uniform([min_x, min_y], [max_x, max_y])
        if place_goal_in_obstacle:
            obstacle = np.random.choice(self.obstacles)
            min_x, min_y = np.min(obstacle.vertices, axis=0)
            max_x, max_y = np.max(obstacle.vertices, axis=0)
            self.goal = np.random.uniform([min_x, min_y], [max_x, max_y])
        else:
            self.goal = np.random.uniform([min_x, min_y], [max_x, max_y])
        self.robot_pos = self.start        
        
    def is_valid_edge(self, edge):
        start, end = edge
        for obstacle in self.obstacles:
            for k in range(len(obstacle.vertices)):
                v1 = obstacle.vertices[k]
                v2 = obstacle.vertices[(k + 1) % len(obstacle.vertices)]
                if (np.array_equal(start, v1) and np.array_equal(end, v2)) or (np.array_equal(start, v2) and np.array_equal(end, v1)):
                    return True

        for obstacle in self.obstacles:
            for i in range(len(obstacle.vertices)):
                v1 = obstacle.vertices[i]
                v2 = obstacle.vertices[(i + 1) % len(obstacle.vertices)]  # Wrap around                
                # Check if the edge intersects the polygon's edge
                if do_lines_intersect([start, end], [v1, v2]):
                    return False
        return True
    
    def build_routing_graph(self):
        # Consider the polygon, start and goal as nodes
        nodes = [self.start, self.goal]
        obstacle_nodes = []
        for obstacle in self.obstacles:
            for vertix in obstacle.vertices:
                obstacle_nodes.append(vertix)

        edges = []
        nodes.extend(obstacle_nodes)

        for i, node in enumerate(nodes):
            for j, neighbor in enumerate(nodes):
                if i == j:
                    continue
                edge = (node, neighbor)
                if self.is_valid_edge(edge):
                    edges.append(edge)

        self.nodes = nodes
        self.edges = edges

    def get_shortest_path(self):
        # Check if we are given valid start and goal
        for obstacle in self.obstacles:
            if obstacle.contains_point(self.start) or obstacle.contains_point(self.goal):
                raise ValueError('Start or goal is inside an obstacle')
        

        pass

    # Visualization tools
    def plot_routing_graph(self, ax: plt.Axes = None):
        if self.edges is None or self.nodes is None:
            print('Routing graph is not built yet!')
            return
        for edge in self.edges:
            start, end = edge
            ax.plot([start[0], end[0]], [start[1], end[1]], 'k--', alpha=0.65, color='red')


    def visualize(self, ax: plt.Axes = None, show: bool =  True, save: bool = False, outdir: str = 'out', tag: str = 'map.png'):
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

        # Plot routing graph
        self.plot_routing_graph(ax)

        if self.bounds is not None:
            min_x, min_y, max_x, max_y = self.bounds
            ax.set_xlim(min_x, max_x)
            ax.set_ylim(min_y, max_y)
        else:
            ax.autoscale()
        ax.set_aspect('equal')
        ax.legend()
        if show and not save: 
            plt.show()
        if save:
            fileout = os.path.join(outdir, tag)
            plt.savefig(fileout, dpi = 600)

if __name__ == '__main__':
    OUT_DIR = os.path.join('output', 'simple_shortest_path')
    os.makedirs(OUT_DIR, exist_ok=True)
    test_cases_num = 100
    test_seeds = np.random.randint(0, 100, size=test_cases_num)
    bounds = (-10, -10, 10, 10)
    map = Map(bounds=bounds)
    map.init_random_map(bounds=bounds, obstacle_size=(2, 4), n_obstacles=2, seed=19)
    map.build_routing_graph()
    map.visualize(save=True, outdir=OUT_DIR)

