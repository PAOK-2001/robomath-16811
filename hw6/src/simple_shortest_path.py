import os
import shapely
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as MplPolygon
from convex_hull_2d import get_triplet_orientation, Orientation, convex_hull_2D

def on_segment(p1, p2, q):
    # Check if q lies on the line segment p1p2
    if (q[0] <= max(p1[0], p2[0]) and q[0] >= min(p1[0], p2[0]) and
        q[1] <= max(p1[1], p2[1]) and q[1] >= min(p1[1], p2[1])):
        return True

def do_lines_intersect(l1: np.array, l2: np.array) -> bool:

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

def dijkstra(edges: list, start: np.array, goal: np.array) -> np.array:
    import heapq
    graph = {}
    # Build the graph
    for edge in edges:
        s, e = edge
        s, e = tuple(s), tuple(e)
        if s not in graph:
            graph[s] = []
        if e not in graph:
            graph[e] = []
        graph[s].append(e)
        graph[e].append(s)
    # Initialize the distance array
    dist = {node: float('inf') for node in graph}
    dist[tuple(start)] = 0

    prev = {node: None for node in graph}
    pq = [(0, tuple(start))] 
    while pq:
        current_dist, node = heapq.heappop(pq)
        if np.array_equal(node, goal):
            break
        for neighbor in graph[node]:
            distance = current_dist + np.linalg.norm(np.array(node) - np.array(neighbor))
            if distance < dist[neighbor]:
                dist[neighbor] = distance
                prev[neighbor] = node
                heapq.heappush(pq, (distance, neighbor))
    
    path = []
    current_node = tuple(goal)
    while current_node is not None:
        path.append(np.array(current_node))
        current_node = prev[current_node]

    path.reverse()
    return np.array(path)
    
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

        for i in range(1, len(self.vertices) + 1):
            next_vertix = self.vertices[i % len(self.vertices)]
            # Check if  point is within bounds
            if y > min(vertix[1], next_vertix[1]) and y <= max(vertix[1], next_vertix[1]):
                if  x <= max(vertix[0], next_vertix[0]):
                    intersect_point = (y - vertix[1]) * (next_vertix[0] - vertix[0]) / (next_vertix[1] - vertix[1]) + vertix[0]
                    if vertix[0] == next_vertix[0] or x <= intersect_point:
                        if point not in self.vertices:
                            intersection_count += 1
            vertix = next_vertix

        return intersection_count % 2 == 1 # Odd number of intersections means point is inside polygon
    
    def line_crosses(self, p1: np.array, p2: np.array):
        for k in range(len(self.vertices)):
            v1 = self.vertices[k]
            v2 = self.vertices[(k + 1) % len(self.vertices)]
            if (np.array_equal(p1, v1) and np.array_equal(p2, v2)) or (np.array_equal(p1, v2) and np.array_equal(p2, v1)):
                return False
       
        sides = len(self.vertices)
        crosses = 0

        for i in range(sides):
            p3 = self.vertices[i]
            p4 = self.vertices[(i + 1) % sides]
            if do_lines_intersect([p1, p2], [p3, p4]):
                crosses += 1
        
        if not any(np.array_equal(p1, v) for v in self.vertices) and not any(np.array_equal(p2, v) for v in self.vertices):
            if crosses > 0:
                return True
            
        return crosses % 2 == 1

    def contains_line(self, p1: np.array, p2: np.array):
        perimeter = []
        for k in range(len(self.vertices)):
            v1 = self.vertices[k]
            v2 = self.vertices[(k + 1) % len(self.vertices)]
            perimeter.append([v1, v2])
        if p1 in self.vertices and p2 in self.vertices:
            if any(np.array_equal([p1, p2], edge) or np.array_equal([p2, p1], edge) for edge in perimeter):
                return False
            else:
                return True
        return False

    def reflect_about_point(self, polygon: 'Polygon' , point: np.array):
        x_r, y_r = point[0], point[1]
        # Reflect each vertex of the polygon
        reflected_vertices = []
        for vertex in polygon.vertices:
            x, y = vertex[0], vertex[1]
            x_reflected = 2 * x_r - x
            y_reflected = 2 * y_r - y
            reflected_vertices.append([x_reflected, y_reflected])
        return Polygon(np.array(reflected_vertices))

    def dialiate_by_polygon(self, polygon: 'Polygon', color: str = '#62D085'):
        reference_point = polygon.vertices[0]
        # Reflect the shape about the reference point
        reflected_poly = self.reflect_about_point(polygon, reference_point)
        # Express points as relative displacements from reference point
        relative_vertices = []
        for vertex in reflected_poly.vertices:
            relative_vertex = vertex - reference_point
            relative_vertices.append(relative_vertex)
        # Add the reflected polygon to each vertex of the original polygon
        dialated_vertices = []
        for vertex in self.vertices:
            for relative_vertex in relative_vertices:
                dialated_vertex = vertex + relative_vertex
                dialated_vertices.append(dialated_vertex)
        dialated_vertices = np.array(dialated_vertices)
        expanded_poly = convex_hull_2D(dialated_vertices)
        poly = Polygon(expanded_poly)
        poly.set_color(color)
        return poly

class Map:
    def __init__(self, bounds: tuple = None):
        self.obstacles: list[Polygon] = []
        self.bounds: tuple = bounds
        self.goal: np.array = None
        self.start: np.array = None
        self.nodes: list = None
        self.edges: list = None
        self.path = None

    def add_obstacle(self, polygon: Polygon):
        self.obstacles.append(polygon)
    
    def set_start(self, start: np.array):
        self.start = start

    def set_goal(self, goal: np.array):
        self.goal = goal
    
    def init_random_map(
        self, 
        bounds: tuple, 
        obstacle_size: tuple,
        n_obstacles: int, 
        seed: int = 42,
        vertix_range: tuple = (3, 10),
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

        if self.start is None:
            self.start = np.random.uniform([min_x, min_y], [max_x, max_y])
        if place_goal_in_obstacle:
            obstacle = np.random.choice(self.obstacles)
            min_x, min_y = np.min(obstacle.vertices, axis=0)
            max_x, max_y = np.max(obstacle.vertices, axis=0)
            self.goal = np.random.uniform([min_x, min_y], [max_x, max_y])
        else:
            if self.goal is None:
                self.goal = np.random.uniform([min_x, min_y], [max_x, max_y])
               
    def is_valid_edge(self, edge: tuple):
        start, end = edge
        for obstacle in self.obstacles:
            if obstacle.contains_point(start) or obstacle.contains_point(end):
                return False
         
            if obstacle.line_crosses(start, end) or obstacle.contains_line(start, end):
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
                edges.append(edge)

        #Insert all polygon edges
        for obstacle in self.obstacles:
            for i in range(len(obstacle.vertices)):
                start = obstacle.vertices[i]
                end = obstacle.vertices[(i + 1) % len(obstacle.vertices)]
                edge = (start, end)
                edges.append(edge)

        for edge in edges:
            if not self.is_valid_edge(edge):
                edges = [e for e in edges if not (np.array_equal(e[0], edge[0]) and np.array_equal(e[1], edge[1]))]

        self.nodes = nodes
        self.edges = edges

    def get_shortest_path(self, id = 0):
        # Check if we are given valid start and goal
        for obstacle in self.obstacles:
            if obstacle.contains_point(self.start) or obstacle.contains_point(self.goal):
                print('Failed to find a valid path.')
                return None

        try:
            path = dijkstra(self.edges, self.start, self.goal)
        except:
            print('Failed to find a valid path.')
            path = None
        self.path = path
        return path

    # Visualization tools
    def plot_routing_graph(self, ax: plt.Axes = None, color: str = 'gray'):
        if self.edges is None or self.nodes is None:
            print('Routing graph is not built yet!')
            return
        for edge in self.edges:
            start, end = edge
            ax.plot([start[0], end[0]], [start[1], end[1]], 'k--', alpha=0.65, color='#B478CE', zorder=0)


    def visualize(self, ax: plt.Axes = None, show: bool =  True, save: bool = False, outdir: str = 'out', tag: str = 'map.png'):
        if ax is None:
            fig, ax = plt.subplots()
        # Plot obstacles
        for obstacle in self.obstacles:
            polygon = MplPolygon(obstacle.vertices, closed=True, edgecolor=obstacle.get_color(), facecolor=obstacle.get_color(), alpha=0.5)
            ax.add_patch(polygon)
            
        for obstacle in self.obstacles:
            ax.scatter(obstacle.vertices[:, 0], obstacle.vertices[:, 1], c='#312F2F')
        # Plot start and goal

        if self.start is not None and self.goal is not None:
            ax.plot(self.start[0], self.start[1], 'bo', markersize=10, color='#DCEC83', label='Start')
            ax.plot(self.goal[0], self.goal[1], 'go', markersize=10, color='#B9DDB3', label='Goal')

        if self.path is not None:
            path = np.array(self.path)
            ax.plot(path[:, 0], path[:, 1], 'r-', linewidth=3.5, label='Shortest Path', color='green', zorder=10)

        # Plot routing graph
        self.plot_routing_graph(ax)

        ax.autoscale()
        ax.set_aspect('equal')
        ax.legend()
        if show and not save: 
            plt.show()
        if save:
            fileout = os.path.join(outdir, tag)
            plt.savefig(fileout, dpi = 600)
            plt.close()

