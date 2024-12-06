import numpy as np
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.patches import Polygon as MplPolygon
from simple_shortest_path import Polygon, Map

def generate_polygon(vertex, sides, radius=1):
    import math
    if sides < 3:
        raise ValueError("A polygon must have at least 3 sides.")
    
    x0, y0 = vertex
    angle_step = 2 * math.pi / sides 
    center_x = x0 - radius  
    center_y = y0  

    vertices = []
    for i in range(sides):
        angle = i * angle_step
        x = center_x + radius * math.cos(angle)
        y = center_y + radius * math.sin(angle)
        vertices.append((x, y))
    
    return vertices


class MapWithBorders(Map):
    def __init__(self, bounds: tuple = None, goal: np.array = None, start: np.array = None):
        super().__init__(bounds=bounds)
        self.original_obstacles: list[Polygon] = [] #For visualization
        self.robot_shape = None
    
    def set_robot_shape(self, point: np.array =  None, shape: str = "triangle", size= 0.5) -> Polygon:
        if point is None:
            point = self.start
        x, y = point[0], point[1]
        if shape == "triangle":
            vertices = generate_polygon(point, 3, radius=size)
        elif shape == "pentagon":
            vertices = generate_polygon(point, 5, radius=size)
        elif shape == "hexagon":
            vertices = generate_polygon(point, 6, radius=size)
        else:
            raise ValueError("Unsupported shape: {}".format(shape))
        
        self.robot_shape = Polygon(np.array(vertices))
        return self.robot_shape

    def dialate_obstacles(self):
        self.original_obstacles = self.obstacles.copy()
        obstacles = []

        for obstacle in self.obstacles:
            dialated_obstacle = obstacle.dialiate_by_polygon(self.robot_shape)
            obstacles.append(dialated_obstacle)

        for obstacle in self.original_obstacles :
            obstacles.append(obstacle)

        self.obstacles = obstacles

    def build_routing_graph(self):
        return super().build_routing_graph()

    def visualize(self, ax: Axes = None, show: bool = True, save: bool = False, outdir: str = 'out', tag: str = 'map.png'):
        # # Swap obstacles for visualization
        dialated_obstacles = self.obstacles.copy()
        self.obstacles = self.original_obstacles.copy()

        if ax is None:
            fig, ax = plt.subplots()

        if self.robot_shape is not None and hasattr(self, 'path') and self.path is not None:
            for point in self.path:
               
                robot_patch = MplPolygon(self.robot_shape.vertices + point - self.start, 
                                         closed=True, 
                                         edgecolor='blue', 
                                         facecolor='blue', 
                                         alpha=0.5)
                ax.add_patch(robot_patch)

        if self.robot_shape is not None:
            robot_patch = MplPolygon(self.robot_shape.vertices, 
                                     closed=True, 
                                     edgecolor=self.robot_shape.get_color(),
                                     facecolor=self.robot_shape.get_color(),
                                    alpha=0.5)
            ax.add_patch(robot_patch)

        for obstacle in dialated_obstacles:
            polygon = MplPolygon(obstacle.vertices, closed=True, edgecolor=obstacle.get_color(), facecolor=obstacle.get_color(), alpha=0.2)
            ax.add_patch(polygon)
            
        super().visualize(ax, show, save, outdir, tag)

