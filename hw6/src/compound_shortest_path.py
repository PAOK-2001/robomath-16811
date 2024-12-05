import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.patches import Polygon as MplPolygon
from simple_shortest_path import Polygon, Map

class MapWithBorders(Map):
    def __init__(self, bounds: tuple = None, goal: np.array = None, start: np.array = None):
        super().__init__(bounds=bounds, goal=goal, start=start)
        self.original_obstacles: list[Polygon] = [] #For visualization
        self.robot_shape = None
    
    def set_robot_shape(self, point: np.array =  None, n_vertices: int = 3, size: int = 3) -> Polygon:
        if point is None:
            point = self.start
        x, y = point[0], point[1]
        angles = np.sort(np.random.uniform(0, 2 * np.pi, n_vertices - 1))
        vertices = np.array([
            [x + size * np.cos(angle), y + size * np.sin(angle)]
            for angle in angles
        ])
        vertices = np.vstack((point, vertices))
        self.robot_shape = Polygon(vertices)

    def dialate_obstacles(self):
        self.original_obstacles = self.obstacles.copy()
        obstacles = []
        for obstacle in self.obstacles:
            dialated_obstacle = obstacle.dialiate_by_polygon(self.robot_shape)
            obstacles.append(dialated_obstacle)
        self.obstacles = obstacles

    def build_routing_graph(self):
        return super().build_routing_graph()

    def visualize(self, ax: Axes = None, show: bool = True, save: bool = False, outdir: str = 'out', tag: str = 'map.png'):
        # # Swap obstacles for visualization
        # dialated_obstacles = self.obstacles.copy()
        # self.obstacles = self.original_obstacles.copy()

        if ax is None:
            fig, ax = plt.subplots()
        if self.robot_shape is not None:
            robot_patch = MplPolygon(self.robot_shape.vertices, closed=True, edgecolor=self.robot_shape.get_color(), facecolor=self.robot_shape.get_color(), alpha=0.5)
            ax.add_patch(robot_patch)
        for obstacle in self.obstacles:
            ax.scatter(obstacle.vertices[:, 0], obstacle.vertices[:, 1], c='b')

            # reflect_robot = self.robot_shape.reflect_about_point(self.robot_shape, self.robot_shape.vertices[0])
            # reflect_robot_patch = MplPolygon(reflect_robot.vertices, closed=True, edgecolor=reflect_robot.get_color(), facecolor=reflect_robot.get_color(), alpha=0.5)
            # ax.add_patch(reflect_robot_patch)
        # if dialated_obstacles is not None:
        #     for obstacle in dialated_obstacles:
        #         obstacle_patch = MplPolygon(obstacle.vertices, closed=True, edgecolor="red", facecolor=obstacle.get_color(), alpha=0.5)
        #         ax.add_patch(obstacle_patch)

        super().visualize(ax, show, save, outdir, tag)

        # # Swap obstacles back
        # self.original_obstacles = self.obstacles.copy()
        # self.obstacles = dialated_obstacles.copy()
        

if __name__ == '__main__':
    OUT_DIR = os.path.join('output', 'compound_shortest_path')
    os.makedirs(OUT_DIR, exist_ok=True)

    side_length  = 4
    bounds = (-side_length, -side_length, side_length, side_length)
    map = MapWithBorders(bounds=bounds)
    map.init_random_map(bounds=bounds, obstacle_size=(1, 3), n_obstacles=2, vertix_range=(4, 10))
    map.set_robot_shape(n_vertices=3, size=1)
    map.dialate_obstacles()
    map.build_routing_graph()
    # path = map.get_shortest_path()
    map.visualize(save=True, outdir=OUT_DIR, tag='map.png')