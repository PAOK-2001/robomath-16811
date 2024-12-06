import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from test_cases import COMPOUND_TESTS
from compound_shortest_path import MapWithBorders
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Compound Shortest Path Testing')
    parser.add_argument('--test_type', type=str, choices=['random', 'fixed'], default='fixed', help='Type of testing to perform')
    args = parser.parse_args()

    OUT_DIR = os.path.join('output', 'compound_shortest_path', args.test_type)
    os.makedirs(OUT_DIR, exist_ok=True)

    print(f'Running {args.test_type} tests...')

    if args.test_type == 'fixed':
        robot_shape = ["triangle", "triangle", "pentagon", "hexagon", "pentagon"]
        for i, test in enumerate(COMPOUND_TESTS):
            map = test
            map.set_robot_shape(shape= robot_shape[i % len(robot_shape)])
            map.dialate_obstacles()
            map.build_routing_graph()
            path = map.get_shortest_path()
            if path is None or len(path) == 0:
                fig, ax = plt.subplots()
                ax.text(0.5, 0.5, 'Failed to find a valid path.', horizontalalignment='center', verticalalignment='center', transform=ax.transAxes, fontsize=12, color='red')
                map.visualize(ax=ax, save=True, outdir=OUT_DIR, tag=f'test_case_{i}_failed.png')
            else:
                map.visualize(save=True, outdir=OUT_DIR, tag=f'test_case_{i}.png')
    elif args.test_type == 'random':
        np.random.seed(42)
        num_tests = 350
        test_seeds = np.random.choice(range(num_tests), size=num_tests, replace=False)

        for seed in test_seeds:
            side_length    = np.random.randint(5, 12)
            bounds = (-side_length, -side_length, side_length, side_length)
            map = MapWithBorders(bounds=bounds)
            map.init_random_map(bounds=bounds, obstacle_size=(1, 2), n_obstacles=np.random.randint(2,4), seed=seed, vertix_range=(3, 6))
            map.set_robot_shape(shape=np.random.choice(['triangle', 'pentagon', 'hexagon']))
            map.dialate_obstacles()
            map.build_routing_graph()
            path = map.get_shortest_path(id=seed)
            if path is None or len(path) == 0:
                fig, ax = plt.subplots()
                ax.text(0.5, 0.5, 'Failed to find a valid path.', horizontalalignment='center', verticalalignment='center', transform=ax.transAxes, fontsize=12, color='red')
                map.visualize(ax=ax, save=True, outdir=OUT_DIR, tag=f'map_{seed}_failed.png')

            else:
                map.visualize(save=True, outdir=OUT_DIR, tag=f'map_{seed}.png')
    
    else:
        raise ValueError('Invalid test type. Choose from [random, fixed]')