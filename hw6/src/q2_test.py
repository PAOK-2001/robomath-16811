import os
import numpy as np
import argparse
import matplotlib.pyplot as plt
from simple_shortest_path import Map
from test_cases import SIMPLE_TESTS

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Simple Shortest Path Testing')
    parser.add_argument('--test_type', type=str, choices=['random', 'fixed'], default='fixed', help='Type of testing to perform')
    args = parser.parse_args()

    OUT_DIR = os.path.join('output', 'simple_shortest_path', args.test_type)
    os.makedirs(OUT_DIR, exist_ok=True)

    if args.test_type == 'fixed':
        for i, test in enumerate(SIMPLE_TESTS):
            map = test
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
            side_length    = np.random.randint(3, 10)
            bounds = (-side_length, -side_length, side_length, side_length)
            map = Map(bounds=bounds)
            map.init_random_map(bounds=bounds, obstacle_size=(1, 3), n_obstacles=np.random.randint(2,7), seed=seed, vertix_range=(4, 10))
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