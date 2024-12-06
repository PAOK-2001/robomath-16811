import numpy as np
from compound_shortest_path import MapWithBorders
from simple_shortest_path import Polygon, Map

def get_test_case_1(type='simple'):
    side_length = 14
    bounds = (-side_length, -side_length, side_length, side_length)
    if type == 'compound':
        map = MapWithBorders(bounds=bounds)
    else:
        map = Map(bounds=bounds)
    map.set_start((np.array([1.5, -1])))
    map.set_goal((np.array([12.5, 8.5])))
    # Add obstacles
    obstacle1 = Polygon(np.array([
        [5, 2], [2, 2],
        [2, 4], [4, 4]
    ]))

    obstacle2 = Polygon(np.array([
        [9, 4],[5, 4],
        [5, 7], [9, 7]
    ]))

    obstacle3 = Polygon(np.array([
        [12, 1], [8, 1], 
        [8, 3], [10, 3],
        [10, 6], [12, 6]
    ]))

    obstacles = [obstacle1, obstacle2, obstacle3]
    for obstacle in obstacles:
        obstacle.set_color('black')
        map.add_obstacle(obstacle)
    return map

def get_test_case_2(type='simple'):
    seed = 30
    side_length  = 5
    bounds = (-side_length, -side_length, side_length, side_length)
    np.random.seed(seed)
    if type == 'compound':
        map = MapWithBorders(bounds=bounds)
    else:
        map = Map(bounds=bounds)
    map.init_random_map(bounds=bounds, obstacle_size=(1, 3), n_obstacles=4, seed=seed, vertix_range=(4, 10))
    return map
    
def get_test_case_3(type='simple'):
    seed = 11
    side_length  = 10
    bounds = (-side_length, -side_length, side_length, side_length)
    np.random.seed(seed)
    if type == 'compound':
        map = MapWithBorders(bounds=bounds)
    else:
        map = Map(bounds=bounds)
    map.init_random_map(bounds=bounds, obstacle_size=(1, 3), n_obstacles=6, seed=seed, vertix_range=(4, 10))
    return map

def get_test_case_4(type='simple'):
    seed = 18
    side_length  = 10
    bounds = (-side_length, -side_length, side_length, side_length)
    np.random.seed(seed)
    if type == 'compound':
        map = MapWithBorders(bounds=bounds)
    else:
        map = Map(bounds=bounds)
    map.init_random_map(bounds=bounds, obstacle_size=(1, 3), n_obstacles=6, seed=seed, vertix_range=(4, 10), place_goal_in_obstacle=True)
    return map
    
def get_test_case_5(type='simple'):
    seed = 233
    side_length  = 5
    bounds = (-side_length, -side_length, side_length, side_length)
    np.random.seed(seed)
    if type == 'compound':
        map = MapWithBorders(bounds=bounds)
    else:
        map = Map(bounds=bounds)
    map.set_start((np.array([-4, -2])))
    map.set_goal((np.array([-7 , 5])))
    map.init_random_map(bounds=bounds, obstacle_size=(1, 3), n_obstacles=7, seed=seed, vertix_range=(4, 10))
    return map
    
SIMPLE_TESTS = [
    get_test_case_1(),
    get_test_case_2(),
    get_test_case_3(),
    get_test_case_4(),
    get_test_case_5(),
]

COMPOUND_TESTS = [
    get_test_case_1('compound'),
    get_test_case_2('compound'),
    get_test_case_3('compound'),
    get_test_case_4('compound'),
    get_test_case_5('compound'),
]
    