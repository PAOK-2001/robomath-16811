import os
import numpy as np

DATA_DIR = "/home/pablo/Documents/coursework/robotmath/robomath-16811/hw2/data/paths.txt"

def load_data() -> np.array:
    with open(DATA_DIR) as f:
        paths_raw = f.read()

    # Given that data is trajectories. The first line contains the x coordinates, and the second contains the y coordinates for one path. Parse into numpy
    paths = paths_raw.split("\n")
    paths = [path.split(" ") for path in paths]
    paths = np.array(paths, dtype= np.float64)
    return paths


if __name__ == "__main__":
    paths = load_data()
    import cv2
    cv2.projectPoints()