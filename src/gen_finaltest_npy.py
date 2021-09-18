import os
import sys

import h5py
import matplotlib.pyplot as plt
import numpy as np
from util import detect_center_points

dir_path = os.path.dirname(os.path.realpath(__file__))
parent_dir_path = os.path.abspath(os.path.join(dir_path, os.pardir))
sys.path.insert(0, parent_dir_path)

filename = "data/finaltest.h5"

input = h5py.File(filename, "r")

FinalImage = input["FinalImage"]


aa = detect_center_points.detect_center_points(FinalImage)

np.save("finaltest_npy/final_10000_15000_retry.npy", aa)
# b = np.load("final.npy")
