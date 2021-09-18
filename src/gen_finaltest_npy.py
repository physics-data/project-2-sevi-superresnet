import threading
import numpy as np
import matplotlib.pyplot as plt
import h5py
import os
import sys
dir_path = os.path.dirname(os.path.realpath(__file__))
parent_dir_path = os.path.abspath(os.path.join(dir_path, os.pardir))
sys.path.insert(0, parent_dir_path)

from util import detect_center_points

start_, end_ = sys.argv[1:]
start_ = int(start_)
end_ = int(end_)


finaltest_filename = "data/finaltest.h5"

output_path = "finaltest_npy/"
output_file_name = "final_{}_{}_retry.npy"

if not os.path.exists(output_path):
    os.mkdir(output_path)

# 读取文件并识别电子位置
with h5py.File(finaltest_filename, "r") as input:
    thread_num = 8
    FinalImage = input["FinalImage"]
    print("开始生成数据：" + (output_path+output_file_name).format(start_, end_))
    output = detect_center_points.detect_center_points(
        FinalImage, start_, end_)
    np.save((output_path+output_file_name).format(start_, end_), output)
    print("生成完毕：" + (output_path+output_file_name).format(start_, end_))

