import threading
import h5py
import matplotlib.pyplot as plt
from torch.utils import data
from tqdm import tqdm
import numpy as np
from tqdm import tqdm
import os
import sys

index_, data_len, data_type = [int(i) for i in sys.argv[1:]]


dir_path = os.path.dirname(os.path.realpath(__file__))
parent_dir_path = os.path.abspath(os.path.join(dir_path, os.pardir))
sys.path.insert(0, parent_dir_path)

from util import gen_my_pic_double

output_path_train = 'data/traindata/'
output_path_valid = 'data/validdata/'
if data_type == 0:
    output_path = output_path_train
elif data_type == 1:
    output_path = output_path_valid
else:
    print("data_type should be 0 or 1.")
    exit(1)
    
if not os.path.exists(output_path):
    os.mkdir(output_path)

output_name = 'dataset_my_double{}.h5'


# 分布参数
limit = [
    [0.35, 1],
    [0.43, 1.434],
    [0.25, 1.24],
    [0.118, 0.9],
    [0.045, 0.6],
    [0.012, 0.36],
    [0.0026, 0.2],
    [0.0004, 0.1],
    [7.0e-5, 0.045],
    [5.6e-6, 0.019],
    [1.2e-7, 0.0067],
    [2e-9, 0.0023]
]
mu = [1, 0.85, 0.55, 0.25, 0.1, 0.04, 0.015, 0.002, 0.004, 0, 0, 0]
sigma = [0.65, 0.5, 0.5, 0.5, 0.5, 0.3, 0.2, 0.1, 0.05, 0.02, 0.01, 0.01]


def gen_random(mean, scale, lim):
    # 正态分布+截断
    while True:
        x = np.random.normal(mean, scale)
        if lim[0] < x < lim[1]:
            break
    return x


def gen_my_beta_dist(index, total_len=200):

    r_list = []
    label_list = []
    pic_list = []
    for j in tqdm(range(total_len)):
        beta1 = [0.5]
        for i in range(12):
            beta1.append(gen_random(mu[i], sigma[i], limit[i]))
        beta2 = [0.5]
        for i in range(12):
            beta2.append(gen_random(mu[i], sigma[i], limit[i]))
        r1 = np.random.uniform(0.3, 0.9)
        r2 = np.random.uniform(0.3, 0.9)

        r = np.array([min(r1, r2), max(r1, r2)], dtype=np.float64)
        label = np.array([beta1[1:], beta2[1:]], dtype=np.float64)


        pic1 = gen_my_pic_double.gen_pic(
            beta1, r[0], np.random.uniform(0.005, 0.009))
        pic2 = gen_my_pic_double.gen_pic(
            beta2, r[1], np.random.uniform(0.005, 0.009))
        pic = (pic1+pic2)
        pic = pic/pic.sum()*255

        r_list.append(r)
        label_list.append(label)
        pic_list.append(pic)
    with h5py.File((output_path+output_name).format(index), 'w') as out:
        out['inputs'] = pic_list
        out['Rs'] = r_list
        out['labels'] = label_list

print(f'开始生成: {output_name.format(index_)}')
gen_my_beta_dist(index_, data_len)
print(f'结束生成: {output_name.format(index_)}')