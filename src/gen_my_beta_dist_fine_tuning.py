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
limit1 = [
    [0.05, 0.3],
    [0.4, 0.8],
    [0.3, 0.8],
    [0.2, 0.65],
    [0.05, 0.5],
    [0.012, 0.36],
    [0.0026, 0.2],
    [0.0004, 0.1],
    [7.0e-5, 0.045],
    [5.6e-6, 0.019],
    [1.2e-7, 0.0067],
    [2e-9, 0.0023]
]
mu1 = [0.1629,  0.6354,  0.5569,  0.4308,  0.3123,  0.1780,
       0.1002,  0.0391, 0.0115,  0.0027, -0.0014,  0.0025]
sigma1 = [0.65, 0.5, 0.5, 0.5, 0.5, 0.3, 0.2, 0.1, 0.05, 0.02, 0.01, 0.01]


limit2 = [
    [0.45, 0.65],
    [0.8, 1.2],
    [0.6, 1.1],
    [0.5, 0.9],
    [0.1, 0.7],
    [0.012, 0.4],
    [0.0026, 0.2],
    [0.0004, 0.1],
    [7.0e-5, 0.045],
    [5.6e-6, 0.019],
    [1.2e-7, 0.0067],
    [2e-9, 0.0023]
]
mu2 = [0.5651,  1.0044,  0.9117,  0.7059, 0.4335,  0.2702,
       0.1128,  0.0683,  0.0353,  0.0208,  0.0024,  0.0032]
sigma2 = [0.65, 0.5, 0.5, 0.5, 0.5, 0.3, 0.2, 0.1, 0.05, 0.02, 0.01, 0.01]


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
            beta1.append(gen_random(mu1[i], sigma1[i], limit1[i]))
        beta2 = [0.5]
        for i in range(12):
            beta2.append(gen_random(mu2[i], sigma2[i], limit2[i]))
        r1 = np.random.uniform(0.47, 0.53)
        r2 = np.random.uniform(0.77, 0.83)

        r = np.array([r1,r2], dtype=np.float64)
        label = np.array([beta1[1:], beta2[1:]], dtype=np.float64)


        pic1 = gen_my_pic_double.gen_pic(
            beta1, r[0], np.random.uniform(0.008, 0.013))
        pic2 = gen_my_pic_double.gen_pic(
            beta2, r[1], np.random.uniform(0.008, 0.013))
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