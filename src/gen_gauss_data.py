#!/usr/bin/python3
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import h5py
from tqdm import tqdm
import sys
import os
filename, pic_len, n_ = sys.argv[1:]
pic_len = int(pic_len)
n_ = int(n_)


# n=2
# 随机负样本占正样本比例
# random_num = 1
# near_random_num = 1
# 矩阵边长为2n+1
# filename = "train0.h5"


def gen_gauss_traindata(filename, pic_len, n=2, random_num=1, near_random_num=1):
    with h5py.File(filename, "r") as input:
        print(input['Truth'][:])
        image_all = []
        if_all = []
        image_list = []
        for i in tqdm(range(pic_len)):
            # 图像以及对应的点
            target = input["DetectedElectrons"]["ImageId"] == i
            x_real = input["DetectedElectrons"][target]["x"]
            z_real = input["DetectedElectrons"][target]["z"]
            image = input["FinalImage"][i][1]
            # 图像坐标
            x = ((x_real+1)*1024/2).astype(np.int32)
            z = ((z_real+1)*1024/2).astype(np.int32)
            x_z = np.stack((x, z), axis=-1)

            x_z_random = np.random.randint(
                n+3, 1024-n-3, size=(random_num*len(x), 2))  # 图像中的随机点
            # 真值点附近的5*5矩阵
            x_z_near_random = np.random.randint(-2,
                                                3, size=(near_random_num*len(x), 2))
            x_z_near = np.tile(x_z, (near_random_num, 1)) + x_z_near_random
            x_z_all = np.concatenate((x_z, x_z_random, x_z_near))  # 训练集的坐标
            
            up = x_z_all[:, 0] - n
            down = x_z_all[:, 0] + n + 1
            left = x_z_all[:, 1] - n
            right = x_z_all[:, 1] + n + 1

            # image_list = np.zeros(
            #     (len(x_z_all), 2*n+1, 2*n+1))
            # if_list = np.zeros(len(x_z_all), dtype=np.int32)
            for j in range(len(x_z_all)):
                if image[x_z_all[j][0]][x_z_all[j][1]] >70:
                    image_all.append(image[up[j]:down[j], left[j]:right[j]])
                    if x_z_all[j][0] in x and x_z_all[j][1] in z:
                        if_all.append(1)
                    else: 
                        if_all.append(0)

            # image_all.append(image_list)
            # if_all.append(if_list)
        image_all = np.array(image_all)
        if_all = np.array(if_all,dtype=np.int32)
        print(len(image_all),len(if_all))
    return image_all, if_all


if __name__ == '__main__':
    print('生成高斯数据')
    image_all, if_all = gen_gauss_traindata(
        filename, pic_len, n=n_, random_num=1, near_random_num=1)
    if not os.path.exists('./data/gauss_traindata'):
        os.makedirs('./data/gauss_traindata')
    if not os.path.exists('./data/gauss_validdata'):
        os.makedirs('./data/gauss_validdata')
    with h5py.File('data/gauss_validdata/gauss_validdata.h5', 'w') as opt:
        opt['inputs'] = image_all[:len(image_all)//10]
        opt['labels'] = if_all[:len(image_all)//10]
    with h5py.File('data/gauss_traindata/gauss_traindata.h5', 'w') as opt:
        opt['inputs'] = image_all[len(image_all)//10:]
        opt['labels'] = if_all[len(image_all)//10:]
    print('finish')
