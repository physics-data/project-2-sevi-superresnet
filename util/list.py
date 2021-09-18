#!/usr/bin/python3
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import h5py


n=2
#矩阵边长为2n+1
filename = "train0.h5"
with h5py.File(filename,"r") as input:
    image_all = []
    if_all = []
    for i in range(100):
        target = input["DetectedElectrons"]["ImageId"] == i
        x_real = input["DetectedElectrons"][target]["x"]
        z_real = input["DetectedElectrons"][target]["z"]
        image = input["FinalImage"][i][1]

        x = ((x_real+1)*1024/2).astype(np.int32)
        z = ((z_real+1)*1024/2).astype(np.int32)
        x_z = np.stack((x, z), axis=-1)
        x_z_random = np.random.randint(n, 1024-n, size=(4*len(x), 2))
        x_z_near_random = np.random.randint(-2, 3, size=(4*len(x), 2))
        x_z_near = np.tile(x_z, (4, 1)) + x_z_near_random
        x_z_all = np.concatenate((x_z, x_z_random, x_z_near))

        up = x_z_all[:, 0] - n
        down = x_z_all[:, 0] + n + 1
        left = x_z_all[:, 1] - n
        right = x_z_all[:, 1] + n + 1

        image_list = np.zeros((9*len(x), 2*n+1, 2*n+1))
        if_list = np.zeros(9*len(x))
        for j in range(len(x_z_all)):
            image_list[j] = image[up[j]:down[j], left[j]:right[j]]
            if x_z_all[j][0] in x and x_z_all[j][1] in z:
                if_list[j] = 1

        image_all.append(image_list)
        if_all.append(if_list)
    image_all = np.concatenate(image_all)
    if_all = np.concatenate(if_all)

        

            

        