#!/usr/bin/python3
# -*- coding: utf-8 -*-

import numpy as np
import h5py
from sys import argv
from math import pi

import matplotlib.pyplot as plt


def gen_pic(b_ij, R_in, sigma_R):
    '''
    根据参数生成双球壳图片
    b_ij: beta分布, 包括第一项0.5
    R_in: 半径
    sigma_R: 半径标准差

    '''
    pixel = 1024
    total_num = int(np.random.uniform(1800000, 2500000))
    Legendre_poly = np.insert(b_ij, np.arange(
        len(b_ij)-1)+1, np.zeros(len(b_ij)-1))
    c = np.polynomial.legendre.legint(Legendre_poly, lbnd=-1)

    def legendre_(x):
        return np.polynomial.legendre.legval(x, c)
    sample = np.linspace(-1, 1, num=1000000)
    b = legendre_(sample)
    r1 = np.random.random(total_num)
    cos_theta = np.interp(r1, b, sample)

    r = np.random.normal(R_in, sigma_R, total_num)
    phi = np.random.random(total_num)*2*pi

    x = r*np.cos(phi)*np.sqrt(1-cos_theta**2)
    z = r*cos_theta

    x_pixel = ((x+1)*pixel/2).astype(np.int32)  # 得到坐标分量值在像素图中对应的像素的坐标
    z_pixel = ((z+1)*pixel/2).astype(np.int32)

    pic = np.zeros((pixel, pixel))
    for i in range(total_num):
        pic[x_pixel[i]][z_pixel[i]] += 1

    return pic
