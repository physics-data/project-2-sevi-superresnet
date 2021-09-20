#!/usr/bin/python3
# -*- coding: utf-8 -*-

import numpy as np
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

    #随机生成总电子数
    total_num = int(np.random.uniform(1800000, 2500000))

    #目标勒让德多项式，即概率密度函数
    Legendre_poly = np.insert(b_ij, np.arange(
        len(b_ij)-1)+1, np.zeros(len(b_ij)-1))

    #对概率密度函数积分，得到累积分布函数
    integrated = np.polynomial.legendre.legint(Legendre_poly, lbnd=-1)

    def legendre_(x):
        return np.polynomial.legendre.legval(x, integrated)
    
    #通过对自变量取微元来将函数值映射回自变量的方法近似得到累积分布函数的反函数
    sample = np.linspace(-1, 1, num=1000000)
    value = legendre_(sample)

    #通过ITM方法将U(0,1)转化为目标勒让德多项式分布
    U_01 = np.random.random(total_num)
    cos_theta = np.interp(U_01, value, sample)

    #生成r和phi，并得到电子的位置
    r = np.random.normal(R_in, sigma_R, total_num)
    phi = np.random.random(total_num)*2*pi

    x = r*np.cos(phi)*np.sqrt(1-cos_theta**2)
    z = r*cos_theta

    x_pixel = ((x+1)*pixel/2).astype(np.int32)
    z_pixel = ((z+1)*pixel/2).astype(np.int32)

    #得到最终的图像
    pic = np.zeros((pixel, pixel))
    np.add.at(pic,(x_pixel, z_pixel), 1)

    return pic
