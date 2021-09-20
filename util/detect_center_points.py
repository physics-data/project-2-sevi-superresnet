#!/usr/bin/python3
# -*- coding: utf-8 -*-

import numpy as np
import math
from tqdm import tqdm


def detect_center_points(FinalImage,start_,end_):

    number_list = np.arange(0, 1024)
    z_grid, x_grid = np.meshgrid(number_list, number_list)
    final_count = np.zeros((1024, 1024))
    for j in tqdm(list(range(len(FinalImage)))[start_:end_]):
        #得到目标的图像
        image = FinalImage[j][1]

        #生成判断一个点是否大于(等于)周围所有点的数组
        #对于多个极大值相邻的情况，先取x最大者，再在x相同的点中取z最大者作为目标点
        left_roll = np.roll(image, 1, axis=1)
        right_roll = np.roll(image, -1, axis=1)
        up_equal = (image >= np.roll(image, 1, axis=0))
        down = (image > np.roll(image, -1, axis=0))
        left_up_equal = (image >= np.roll(left_roll, 1, axis=0))
        left_down = (image > np.roll(left_roll, -1, axis=0))
        right_up = (image > np.roll(right_roll, 1, axis=0))
        right_down = (image > np.roll(right_roll, -1, axis=0))
        left_equal = (image >= np.roll(image, 1, axis=1))
        right_up_equal = (image >= np.roll(right_roll, 1, axis=0))

        left = (image > left_roll)
        right = (image > right_roll)

        #设置可判断为电子的最小亮度
        limit = (image > 70)

        center_points = np.logical_and.reduce([left_equal, right, up_equal, down,
                                               left_up_equal, left_down,
                                               right_up_equal, right_down, limit])

        left_or_right_up = np.logical_or(left, right_up)

        #此处的center_points即为初步判断的可能有电子的位置
        center_points = np.logical_and(center_points, left_or_right_up)

        #用于多个电子重叠时估计其理论平均距离的参数
        overlap_coefficient = 0.015*512*math.sqrt(2)

        #对于大于114且小于195的店，有可能为两个电子
        two_points = np.logical_and.reduce(
            [center_points, image > 114, image < 195])

        center_x_over115, center_z_over115 = np.where(two_points)
        #对所有这样的点，判断其是否为两个电子，并生成新的电子位置
        for i in range(len(center_x_over115)):
            x_index = center_x_over115[i]
            z_index = center_z_over115[i]
            value = image[x_index, z_index]

            #得到矩阵中所有点到目标点的距离
            r_matrix = np.sqrt((x_grid-x_index)**2 +
                               (z_grid-z_index)**2)

            judge_number = image*np.logical_and(r_matrix >= 4, r_matrix < 5)
            min_for_judge = judge_number[judge_number != 0].min()

            #若在离目标点距离大于等于4而小于5的范围内，最大值和最小值之差大于16，
            #或是最大亮度大于130，则判定为两个电子
            if np.max(judge_number) - min_for_judge > 16 or value > 130:

                #对于有一片极大值点的情况，将目标点移到这一片的中央
                max_area = (image[x_index-6:x_index+1,
                            z_index-4:z_index+5] == value)
                max_value_x, max_value_z = np.where(max_area)
                max_value_x += x_index - 6
                max_value_z += z_index - 4
                center_points[x_index][z_index] = False
                x_index = int((x_index+np.min(max_value_x))/2)
                z_index = int((np.min(max_value_z)+np.max(max_value_z))/2)

                r_matrix = np.sqrt((x_grid-x_index)**2 +
                                   (z_grid-z_index)**2)

                #估算两个各参数均为平均值的光斑叠加得到该光斑的亮度所需的距离
                estimated_r = overlap_coefficient * math.sqrt(math.log(200/value))

                #得到在该距离范围中的最大值方位，作为第一个电子的方位
                required_r = image*np.logical_and(r_matrix > estimated_r,
                                                  r_matrix < (estimated_r+1))
                max_for_r = np.max(required_r)
                max_x_list, max_z_list = np.where(required_r == max_for_r)
                max_x = max_x_list[0]
                max_z = max_z_list[0]

                #取与最大值方位相反的方位，作为第二个电子的方位
                opposite_x = 2*x_index - max_x
                opposite_z = 2*z_index - max_z

                #判断两个方位上的点的大小的比值，以此大致得到两个方位到极大值点的距离
                rate = image[max_x, max_z]/image[opposite_x, opposite_z]
                larger_x = int(x_index + (max_x-x_index)*rate**3)
                larger_z = int(z_index + (max_z-z_index)*rate**3)
                smaller_x = int(x_index - (max_x-x_index)/(rate**3))
                smaller_z = int(z_index - (max_z-z_index)/(rate**3))

                if larger_x >= 1024 or larger_z >= 1024 or larger_x < 0 or larger_z < 0:
                    continue

                #生成新的电子坐标
                center_points[larger_x][larger_z] = True
                center_points[smaller_x][smaller_z] = True

        #极大值大于194时，判断为三个电子
        three_points = np.logical_and(center_points, image > 194)
        center_x_over195, center_z_over195 = np.where(three_points)
        for i in range(len(center_x_over195)):
            x_index = center_x_over195[i]
            z_index = center_z_over195[i]
            value = image[x_index, z_index]

            max_area = (image[x_index-6:x_index+1,
                        z_index-4:z_index+5] == value)
            max_value_x, max_value_z = np.where(max_area)
            max_value_x += x_index - 6
            max_value_z += z_index - 4
            center_points[x_index][z_index] = False

            x_index = int((x_index+np.min(max_value_x))/2)
            z_index = int((np.min(max_value_z)+np.max(max_value_z))/2)

            #估算三个各参数均为平均值的光斑叠加得到该光斑的亮度所需的距离
            estimated_r = overlap_coefficient*math.sqrt(math.log(300/value))
            r_matrix = np.sqrt((x_grid-x_index)**2 +
                               (z_grid-z_index)**2)

            #找到该距离范围内亮度值最大的方位，设置新的电子坐标
            required_r = image*np.logical_and(r_matrix > estimated_r,
                                              r_matrix < (estimated_r+1))
            max_for_r = np.max(required_r)
            max_x_list, max_z_list = np.where(required_r == max_for_r)
            max_x = max_x_list[0]
            max_z = max_z_list[0]

            #在与最大方位成120度角的两个方向分别设置两个新的电子坐标
            theta = math.pi*2/3
            delta_x = max_x - x_index
            delta_z = max_z - z_index
            x_other1 = x_index + int(delta_x*math.cos(theta) -
                                     delta_z*math.sin(theta))
            z_other1 = z_index + int(delta_x*math.sin(theta) +
                                     delta_z*math.cos(theta))
            x_other2 = x_index + int(delta_x*math.cos(-theta) -
                                     delta_z*math.sin(-theta))
            z_other2 = z_index + int(delta_x*math.sin(-theta) +
                                     delta_z*math.cos(-theta))

            center_points[max_x][max_z] = True
            center_points[x_other1][z_other1] = True
            center_points[x_other2][z_other2] = True

        #累加电子数
        final_count += center_points.astype(np.int32)
    return final_count
