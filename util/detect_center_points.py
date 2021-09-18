#!/usr/bin/python3
# -*- coding: utf-8 -*-

from typing_extensions import Final
import numpy as np
import h5py
import matplotlib.pyplot as plt
import math
from tqdm import tqdm


def detect_center_points(FinalImage):

    number_list = np.arange(0, 1024)
    z_grid, x_grid = np.meshgrid(number_list, number_list)
    final_count = np.zeros((1024, 1024))
    for j in tqdm(list(range(len(FinalImage)))[10000:15000]):
        image = FinalImage[j][1]

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

        limit = (image > 70)

        center_points = np.logical_and.reduce([left_equal, right, up_equal, down,
                                               left_up_equal, left_down,
                                               right_up_equal, right_down, limit])

        left_or_right_up = np.logical_or(left, right_up)

        center_points = np.logical_and(center_points, left_or_right_up)

        overlap_coefficient = 0.015*512*math.sqrt(2)
        two_points = np.logical_and.reduce(
            [center_points, image > 114, image < 195])

        center_x_over115, center_z_over115 = np.where(two_points)
        for i in range(len(center_x_over115)):
            x_index = center_x_over115[i]
            z_index = center_z_over115[i]
            value = image[x_index, z_index]

            r_matrix = np.sqrt((x_grid-x_index)**2 +
                               (z_grid-z_index)**2)

            judge_number = image*np.logical_and(r_matrix >= 4, r_matrix < 5)
            min_for_judge = judge_number[judge_number != 0].min()

            if np.max(judge_number) - min_for_judge > 16 or value > 130:
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

                estimated_r = overlap_coefficient * \
                    math.sqrt(math.log(200/value))
                required_r = image*np.logical_and(r_matrix > estimated_r,
                                                  r_matrix < (estimated_r+1))
                max_for_r = np.max(required_r)
                max_x_list, max_z_list = np.where(required_r == max_for_r)
                max_x = max_x_list[0]
                max_z = max_z_list[0]
                opposite_x = 2*x_index - max_x
                opposite_z = 2*z_index - max_z
                rate = image[max_x, max_z]/image[opposite_x, opposite_z]

                larger_x = int(x_index + (max_x-x_index)*rate**3)
                larger_z = int(z_index + (max_z-z_index)*rate**3)
                smaller_x = int(x_index - (max_x-x_index)/(rate**3))
                smaller_z = int(z_index - (max_z-z_index)/(rate**3))

                if larger_x >= 1024 or larger_z >= 1024:
                    continue
                center_points[larger_x][larger_z] = True
                center_points[smaller_x][smaller_z] = True

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
            estimated_r = overlap_coefficient*math.sqrt(math.log(300/value))
            r_matrix = np.sqrt((x_grid-x_index)**2 +
                               (z_grid-z_index)**2)
            required_r = image*np.logical_and(r_matrix > estimated_r,
                                              r_matrix < (estimated_r+1))
            max_for_r = np.max(required_r)
            max_x_list, max_z_list = np.where(required_r == max_for_r)
            max_x = max_x_list[0]
            max_z = max_z_list[0]
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

        final_count += center_points.astype(np.int32)
    return final_count
