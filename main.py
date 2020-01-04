# coding=utf-8
'''
@Author: yanxinhao
@Email: 1914607611xh@i.shu.edu.cn
@LastEditTime : 2020-01-04 15:55:35
@LastEditors  : yanxinhao
@Description: 
'''
import cv2
import numpy as np
import os
from pathlib import Path
from utils.utils import *
from calibration.epipolar_geometry import EpipolarGeometry

image_paths = get_image_paths()
camera_matrix = np.array([[5.40756340e+03, 0.00000000e+00, 1.36694579e+03],
       [0.00000000e+00, 5.33216612e+03, 7.99420693e+02],
       [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]])
calu=EpipolarGeometry(camera_matrix,camera_matrix)
