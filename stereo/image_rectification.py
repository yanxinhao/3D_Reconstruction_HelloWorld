# coding=utf-8
'''
@Author: yanxinhao
@Email: 1914607611xh@i.shu.edu.cn
@LastEditTime : 2020-01-04 15:47:56
@LastEditors  : yanxinhao
@Description: 
'''
import cv2
import numpy as np

class ImageRectification(object):
    def __init__(self,cam_K,R,T,essential_matrix=None,fundamental_matrix=None):
        self.cam_k=cam_K
        self.a2b_R=R
        self.a2b_T=T
        self.essential_matrix=essential_matrix
        self.fundamental_matrix=fundamental_matrix

    
    def rectify(self,image_a,points_a,image_b,points_b):
        _,H1,H2=cv2.stereoRectifyUncalibrated(points_a,points_b,self.fundamental_matrix,image_a.shape[:2])
        rectified_image_a=cv2.warpPerspective(image_a,H1,image_a.shape[:2])
        rectified_image_b=cv2.warpPerspective(image_b,H2,image_b.shape[:2])
        return rectified_image_a,rectified_image_b

    def _calculate_R_rec(self):
        self.R_rec_a=None
        self.R_rec_b=None
        pass

    def __call__(self,image_a,image_b):
        pass