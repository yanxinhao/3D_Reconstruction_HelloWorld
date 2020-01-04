# coding=utf-8
'''
@Author: yanxinhao
@Email: 1914607611xh@i.shu.edu.cn
@LastEditTime : 2020-01-04 14:03:07
@LastEditors  : yanxinhao
@Description:  A class to define epipolar geometry constraint and get tranform matrix from essential matrix
'''
import cv2
import numpy as np
from scipy.linalg import null_space

class EpipolarGeometry(object):

    def __init__(self,cam_a_K,cam_b_K):
        self.cam_a_K=cam_a_K.copy()
        self.cam_b_K=cam_b_K.copy()
        self.cam_a_K_inv = np.linalg.inv(cam_a_K)
        self.cam_b_K_inv = np.linalg.inv(cam_b_K)

    def calculate(self,points_a,points_b):
        essential_matrix=self._get_essential_matrix(points_a,points_b)

        p1,p2=self._get_normalized_3d_coordinates(points_a,points_b)
        p1_2d,p2_2d=p1[:,:2],p2[:,:2]
        E,mask = cv2.findEssentialMat(p1_2d,p2_2d,self.cam_a_K)

        # print(E)
        # print(essential_matrix)

        # decompose R,T from essential matrix
        out=cv2.recoverPose(np.mat(E),p1_2d,p2_2d)
        _,R,T,mask=out
        print(R,T)
        print('------------------------------------------')
        out=cv2.recoverPose(np.mat(essential_matrix),p1_2d,p2_2d)
        _,R,T,mask=out
        print(R,T)



    def __call__(self,points_a,points_b):
        return self.calculate(points_a,points_b)
    
    def _get_normalized_3d_coordinates(self,points_a,points_b):
        """convert the 2d uv points to normalized 3d points(z=1)
        
        Args:
            points_a (np.array): the shape is [8,2]
            points_b (np.array): the shape is [8,2]
        
        Returns:
            points_3d_a,points_3d_b : the type is np.mat,the shape is[8,3]
        """
        padding=np.ones((len(points_a),1))
        norm_3d_a=np.hstack((points_a,padding))
        norm_3d_b=np.hstack((points_b,padding))
        points_3d_a=np.mat(self.cam_a_K_inv)*np.mat(norm_3d_a.T)
        points_3d_b=np.mat(self.cam_b_K_inv)*np.mat(norm_3d_b.T)

        return points_3d_a.T,points_3d_b.T

    def _get_essential_matrix(self,points_a,points_b):
        p1,p2=self._get_normalized_3d_coordinates(points_a,points_b)
        coefficient_matrix = np.zeros((8,9))
        for i in range(8):
            coefficient_matrix[i]=[p1[i,0]*p2[i,0],p1[i,0]*p2[i,1],p1[i,0],
                                p1[i,1]*p2[i,0],p1[i,1]*p2[i,1],p1[i,1],
                                p2[i,0],p2[i,1],1]
        essential_matrix = null_space(coefficient_matrix).reshape(3,3)
        return essential_matrix
    

    # def _decompose_essential_matrix(self,essential_matrix):
    #     # decompose  essential matrix into R,T

    #     # decompose essential matrix into R, t (See Hartley and Zisserman 9.13)
    #     U, S, Vt = np.linalg.svd(E)
    #     W = np.array([0.0, -1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0]).reshape(3, 3)

    #     # iterate over all point correspondences used in the estimation of the fundamental matrix
    #     first_inliers = []
    #     second_inliers = []
    #     for i in range(len(mask)):
    #         if mask[i]:
    #             # normalize and homogenize the image coordinates
    #             first_inliers.append(K_inv.dot([first_match_points[i][0], first_match_points[i][1], 1.0]))
    #             second_inliers.append(K_inv.dot([second_match_points[i][0], second_match_points[i][1], 1.0]))

    #     # Determine the correct choice of second camera matrix
    #     # only in one of the four configurations will all the points be in front of both cameras
    #     # First choice: R = U * Wt * Vt, T = +u_3 (See Hartley Zisserman 9.19)
    #     R = U.dot(W).dot(Vt)
    #     T = U[:, 2]

