# coding=utf-8
'''
@Author: yanxinhao
@Email: 1914607611xh@i.shu.edu.cn
@LastEditTime : 2020-01-04 14:11:23
@LastEditors  : yanxinhao
@Description: 
'''
import cv2
from matplotlib import pyplot as plt
import numpy as np
from calibration.epipolar_geometry import EpipolarGeometry

img1 = cv2.imread('./data/0000.JPG')#, cv2.IMREAD_GRAYSCALE)
img2 = cv2.imread('./data/0001.JPG')#, cv2.IMREAD_GRAYSCALE)


# stereo=cv2.StereoBM_create(numDisparities=16,blockSize=15)
# disparity=stereo.compute(img1,img2)
# cv2.imshow("disparity",disparity)
# cv2.waitKey(0)



orb = cv2.ORB_create()
kp1, des1 = orb.detectAndCompute(img1, None)
kp2, des2 = orb.detectAndCompute(img2, None)
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
matches = bf.match(des1, des2)
matches = sorted(matches, key=lambda x: x.distance)

kp_list=[]
kp_list1=[]
kp_list2=[]

for match in matches[:8]:
    kp_list.append((kp1[match.queryIdx].pt,kp2[match.trainIdx].pt))
    kp_list1.append(kp1[match.queryIdx].pt)
    kp_list2.append(kp2[match.trainIdx].pt)
    # print(kp1[match.queryIdx].pt,kp2[match.trainIdx].pt)
match_points_a = np.array(kp_list1).astype(np.int)
match_points_b = np.array(kp_list2).astype(np.int)


camera_matrix = np.array([[5.40756340e+03, 0.00000000e+00, 1.36694579e+03],
       [0.00000000e+00, 5.33216612e+03, 7.99420693e+02],
       [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]])
calu=EpipolarGeometry(camera_matrix,camera_matrix)
calu.calculate(match_points_a,match_points_b)











class Visualize(object):
    def __init__(self,imageA,imageB):
        self.imageA=imageA
        self.imageB=imageB
        # h,w=imageA.shape[:2]
        # self.imagecat=np.zeros((h,w*2,3))
    def visualize(self,match_idxs):
        for match in match_idxs:
            # center is (x,y)
            imgA_center= (int(match[0][0]),int(match[0][1]))
            imgB_center=(int(match[1][0]),int(match[1][1]))
            print(imgA_center,imgB_center)
            self.imageA=cv2.circle(self.imageA,imgA_center,5,(0,0,255),thickness=5)
            self.imageB=cv2.circle(self.imageB,imgB_center,5,(0,0,255),thickness=5)
        cv2.imshow("imga",self.imageA)
        cv2.imshow("imgb",self.imageB)
        cv2.waitKey(0)

# vis=Visualize(img1,img2)
# vis.visualize(kp_list)

# img3 = cv2.drawMatches(img1, kp1, img2, kp2, matches[:80], img2, flags=2)

# # plt.imshow(img3)
# # cv2.imwrite("result.png",img3)
# cv2.imshow("result",img3)
# cv2.waitKey(0)
# # plt.show()