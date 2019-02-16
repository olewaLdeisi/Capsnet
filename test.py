"""
-------------------------------------------------
   File Name：     test
   Description :
   Author :        lin
   Software:       PyCharm
   date：          2019/2/8 13:49
-------------------------------------------------
   Change Activity:
                   2019/2/8 13:49
-------------------------------------------------
"""
__author__ = 'lin'

import numpy as np
import cv2

def img_polar_transform(input_img,center,r_range,theta_rangge=(0,360),r_step=0.5,theta_step=360.0/(180*8)):
    minr,maxr=r_range
    mintheta,maxtheta=theta_rangge
    H=int((maxr-minr)/r_step+1)
    W=int((maxtheta-mintheta)/theta_step+1)
    output_img=125*np.ones((H,W),input_img.dtype)
    x_center,y_center=center
    #极坐标变换
    r=np.linspace(minr,maxr,H)
    r=np.tile(r,(W,1))
    r=np.transpose(r)#矩阵的转置
    theta=np.linspace(mintheta,maxtheta,W)
    theta=np.tile(theta,(H,1))#在垂直方向重复H次，水平重复1次
    x,y=cv2.polarToCart(r,theta,angleInDegrees=True)
    #最邻近插值
    for i in range(H):
        for j in range(W):
            px=int(round(x[i,j])+x_center)
            py = int(round(y[i, j]) + y_center)
            if ((px>=0 and px<=W-1) and (py>=0 and py<=H-1)):
                output_img[i,j]=input_img[px,py]
    return output_img

input_img = cv2.imread('data/img/001.png',cv2.IMREAD_COLOR)
h,w=input_img.shape[:2]
center=(int(h/2),int(w/2))
r_range=(0,300)
output_img=img_polar_transform(input_img,center,r_range)
cv2.namedWindow("img")
cv2.namedWindow("polar")
cv2.imshow('img',input_img)
cv2.imshow('polar',output_img)
cv2.waitKey(0)
cv2.destroyAllWindows()