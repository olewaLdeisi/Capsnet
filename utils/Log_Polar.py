"""
-------------------------------------------------
   File Name：     Log_Polar
   Description :
   Author :        lin
   Software:       PyCharm
   date：          2019/2/3 22:33
-------------------------------------------------
   Change Activity:
                   2019/2/3 22:33
-------------------------------------------------
"""
__author__ = 'lin'
import cv2
import math
import numpy as np
import torchvision
from PIL import Image
import os

if not os.path.exists('../data/img1'):
    os.makedirs('../data/img1')

img = Image.open('../data/img/001.png').convert('RGB')
img = torchvision.transforms.ToTensor()(img)


