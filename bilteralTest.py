import numpy as np
import rawpy
import os
import sys
import matplotlib.pyplot as plt
from HDR_fusion import *
from BayerDomainProcessor import *
from RGBDomainProcessor import *
from YUVDomainProcessor import *
from skimage import io
from skimage import data
import time
import imageio
import cv2

kernel = gaussian_kernel(3, 2)
local_img = np.arange(1, 49 + 1).reshape((7, 7))
value_kernel = value_kernel(7, 2, local_img)

camera = data.camera() / 255
# plt.imshow(camera, cmap='gray')
# plt.show()

HDR_image_gray = camera
std = np.std(camera)
range_sigma = 0.02
space_sigma = 0.001
kernel_size = 7
img_bilateral = fastbilateral2d(HDR_image_gray, range_sigma, space_sigma, kernel_size)
plt.imshow(img_bilateral, cmap='gray')
# plt.show()
# plt.title('image bilateral')

plt.subplot(1,2,1)
plt.imshow(camera, cmap='gray')
plt.title('origin')
plt.subplot(1,2,2)
plt.imshow(img_bilateral, cmap='gray')
plt.title('image bilateral')
plt.show()
