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

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
# print(os.path.abspath(__file__))
# print(os.path.dirname(os.path.abspath(__file__)))
image_path = os.path.dirname(os.path.abspath(__file__))

# raw_01_49 = rawpy.imread(os.path.join(image_path, "data\\set01\\DSC00049.ARW"))
# raw_01_50 = rawpy.imread(os.path.join(image_path, "data\\set01\\DSC00049.ARW"))
# raw_01_51 = rawpy.imread(os.path.join(image_path, "data\\set01\\DSC00049.ARW"))
# raw_01_52 = rawpy.imread(os.path.join(image_path, "data\\set01\\DSC00049.ARW"))
# raw_01_53 = rawpy.imread(os.path.join(image_path, "data\\set01\\DSC00049.ARW"))
#
# raw_02_113= rawpy.imread(os.path.join(image_path, "data\\set02\\DSC00113.ARW"))
# raw_02_114 = rawpy.imread(os.path.join(image_path, "data\\set02\\DSC00114.ARW"))
# raw_02_115 = rawpy.imread(os.path.join(image_path, "data\\set02\\DSC00115.ARW"))


raw_03_163 = rawpy.imread(os.path.join(image_path, "data\\set03\\DSC00163.ARW"))  # 1/1000 iso 25600 f/5.6
raw_03_164 = rawpy.imread(os.path.join(image_path, "data\\set03\\DSC00164.ARW"))  # 1/500
raw_03_165 = rawpy.imread(os.path.join(image_path, "data\\set03\\DSC00165.ARW"))  # 1/250
raw_03_166 = rawpy.imread(os.path.join(image_path, "data\\set03\\DSC00166.ARW"))  # 1/125
raw_03_167 = rawpy.imread(os.path.join(image_path, "data\\set03\\DSC00167.ARW"))  # 1/60
raw_03_168 = rawpy.imread(os.path.join(image_path, "data\\set03\\DSC00168.ARW"))  # 1/30
raw_03_169 = rawpy.imread(os.path.join(image_path, "data\\set03\\DSC00169.ARW"))  # 1/15

images = [raw_03_163, raw_03_164, raw_03_165, raw_03_166, raw_03_167, raw_03_168, raw_03_169]
exposure_times = np.array([1 / 1000, 1 / 500, 1 / 250, 1 / 125, 1 / 60, 1 / 30, 1 / 15])
masks = compute_mask(images)
weights = get_fusion_weight(images)
fuse_image = raw_exposure_fuse(images, weights, exposure_times)
fuse_image_01 = fuse_image / np.max(fuse_image)
fuse_image_255 = fuse_image_01*255
fuse_image_uint8 = fuse_image_255.astype(np.uint8)
fuse_image_uint32 = fuse_image.astype(np.uint32)
plt.imshow(np.log(1+fuse_image_uint32), cmap='gray')
plt.show()

color_image = CFA_Interpolation_function(fuse_image_uint32)
HDR_image_log_base = fastbilateral2d(color_image)

color_image_uint8 = color_image
color_image_01 = color_image_uint8 / np.max(color_image_uint8)
plt.imshow(color_image_01)
plt.show()

plt.imshow(fuse_image, cmap='gray')
plt.show()

plt.imshow(np.log(fuse_image + 1), cmap='gray')
plt.show()

image0 = images[0].raw_image
plt.imshow(image0, cmap='gray')
plt.show()
image1 = images[1].raw_image
plt.imshow(image1, cmap='gray')
plt.show()
image2 = images[2].raw_image
plt.imshow(image2, cmap='gray')
plt.show()
image3 = images[3].raw_image
plt.imshow(image3, cmap='gray')
plt.show()
image4 = images[4].raw_image
plt.imshow(image4, cmap='gray')
plt.show()
image5 = images[5].raw_image
plt.imshow(image5, cmap='gray')
plt.show()

awb = AWB(fuse_image, [1, 1, 1, 1], 'rggb', 2 ** 32)
Bayer_awb = awb.execute()
cfa_interpolator = CFA_Interpolation(Bayer_awb, 'malvar', 'rggb', 2 ** 32)
cfa_img = cfa_interpolator.execute()
cfa_img_log = np.log(1+cfa_img)
cfa_img_log_uint8 = cfa_img_log.astype(np.uint8)
plt.imshow(cfa_img_log_uint8*20)
plt.show()

# a = imageio.v2.imread('IMG_20220915_192911.dng')
# imageio.imwrite('a.jpg', color_image.astype(np.uint8))
# imageio.imwrite('test1.exr', color_image.astype(np.float64))
# cv2.imwrite('test1.exr', color_image.astype(np.float64))
