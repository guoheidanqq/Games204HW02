import scipy
from skimage import io
from skimage import data
from matplotlib import pyplot as plt
import numpy as np
from BayerDomainProcessor import *
from RGBDomainProcessor import *
from scipy import interpolate
from matplotlib import pylab
from skimage import color
from RGBDomainProcessor import *
from YUVDomainProcessor import *
from HDR_fusion import *

# imread to images list
images = []
for i in range(1, 17):
    filename = 'data/door_stack/exposure' + str(i) + '.tiff'
    imgTmp = io.imread(filename)
    imgTmp = imgTmp.astype(np.float32)
    imgTmpNormalized01 = imgTmp / (2 ** 16 - 1)
    images.append(imgTmpNormalized01)

ex_times = []
for i in range(1, 17):
    ex_time = 2 ** (i - 1) / 2048
    ex_times.append(ex_time)
exposure_times = np.array(ex_times)

# display images list
# for i in range(1, 17):
#     plt.subplot(4, 4, i)
#     plt.imshow(images[i - 1])
#     plt.axis('off')
# plt.show()

masks = compute_mask_01(images)
weights = get_gaussian_weights(images)
fuse_image = raw_exposure_fuse_01(images, weights, exposure_times)

#
fuse_image_01 = fuse_image / np.max(fuse_image)
plt.imshow(np.power(fuse_image_01, 0.45))
plt.show()

# Tone mapping
K = 0.05
B = 0.95
eps = 10 ** -10
Iij_HDR = fuse_image
Iij_HDR_log = np.log(Iij_HDR + eps)
Iij_HDR_log_mean = np.mean(Iij_HDR_log)
Im_HDR = np.exp(Iij_HDR_log_mean)

I_tilt_ij_HDR = (K / Im_HDR) * Iij_HDR
I_tilt_white = B * np.max(I_tilt_ij_HDR)
Iij_TM = I_tilt_ij_HDR * (1 + I_tilt_ij_HDR / (I_tilt_white ** 2)) / (1 + I_tilt_ij_HDR)

plt.imshow(np.power(Iij_TM / np.max(Iij_TM), 1))
plt.show()

plt.imshow(Iij_TM / np.max(Iij_TM))
plt.show()

# noise calibration and optima weights

rampImg = np.tile(np.linspace(0, 1, 255), (255, 1))
plt.imshow(rampImg, cmap='gray')
plt.show()
imageio.imwrite('rampgray.png',rampImg)
