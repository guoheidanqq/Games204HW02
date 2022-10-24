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

# # imread to images list
# images = []
# for i in range(1, 17):
#     filename = 'data/door_stack/exposure' + str(i) + '.tiff'
#     imgTmp = io.imread(filename)
#     imgTmp = imgTmp.astype(np.float32)
#     imgTmpNormalized01 = imgTmp / (2 ** 16 - 1)
#     images.append(imgTmpNormalized01)
#
# ex_times = []
# for i in range(1, 17):
#     ex_time = 2 ** (i - 1) / 2048
#     ex_times.append(ex_time)
# exposure_times = np.array(ex_times)
#
# # display images list
# # for i in range(1, 17):
# #     plt.subplot(4, 4, i)
# #     plt.imshow(images[i - 1])
# #     plt.axis('off')
# # plt.show()
#
# masks = compute_mask_01(images)
# weights = get_gaussian_weights(images)
# fuse_image = raw_exposure_fuse_01(images, weights, exposure_times)
#
# #
# fuse_image_01 = fuse_image / np.max(fuse_image)
# plt.imshow(np.power(fuse_image_01, 0.45))
# plt.show()
#
# # Tone mapping
# K = 0.05
# B = 0.95
# eps = 10 ** -10
# Iij_HDR = fuse_image
# Iij_HDR_log = np.log(Iij_HDR + eps)
# Iij_HDR_log_mean = np.mean(Iij_HDR_log)
# Im_HDR = np.exp(Iij_HDR_log_mean)
#
# I_tilt_ij_HDR = (K / Im_HDR) * Iij_HDR
# I_tilt_white = B * np.max(I_tilt_ij_HDR)
# Iij_TM = I_tilt_ij_HDR * (1 + I_tilt_ij_HDR / (I_tilt_white ** 2)) / (1 + I_tilt_ij_HDR)
#
# plt.imshow(np.power(Iij_TM / np.max(Iij_TM), 1))
# plt.show()
#
# plt.imshow(Iij_TM / np.max(Iij_TM))
# plt.show()
#
# # noise calibration and optima weights
#
# rampImg = np.tile(np.linspace(0, 1, 255), (255, 1))
# plt.imshow(rampImg, cmap='gray')
# plt.show()
# imageio.imwrite('rampgray.png', rampImg)


# noise calibration
noise_images = []
for i in range(1, 51):
    numstr = str(i).zfill(2)
    filename = 'data/blacknoise/' + numstr + '.tiff'
    imgTmp = io.imread(filename)
    if imgTmp.shape[0] > imgTmp.shape[1]:
        imgTmp = np.transpose(imgTmp, [1, 0, 2])
    noise_images.append(imgTmp)

clip_min = 64
clip_max = 1023
paper_images = []
for i in range(1, 51):
    numstr = str(i).zfill(2)
    filename = 'data/whitepaper/' + numstr + '.tiff'
    imgTmp = io.imread(filename)
    if imgTmp.shape[0] > imgTmp.shape[1]:
        imgTmp = np.transpose(imgTmp, (1, 0, 2))
    imgTmp
    paper_images.append(imgTmp)

black_frame = np.zeros_like(noise_images[0], np.float32)
for i in range(0, 50):
    black_frame = black_frame + noise_images[i]
black_frame = black_frame / 50
black_frame = black_frame.astype(np.uint16)
paper_images_noise_calibration = []
for i in range(0, 50):
    tmp_noise_calibration = paper_images[i].astype(np.float32) - black_frame.astype(np.float32)
    paper_images_noise_calibration.append(tmp_noise_calibration)

# caculate mean image and variance image for each pixel
images = paper_images_noise_calibration
images_mean = np.zeros_like(images[0], dtype=np.float32)
images_variance = np.zeros_like(images[0], dtype=np.float32)
for i in range(0, 50):
    images_mean = images_mean + images[i]

images_mean = images_mean / 50

for i in range(0, 50):
    images_variance = images_variance + (images[i] - images_mean) ** 2

images_variance = 1.0 / (50.0 - 1) * images_variance

xx = images_mean.reshape(-1)
yy = images_variance.reshape(-1)
plt.scatter(xx,yy)
plt.show()

plt.imshow(black_frame / np.max(black_frame))
plt.show()

plt.imshow(paper_images[3] / np.max(paper_images[3]))
plt.show()

plt.imshow(paper_images_noise_calibration[3] / np.max(paper_images_noise_calibration[3]))
plt.show()
# plot the histogram of noise
local_img = black_frame[::, 0]
a = local_img.reshape(-1)
plt.hist(a, bins=255)
plt.show()

plt.imshow(paper_images_noise_calibration[1] / 2 ** 16)
plt.show()
