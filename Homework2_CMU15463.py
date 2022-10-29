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
from matplotlib.patches import Rectangle
from cp_hw2 import *

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
fuse_image_gamma = np.power(fuse_image_01, 0.45)

plt.imshow(fuse_image_gamma * 10)
# [x,y] = plt.ginput()
# row 1
plt.gca().add_patch(Rectangle((3310, 625), 50, 50, linewidth=1, edgecolor='r', facecolor='none'))
plt.gca().add_patch(Rectangle((3485, 625), 50, 50, linewidth=1, edgecolor='r', facecolor='none'))
plt.gca().add_patch(Rectangle((3635, 625), 50, 50, linewidth=1, edgecolor='r', facecolor='none'))
plt.gca().add_patch(Rectangle((3785, 625), 50, 50, linewidth=1, edgecolor='r', facecolor='none'))
# row2
plt.gca().add_patch(Rectangle((3310, 800), 50, 50, linewidth=1, edgecolor='r', facecolor='none'))
plt.gca().add_patch(Rectangle((3485, 800), 50, 50, linewidth=1, edgecolor='r', facecolor='none'))
plt.gca().add_patch(Rectangle((3635, 800), 50, 50, linewidth=1, edgecolor='r', facecolor='none'))
plt.gca().add_patch(Rectangle((3785, 800), 50, 50, linewidth=1, edgecolor='r', facecolor='none'))
# row3
plt.gca().add_patch(Rectangle((3310, 950), 50, 50, linewidth=1, edgecolor='r', facecolor='none'))
plt.gca().add_patch(Rectangle((3485, 950), 50, 50, linewidth=1, edgecolor='r', facecolor='none'))
plt.gca().add_patch(Rectangle((3635, 950), 50, 50, linewidth=1, edgecolor='r', facecolor='none'))
plt.gca().add_patch(Rectangle((3785, 950), 50, 50, linewidth=1, edgecolor='r', facecolor='none'))
# row4
plt.gca().add_patch(Rectangle((3310, 1125), 50, 50, linewidth=1, edgecolor='r', facecolor='none'))
plt.gca().add_patch(Rectangle((3485, 1125), 50, 50, linewidth=1, edgecolor='r', facecolor='none'))
plt.gca().add_patch(Rectangle((3635, 1125), 50, 50, linewidth=1, edgecolor='r', facecolor='none'))
plt.gca().add_patch(Rectangle((3785, 1125), 50, 50, linewidth=1, edgecolor='r', facecolor='none'))
# row5
plt.gca().add_patch(Rectangle((3310, 1275), 50, 50, linewidth=1, edgecolor='r', facecolor='none'))
plt.gca().add_patch(Rectangle((3485, 1275), 50, 50, linewidth=1, edgecolor='r', facecolor='none'))
plt.gca().add_patch(Rectangle((3635, 1275), 50, 50, linewidth=1, edgecolor='r', facecolor='none'))
plt.gca().add_patch(Rectangle((3785, 1275), 50, 50, linewidth=1, edgecolor='r', facecolor='none'))
# row6
plt.gca().add_patch(Rectangle((3310, 1450), 50, 50, linewidth=1, edgecolor='r', facecolor='none'))
plt.gca().add_patch(Rectangle((3485, 1450), 50, 50, linewidth=1, edgecolor='r', facecolor='none'))
plt.gca().add_patch(Rectangle((3635, 1450), 50, 50, linewidth=1, edgecolor='r', facecolor='none'))
plt.gca().add_patch(Rectangle((3785, 1450), 50, 50, linewidth=1, edgecolor='r', facecolor='none'))
plt.show()

# create rectangle patches

color_set = read_color_checker_from_image(fuse_image_gamma)
r, g, b = read_colorchecker_gm()
color_checker_4_6 = np.stack((r, g, b), axis=2)
plt.imshow(color_checker_4_6)
plt.show()

plt.imshow(fuse_image_gamma*10)
plt.show()

color_checker_gm_50_50 = np.zeros_like(color_set)
for i in range(0, 4):
    for j in range(0, 6):
        color_checker_gm_50_50[i * 50:i * 50 + 50, j * 50:j * 50 + 50] = color_checker_4_6[i, j]
plt.imshow(color_checker_gm_50_50)
plt.show()

# construct A  4 * 60000

homo_ones_matrix = np.ones((color_set.shape[0], color_set.shape[1]))
color_set_homo = np.stack((color_set[:, :, 0], color_set[:, :, 1], color_set[:, :, 0], homo_ones_matrix), axis=2)
color_checker_gm_50_50_homo = np.stack(
    (color_checker_gm_50_50[:, :, 0], color_checker_gm_50_50[:, :, 1], color_checker_gm_50_50[:, :, 2],
     homo_ones_matrix), axis=2)
A = color_set_homo.reshape(-1,4)
b = color_checker_gm_50_50_homo.reshape(-1,4)
X = np.linalg.lstsq(A,b,rcond=None)
color_correction_matrix = X[0]

# construct b


gray = (r + g + b) / 3

plt.imshow(color_set / np.max(color_set))
plt.show()
plt.imshow(images[15])
plt.show()

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


# # imread to images list
# images = []
# for i in range(1, 17):
#     filename = 'data/door_stack/exposure' + str(i) + '.jpg'
#     imgTmp = io.imread(filename)
#     imgTmp = imgTmp.astype(np.uint8)
#     imgTmpNormalized01 = imgTmp
#     images.append(imgTmpNormalized01)
#
# ex_times = []
# for i in range(1, 17):
#     ex_time = 2 ** (i - 1) / 2048
#     ex_times.append(ex_time)
# exposure_times = np.array(ex_times)
#
# # display images list
# for i in range(1, 17):
#     plt.subplot(4, 4, i)
#     plt.imshow(images[i - 1])
#     plt.axis('off')
# plt.show()
#
# N = 200
# images_downsampling = []
# for i in range(0, 16):
#     img = images[i]
#     img = img[::N, ::N]
#     images_downsampling.append(img)
#
# for i in range(1, 17):
#     plt.subplot(4, 4, i)
#     plt.imshow(images_downsampling[i - 1])
#     plt.axis('off')
# plt.show()
#
# images_downsampling_01 = []
# for i in range(0, len(images_downsampling)):
#     tmp = images_downsampling[i] / 255.0
#     images_downsampling_01.append(tmp)
# weights = get_tent_weights(images_downsampling_01)
#
# # construct matrix A  b
# Height, Width, channels = images_downsampling[0].shape
# I = np.zeros((len(images_downsampling), Height * Width * channels))
# T = np.zeros_like(I)
# for i in range(0, T.shape[0]):
#     T[i, :] = exposure_times[i]
#
# K = len(images_downsampling)
# N = 256
# pixelTotal = I.shape[1]
# for k in range(0, K):
#     I[k, :] = images_downsampling[k].reshape(-1)
#
# N = 256
# K = 16
# pixelTotal = 1800
# lamda = 50
# z = np.arange(0, 256) / 255.0
# w = w_tent(z)
# plt.plot(z, w)
# plt.show()
#
# g, le = g_solve(I, T, w, lamda)
# z = np.arange(0, 256)
# plt.plot(z, g)
# plt.show()
# recovered_radiance = np.exp(np.reshape(le, images_downsampling[0].shape))
# plt.imshow(recovered_radiance / np.max(recovered_radiance))
# plt.show()
#
# # use g function to recover the linear radiance
# images_linear = []
# for i in range(0, 16):
#     img = g[images[i].astype(np.int32)]
#     img_exp = np.exp(img[:, :, :, 0])
#     images_linear.append(img_exp/np.max(img_exp))
#
# # display images linear  list
# for i in range(1, 17):
#     plt.subplot(4, 4, i)
#     plt.imshow(images_linear[i - 1])
#     plt.axis('off')
# plt.show()
#
#
# plt.imshow(images_linear[15])
# plt.show()

# K = I.shape[0]
# N = I.shape[1]
# Z_levels = 256
# lamda = 100
# A = np.zeros((K * N + Z_levels - 2, Z_levels + N))
# b = np.zeros((A.shape[0], 1))
# row = 0
# for z in range(0, 254):
#     A[row, z] = 1 * w[z + 1] * lamda
#     A[row, z + 1] = -2 * w[z + 1] * lamda
#     A[row, z + 2] = 1 * w[z + 1] * lamda
#     row = row + 1
#
# row = 254
# for k in range(0, K):
#     for i in range(0, N):
#         z = I[k, i].astype(np.int32)
#         weight = w[z]
#         ex_time = T[k, i]
#         A[row, z] = weight
#         A[row, i + 256] = -weight
#         b[row] = weight * np.log(ex_time)
#         row = row + 1
#
# X = np.linalg.lstsq(A, b, rcond=None)
#
# g = X[0][0:256]
# z = np.arange(0, 256)
# plt.plot(z, g)
# plt.show()


# initialize the lamda term


# noise calibration
# noise_images = []
# for i in range(1, 51):
#     numstr = str(i).zfill(2)
#     filename = 'data/blacknoise/' + numstr + '.tiff'
#     imgTmp = io.imread(filename)
#     if imgTmp.shape[0] < imgTmp.shape[1]:
#         imgTmp = np.transpose(imgTmp, [1, 0, 2])
#     noise_images.append(imgTmp)
#
# clip_min = 64
# clip_max = 1023
# paper_images = []
# for i in range(1, 51):
#     numstr = str(i).zfill(2)
#     filename = 'data/whitepaper/' + numstr + '.tiff'
#     imgTmp = io.imread(filename)
#     if imgTmp.shape[0] < imgTmp.shape[1]:
#         imgTmp = np.transpose(imgTmp, (1, 0, 2))
#     imgTmp
#     paper_images.append(imgTmp)
#
# black_frame = np.zeros_like(noise_images[0], np.float32)
# for i in range(0, 50):
#     black_frame = black_frame + noise_images[i]
# black_frame = black_frame / 50
# black_frame = black_frame.astype(np.uint16)
# paper_images_noise_calibration = []
# for i in range(0, 50):
#     tmp_noise_calibration = paper_images[i].astype(np.float32) - black_frame.astype(np.float32)
#     paper_images_noise_calibration.append(tmp_noise_calibration)
#
# # caculate mean image and variance image for each pixel
# images = paper_images_noise_calibration
# images_mean = np.zeros_like(images[0], dtype=np.float32)
# images_variance = np.zeros_like(images[0], dtype=np.float32)
# for i in range(0, 50):
#     images_mean = images_mean + images[i]
#
# images_mean = images_mean / 50
#
# for i in range(0, 50):
#     images_variance = images_variance + (images[i] - images_mean) ** 2
#
# images_variance = 1.0 / (50.0 - 1) * images_variance
#
# xx = images_mean.reshape(-1)
# yy = images_variance.reshape(-1)
# a,b = np.polyfit(xx,yy,1)
# plt.scatter(xx, yy)
# plt.scatter(xx,a*xx + b)
# plt.show()
#
# #line fit
#
#
#
# plt.imshow(black_frame / np.max(black_frame))
# plt.show()
#
# plt.imshow(paper_images[3] / np.max(paper_images[3]))
# plt.show()
#
# plt.imshow(paper_images_noise_calibration[3] / np.max(paper_images_noise_calibration[3]))
# plt.show()
# # plot the histogram of noise
# local_img = black_frame[::, 0]
# a = local_img.reshape(-1)
# plt.hist(a, bins=255)
# plt.show()
#
# plt.imshow(paper_images_noise_calibration[1] / 2 ** 16)
# plt.show()
#
# #show mean image
# plt.imshow(np.sqrt(images_mean/np.max(images_mean)))
# plt.show()
# #show variance image
# plt.imshow(np.sqrt(images_variance/np.max(images_variance)))
# plt.show()
