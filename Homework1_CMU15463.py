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
from cp_exr import *

rawimg_uint16 = io.imread('DSC02878.tiff')
plt.imshow(rawimg_uint16, cmap='gray')
plt.colorbar()
plt.show()
rawimg_blackcompensation = rawimg_uint16.copy()
rawimg_blackcompensation = rawimg_blackcompensation.astype(np.float64)
rawimg_blackcompensation = (rawimg_blackcompensation - 512)
rawimg_blackcompensation = rawimg_blackcompensation / (2 ** 14 - 512)
rawimg_blackcompensation = np.clip(rawimg_blackcompensation, 0, 1)
plt.imshow(rawimg_blackcompensation, cmap='gray')
plt.colorbar()
plt.show()

img_whitebalancing = rawimg_blackcompensation.copy()
plt.imshow(img_whitebalancing, cmap='gray')
plt.show()
paramters = [2.758397, 1.000000, 1.238742, 1.000000]
r_gain = paramters[0]
gr_gain = paramters[1]
gb_gain = paramters[2]
b_gain = paramters[3]
awb = AWB(img_whitebalancing, paramters, 'rggb', 2 ** 14)
# img_whitebalanced = awb.execute_gaincontrol()
img_whitebalanced = awb.execute_whiteworld()
plt.imshow(img_whitebalanced, cmap='gray')
plt.show()

cfa_img_pre = img_whitebalanced.copy()
img = cfa_img_pre
cfa = CFA_Interpolation(img, 'bilinear', 'rggb', 1)
cfa_img_processed = cfa.execute_bilinear()
plt.imshow(cfa_img_processed)
plt.show()

#    { "Sony ILCE-7M2", 0, 0,
#	{ 5271,-712,-347,-6153,13653,2763,-1601,2366,7242 } },
xyz2cam = np.array([5271, -712, -347, -6153, 13653, 2763, -1601, 2366, 7242]) / 10000
xyz2cam = np.array([[0.5271, -0.0712, -0.0347],
                    [-0.6153, 1.3653, 0.2763],
                    [-0.1601, 0.2366, 0.7242]])
srgb2xyz = np.array(
    [[0.412453, 0.357580, 0.180423],
     [0.212671, 0.715160, 0.072169],
     [0.019334, 0.119193, 0.950227]])
# xyz2cam = np.matrix(xyz2cam)
# srgb2xyz =  np.matrix(srgb2xyz)
srgb2cam = xyz2cam @ srgb2xyz
cam2srgb = np.linalg.inv(srgb2cam)

cam2srgb_norm = np.zeros_like(cam2srgb)
rows_norm = cam2srgb.sum(axis=1)
rows_norm = rows_norm.reshape(3, 1)
cam2srgb_norm = cam2srgb / rows_norm
cam2srgb / cam2srgb_norm

HEIGHT = cfa_img_processed.shape[0]
WIDTH = cfa_img_processed.shape[1]
cfa_img_cam2srg = np.zeros_like(cfa_img_processed)
for i in range(0, HEIGHT):
    for j in range(0, WIDTH):
        cfa_img_cam2srg[i, j, :] = cam2srgb_norm @ cfa_img_processed[i, j, :]
cfa_img_cam2srg = cfa_img_cam2srg.clip(0, 1)
plt.imshow(cfa_img_cam2srg)
plt.show()

cfa_img_gray = color.rgb2gray(cfa_img_cam2srg)
plt.imshow(cfa_img_gray, cmap='gray')
plt.show()
cfa_mean_gray = cfa_img_gray.mean()
cfa_gray_gain = 0.25 / cfa_mean_gray
cfa_img_cam2srg_brighter = cfa_img_cam2srg * cfa_gray_gain
cfa_img_cam2srg_brighter = cfa_img_cam2srg_brighter.clip(0, 1)
plt.imshow(cfa_img_cam2srg_brighter)
plt.show()
awb1 = AWB(cfa_img_cam2srg_brighter, paramters, 'rggb', 2 ** 14)
cfa_reaw = awb1.execute_whiteworld()
cfa_img_cam2srg_brighter = cfa_reaw
plt.imshow(cfa_img_cam2srg_brighter)
plt.show()

# YUV change
yuv = RGB2YUV(cfa_reaw)
huescontrol = HueSaturationControl(yuv, -30, 1, 1)
yuv_adjusted = huescontrol.execute()
rgb = YUV2RGB(yuv_adjusted)
plt.imshow(rgb)
plt.show()

# for hue in range(0, 360, 5):
#     yuv = RGB2YUV(cfa_reaw)
#     huescontrol = HueSaturationControl(yuv, hue, 1, 1)
#     yuv_adjusted = huescontrol.execute()
#     rgb = YUV2RGB(yuv_adjusted)
#     plt.imshow(rgb)
#     plt.show()
#     plt.title('hue: '+ str(hue))
#     print(hue)

yuv = yuv_adjusted
brightControl = BrightnessContrastControl(yuv, 0, 1, 1)  # 0 for default brightnes ,1 for default constrast
yuvBrighter = brightControl.execute()
rgb = YUV2RGB(yuvBrighter)
plt.imshow(rgb)
plt.show()

cfa_img_cam2srg_brighter = rgb
cfa_img_ca2srgb_gamma = srgb_pre_gamma_compensation(cfa_img_cam2srg_brighter)
plt.imshow(cfa_img_ca2srgb_gamma)
plt.show()
io.imsave('room.png', cfa_img_ca2srgb_gamma)
io.imsave('room.jpg', cfa_img_ca2srgb_gamma, quality=90)
cfa_img_uint8 = (cfa_img_ca2srgb_gamma * 255).astype(np.uint8)

io.imsave('room.png', cfa_img_ca2srgb_gamma)
io.imsave('room.jpg', cfa_img_ca2srgb_gamma, quality=80)
cfa_img_read = io.imread('room.jpg')
cfa_img_ca2srgb_gamma_0_255 = cfa_img_ca2srgb_gamma * 255

# G_rows, G_cols = np.nonzero(G)
# Z_G = G[G_rows, G_cols]
# X_G = G_rows[0:10]
# Y_G = G_cols[0:10]
# Z_G = Z_G[0:10]
# f_G = interpolate.interp2d(Y_G, X_G, Z_G, kind='linear')
cfa_img_read = io.imread('room.jpg') / 255
yuv = RGB2YUV(cfa_img_read)
YUV = yuv.copy()
Y = YUV[:, :, 0]  # Y is in 0 1
plt.imshow(Y, cmap='gray')
plt.show()
gray_std2 = np.std(Y)
space_sigma = 0.4

range_sigma = 0.02
space_sigma = 0.001
kernel_size = 7
Y_down_N = Y[::8, ::8]
plt.imshow(Y_down_N, cmap='gray')
plt.show()
y_base = fastbilateral2d(Y_down_N, range_sigma, space_sigma, kernel_size)
plt.imshow(y_base, cmap='gray')
plt.show()

# cv2.imwrite('1.exr',cfa_img_read)
img = cfa_img_read * (2 ** 14)
Img = img.astype(np.uint16)
writeEXR('1.exr', Img, 'HALF')
readImg = readEXR('1.exr')
plt.imshow(readImg/(2**14))
plt.show()
sanmiguel = readEXR('sanmiguel_cam25.exr')
plt.imshow(sanmiguel/np.max(sanmiguel))
plt.show()
