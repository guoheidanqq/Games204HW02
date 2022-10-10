import scipy
from skimage import io
from skimage import data
from matplotlib import pyplot as plt
import numpy as np
from BayerDomainProcessor import *
from scipy import interpolate
from matplotlib import pylab

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
img_whitebalanced = awb.execute_gaincontrol()
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
xyz2cam = np.array([[ 0.5271, -0.0712, -0.0347],
                    [-0.6153,  1.3653,  0.2763],
                    [-0.1601,  0.2366,  0.7242]])
srgb2xyz =   np.array(
              [[0.412453, 0.357580, 0.180423] ,
              [0.212671, 0.715160, 0.072169],
              [0.019334, 0.119193, 0.950227]] )
#xyz2cam = np.matrix(xyz2cam)
#srgb2xyz =  np.matrix(srgb2xyz)
srgb2cam = xyz2cam @ srgb2xyz
cam2srgb = np.linalg.inv(srgb2cam)
cam2srgb @ cfa_img_processed




# G_rows, G_cols = np.nonzero(G)
# Z_G = G[G_rows, G_cols]
# X_G = G_rows[0:10]
# Y_G = G_cols[0:10]
# Z_G = Z_G[0:10]
# f_G = interpolate.interp2d(Y_G, X_G, Z_G, kind='linear')
