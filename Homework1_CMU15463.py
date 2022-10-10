import scipy
from skimage import io
from skimage import data
from matplotlib import pyplot as plt
import numpy as np
from BayerDomainProcessor import *
from scipy import interpolate

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
HEIGHT = img.shape[0]
WIDTH = img.shape[1]
R = np.zeros_like(img)
G = np.zeros_like(img)
G = np.zeros_like(img)
B = np.zeros_like(img)
R[0::2, 0::2] = img[0::2, 0::2]
G[0::2, 1::2] = img[0::2, 1::2]
G[1::2, 0::2] = img[1::2, 0::2]
B[1::2, 1::2] = img[1::2, 1::2]
X_R = np.arange(0, HEIGHT, 2)
Y_R = np.arange(0, WIDTH, 2)
Z_R = R[0::2, 0::2]
f_R = interpolate.interp2d(Y_R, X_R, Z_R, kind='linear')  # Y for cols X for rows
X_B = np.arange(1, HEIGHT, 2)
Y_B = np.arange(1, WIDTH, 2)
Z_B = B[1::2, 1::2]
f_B = interpolate.interp2d(Y_B, X_B, Z_B, kind='linear')
rows = np.arange(0, HEIGHT, 1)
cols = np.arange(0, WIDTH, 1)
R_new = f_R(cols, rows)
B_new = f_B(cols,rows)

G_rows,G_cols = np.nonzero(G)
Z_G = G[G_rows,G_cols]



