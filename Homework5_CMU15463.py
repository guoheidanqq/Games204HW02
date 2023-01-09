import time
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
import os
from matplotlib.colors import LightSource
from mpl_toolkits.mplot3d import Axes3D
from cp_hw2 import *
image1 = io.imread('data/stereo/input_1.tif')
image2 = io.imread('data/stereo/input_2.tif')
image3 = io.imread('data/stereo/input_3.tif')
image4 = io.imread('data/stereo/input_4.tif')
image5 = io.imread('data/stereo/input_5.tif')
image6 = io.imread('data/stereo/input_6.tif')
image7 = io.imread('data/stereo/input_7.tif')
HEIGHT = 431
WIDTH = 369
images_input =np.stack((image1,image2,image3,image4,image5,image6,image7),axis=3)/65536.0
plt.imshow(images_input[...,0])
plt.show()

XYZ = np.zeros_like(images_input)
N = 7
P = HEIGHT*WIDTH
for i in range(0,N):
    XYZ[...,i] = lRGB2XYZ(images_input[...,i])

Y = XYZ[:,:,1,:]

Y_7P = np.zeros((7,P))
for i in range(0,N):
    Y_7P[i,:] = Y[:,:,i].flatten()

plt.imshow(Y_7P)
plt.show()

plt.imshow(Y[...,1],cmap='gray')
plt.show()

plt.imshow(XYZ[...,1])
plt.show()


# light matrix L
L = np.zeros((3,7))
LT = L.T
# psudo normal matrix B
B = np.zeros((3,P))
I=LT@B

U,S,V = np.linalg.svd(Y_7P,full_matrices=False)
S_reduce = np.zeros_like(S)
S_reduce[0:3] = S[0:3]
Y_7P_recon = U@S@V




#step plot surface of reconstructed image
z = Y[..., 1]
HEIGHT = z.shape[0]
WIDTH = z.shape[1]
x,y = np.meshgrid(np.arange(0,WIDTH),np.arange(0,HEIGHT))
ls = LightSource()
fig = plt.figure()
ax = fig.gca(projection ='3d')
color_shade  = ls.shade(z, plt.cm.gray)
surf = ax.plot_surface(x, y, z,facecolors=color_shade,rstride = 4 ,cstride=4)
plt.show()
