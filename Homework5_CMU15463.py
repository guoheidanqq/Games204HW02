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

