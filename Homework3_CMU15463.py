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

lamp_ambient = io.imread('data/lamp/lamp_ambient.tif')
lamp_flash = io.imread('data/lamp/lamp_flash.tif')
plt.sub
plt.imshow(lamp_ambient)
plt.show()