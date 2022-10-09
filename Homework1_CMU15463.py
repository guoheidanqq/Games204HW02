import scipy
from skimage import io
from skimage import data
from matplotlib import pyplot as plt
import numpy as np

rawimg_uint16 = io.imread('DSC02878.tiff')
plt.imshow(rawimg_uint16, cmap='gray')
plt.show()
rawimg_blackcompensation= rawimg_uint16.copy()
rawimg_blackcompensation  = rawimg_blackcompensation - 512
raw_bla


