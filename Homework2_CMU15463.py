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
    filename = 'data/door_stack/exposure' + str(i) + '.jpg'
    imgTmp = io.imread(filename)
    images.append(imgTmp)
# display images list
for i in range(1, 17):
    plt.subplot(4, 4, i)
    plt.imshow(images[i - 1])
    plt.axis('off')
plt.show()
