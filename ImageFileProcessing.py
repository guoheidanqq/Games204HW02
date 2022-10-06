"""
@author
guojianlin
add for process file for matlab and python exchange

"""

import numpy as np
import csv
import json


class ImageIO(object):
    def __init__(self, img):
        self.img = img
        self.WIDTH = img.shape[1]
        self.HEIGHT = img.shape[0]

    def write_to_txt(self, filename):
        np.savetxt(filename, self.img, fmt='%5d')


