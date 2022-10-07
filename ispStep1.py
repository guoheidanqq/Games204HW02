# -*- coding: utf-8 -*-
"""
Created on Fri Jul 29 13:45:23 2022

@author: qilin sun
"""

# Part of the codes are from OpenISP, HDR plus and Halid
# Do not cheat and post any code to the public!

from BayerDomainProcessor import *
from ImageFileProcessing import *
from RGBDomainProcessor import *
from YUVDomainProcessor import *

import rawpy
import os
import sys
import time
import cv2  # only for save images to jpg

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
print(os.path.abspath(__file__))
print(os.path.dirname(os.path.abspath(__file__)))

from matplotlib import pyplot as plt  # use it only for debug

import numpy as np

raw_path = 'test.RAW'
config_path = 'config.csv'
raw = rawpy.imread(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'DSC02878.ARW'))

rgb2xyz = np.asarray([[0.5767309, 0.1855540, 0.1881852],
                      [0.2973769, 0.6273491, 0.0752741],
                      [0.0270343, 0.0706872, 0.9911085]])

xyz2rgb = np.asarray([[2.0413690, -0.5649464, -0.3446944],
                      [-0.9692660, 1.8760108, 0.0415560],
                      [0.0134474, -0.1183897, 1.0154096]])

raw_h, raw_w = raw.raw_image.shape
dpc_thres = 30
dpc_mode = 'gradient'
dpc_clip = raw.camera_white_level_per_channel[0]
bl_r, bl_gr, bl_gb, bl_b = raw.black_level_per_channel
alpha, beta = 0, 0
blackLevelCompensation_clip = raw.camera_white_level_per_channel[0]
bayer_pattern = 'rggb'
r_gain, gr_gain, gb_gain, b_gain = 1 / (1996 / 1024), 1.0, 1.0, 1 / (2080 / 1024)
awb_clip = raw.camera_white_level_per_channel[0]
cfa_mode = 'malvar'
cfa_clip = raw.camera_white_level_per_channel[0]
csc = np.asarray([[0.299, 0.587, 0.114], [-0.1687, -0.3313, 0.5], [0.5, -0.4187, -0.0813], [0, 128, 128]])
ccm = np.asarray([[1024., 0., 0., 0.], [0., 1024., 0., 0.], [0., 0., 1024., 0.]])
edge_filter = np.asarray([[-1., 0., -1., 0., -1.], [-1., 0., 8., 0., -1.], [-1., 0., -1., 0., -1.]])
ee_gain, ee_thres, ee_emclip = [32, 128], [32, 64], [-64, 64]
fcs_edge, fcs_gain, fcs_intercept, fcs_slope = [32, 64], 32, 2, 3
nlm_h, nlm_clip = 10, 255
bnf_dw = np.asarray(
    [[8., 12., 32., 12., 8.], [12., 64., 128., 64., 12.], [32., 128., 1024., 128., 32.], [0., 0., 0., 0., 0.],
     [0., 0., 0., 0., 0.]])
bnf_rw, bnf_rthres, bnf_clip = [0, 8, 16, 32], [128, 32, 8], 255
ee_gain, ee_thres, ee_emclip = [32, 128], [32, 64], [-64, 64]
dpc_thres, dpc_mode, dpc_clip = 30, 'gradient', 4095
fcs_edge, fcs_gain, fcs_intercept, fcs_slope = [32, 32], 32, 2, 3
nlm_h, nlm_clip = 15, 255
hue, saturation, hsc_clip, brightness, contrast, bcc_clip = 128, 256, 255, 10, 10, 255

rawimg = raw.raw_image
rawimg = np.clip(rawimg, raw.black_level_per_channel[0], 2 ** 14)
# min = 512 max = 2**14
print(50 * '-' + '\nLoading RAW Image Done......')
plt.imshow(rawimg, cmap='gray')
plt.show()

#  step  last  write csv files
# rawImgFile = ImageIO(rawimg)
# rawImgFile.write_to_txt('rawImg.csv')

#################################################################################################################
#####################################  Part 1: Bayer Domain Processing Steps  ###################################
#################################################################################################################
# t_start = time.time()
# # Step 1. Dead Pixel Correction (10pts)
# dpc = deadPixelCorrection(rawimg, dpc_thres, dpc_mode, dpc_clip)
# Bayer_dpc = dpc.execute()
# print(50 * '-' + '\n 1.1 Dead Pixel Correction Done......')
# plt.imshow(rawimg, cmap='gray')
# plt.show()


# # Step 2.'Black Level Compensation' (5pts)
# parameter = raw.black_level_per_channel + [alpha, beta]
# # blkC = blackLevelCompensation(Bayer_dpc, parameter, bayer_pattern, clip = 2**14)
# blkC = blackLevelCompensation(rawimg, parameter, bayer_pattern, clip=2 ** 14)
# Bayer_blackLevelCompensation = blkC.execute()
# print(50 * '-' + '\n 1.2 Black Level Compensation Done......')
# plt.imshow(Bayer_blackLevelCompensation, cmap='gray')
#
# # Step 4. Anti Aliasing Filter (10pts)
# antiAliasingFilter = antiAliasingFilter(Bayer_blackLevelCompensation)
# Bayer_antiAliasingFilter = antiAliasingFilter.execute()
# print(50 * '-' + '\n 1.4 Anti-aliasing Filtering Done......')
# plt.imshow(Bayer_antiAliasingFilter, cmap='gray')
# plt.show()

# Step 5. Auto White Balance and Gain Control (10pts)
parameter = [r_gain, gr_gain, gb_gain, b_gain]
awb = AWB(rawimg, parameter, bayer_pattern, awb_clip)
Bayer_awb = awb.execute()
print(50*'-' + '\n 1.5 White Balance Gain Done......')
plt.imshow(Bayer_awb,cmap = 'gray')
plt.show()

# Step 6. Chroma Noise Filtering (Extra 20pts)
cnf = ChromaNoiseFiltering(Bayer_awb, bayer_pattern, 0, parameter, cfa_clip)
Bayer_cnf = cnf.execute()
print(50*'-' + '\n 1.6 Chroma Noise Filtering Done......')

