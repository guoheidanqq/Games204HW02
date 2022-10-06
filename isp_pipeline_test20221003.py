# -*- coding: utf-8 -*-
"""
Created on Fri Jul 29 13:45:23 2022

@author: qilin sun
"""

# Part of the codes are from OpenISP, HDR plus and Halid
# Do not cheat and post any code to the public!

from BayerDomainProcessor import *
from RGBDomainProcessor import *
from YUVDomainProcessor import *

import rawpy
import os
import sys
import time
from skimage import io
from skimage import data
import cv2  # only for save images to jpg

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

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
rawimg = np.clip(rawimg, raw.black_level_per_channel[0], 2 ** 14)  # black level per channel 512 512 512 512
# print(50*'-' + '\nLoading RAW Image Done......')
plt.imshow(rawimg, cmap='gray')
plt.show()

#################################################################################################################
#####################################  Part 1: Bayer Domain Processing Steps  ###################################
#################################################################################################################
t_start = time.time()
# Step 1. Dead Pixel Correction (10pts)
# dpc = deadPixelCorrection(rawimg, dpc_thres, dpc_mode, dpc_clip)
# Bayer_dpc = dpc.execute()
# print(50 * '-' + '\n 1.1 Dead Pixel Correction Done......')
# t_dpc = time.time()
# print(50 * '-' + '\n 1.1 time used{} ', t_dpc - t_start)
# np.save('Bayer_dpc.npy', Bayer_dpc)
t_dpc = time.time()
Bayer_dpc_load = np.load('Bayer_dpc.npy')
Bayer_dpc = Bayer_dpc_load
print(50 * '-' + '\n 1.1 time used{} ', t_dpc - t_start)
# Step 2.'Black Level Compensation' (5pts)
parameter = raw.black_level_per_channel + [alpha, beta]
blkC = blackLevelCompensation(Bayer_dpc, parameter, bayer_pattern, clip=2 ** 14)  # 16384
Bayer_blackLevelCompensation = blkC.execute()
print(50 * '-' + '\n 1.2 Black Level Compensation Done......')
t_blkc = time.time()
print(50 * '-' + '\n 1.2 time used {}', t_blkc - t_dpc)
plt.imshow(Bayer_blackLevelCompensation, cmap='gray')
plt.show()

# Step 4. Anti Aliasing Filter (10pts)
# time_antiAlias = time.time()
# antiAliasingFilter = antiAliasingFilter(Bayer_blackLevelCompensation)
# Bayer_antiAliasingFilter = antiAliasingFilter.execute()
# time_antiAlias_end = time.time()
# print(50 * '-' + '\n 1.4 Anti-aliasing Filtering Done......')
# print(50 * '-' + '\n 1.2 time used {}', time_antiAlias_end - time_antiAlias)
# plt.imshow(Bayer_antiAliasingFilter, cmap='gray')
# plt.show()

# np.save('Bayer_antiAliasingFilter.npy', Bayer_antiAliasingFilter)
Bayer_antiAliasingFilter = np.load('Bayer_antiAliasingFilter.npy')
plt.imshow(Bayer_antiAliasingFilter, cmap='gray')
plt.show()

# Step 5. Auto White Balance and Gain Control (10pts)
# parameter = [r_gain, gr_gain, gb_gain, b_gain]
# awb = AWB(Bayer_blackLevelCompensation, parameter, bayer_pattern, awb_clip)
# Bayer_awb = awb.execute()
# print(50 * '-' + '\n 1.5 White Balance Gain Done......')
# plt.imshow(Bayer_awb, cmap='gray')
# plt.show()

#np.save('Bayer_awb.npy', Bayer_awb)
Bayer_awb = np.load('Bayer_awb.npy')
plt.imshow(Bayer_awb, cmap='gray')
plt.show()



# Step 6. Chroma Noise Filtering (Extra 20pts)
cnf = ChromaNoiseFiltering(Bayer_awb, bayer_pattern, 0, parameter, cfa_clip)
Bayer_cnf = cnf.execute()
print(50*'-' + '\n 1.6 Chroma Noise Filtering Done......')

# Step 7. 'Color Filter Array Interpolation'  Malvar (20pts)
#cfa = CFA_Interpolation(Bayer_cnf, cfa_mode, bayer_pattern, cfa_clip)
cfa = CFA_Interpolation(Bayer_awb, cfa_mode, bayer_pattern, cfa_clip)
rgbimg_cfa = cfa.execute()
print(50*'-' + '\n 1.7 Demosaicing Done......')

plt.imshow(rgbimg_cfa/2**14)
plt.show()
plt.imshow(np.sqrt(rgbimg_cfa/2**14))
plt.show()




# lena = io.imread('lena.jpg')
# plt.imshow(lena)
# plt.show()
# lena = rgbimg_cfa/2**14
# yuv = RGB2YUV(lena)
# #yuv = RGB2YUV(data.coffee())
# # yuv =np.around(yuv).astype(np.uint8)
# rgb = YUV2RGB(yuv)
# rgb = np.around(rgb).astype(np.uint8)
# plt.subplot(2, 2, 1)
# plt.imshow(lena)
# plt.subplot(2, 2, 2)
# plt.imshow(yuv[:, :, 0], cmap='gray')
# plt.subplot(2, 2, 3)
# plt.imshow(yuv[:, :, 1], cmap='gray')
# plt.subplot(2, 2, 4)
# plt.imshow(yuv[:, :, 2], cmap='gray')
# plt.show()

rgb = rgbimg_cfa/(2**14)*255
yuv = RGB2YUV(rgb)
rgb = YUV2RGB(yuv)
plt.imshow(rgb)
plt.show()

yuvBrighter = BrightnessContrastControl(yuv, 100, 1, 127)
yuvBrighterImg = yuvBrighter.execute()
rgb = YUV2RGB(yuvBrighterImg)
rgb = np.clip(np.around(rgb), 0, 255).astype(np.uint8)
plt.imshow(rgb)
plt.show()
#
hsControl = HueSaturationControl(yuv, 100, 100, 255)
saturationImage = hsControl.execute()
rgb = YUV2RGB(saturationImage)
rgb = np.clip(np.around(rgb), 0, 255).astype(np.uint8)
plt.imshow(rgb)
plt.show()
#
# colorLut = hsControl.lut()
# sinLut,cosLut = hsControl.lut()
# plt.plot(colorLut[0].keys(), colorLut[0].values())
# plt.plot(colorLut[1].keys(), colorLut[1].values())
# plt.legend(['sin', 'cos'], loc='best')
# plt.show()

#
# yuvBrighter = BrightnessContrastControl(yuv, 0, 0.4, 127)
# yuvBrighterImg = yuvBrighter.execute()
# rgb = YUV2RGB(yuvBrighterImg)
# rgb = np.clip(np.around(rgb), 0, 255).astype(np.uint8)
# plt.imshow(rgb)
# plt.show()
#
# yuvBrighter = BrightnessContrastControl(yuv, 0, 1, 127)
# yuvBrighterImg = yuvBrighter.execute()
# rgb = YUV2RGB(yuvBrighterImg)
# rgb = np.clip(np.around(rgb), 0, 255).astype(np.uint8)
# plt.imshow(rgb)
# plt.show()
#
# yuvBrighter = BrightnessContrastControl(yuv, 0, 1.9, 127)
# yuvBrighterImg = yuvBrighter.execute()
# rgb = YUV2RGB(yuvBrighterImg)
# rgb = np.clip(np.around(rgb), 0, 255).astype(np.uint8)
# plt.imshow(rgb)
# plt.show()
# # Step 2.'Black Level Compensation' (5pts)
# parameter = raw.black_level_per_channel + [alpha, beta]
# blkC = blackLevelCompensation(Bayer_dpc, parameter, bayer_pattern, clip = 2**14)
# Bayer_blackLevelCompensation = blkC.execute()
# print(50*'-' + '\n 1.2 Black Level Compensation Done......')
#
# # Step 3.'lens shading correction
# # skip this
#
# # Step 4. Anti Aliasing Filter (10pts)
# antiAliasingFilter = antiAliasingFilter(Bayer_blackLevelCompensation)
# Bayer_antiAliasingFilter = antiAliasingFilter.execute()
# print(50*'-' + '\n 1.4 Anti-aliasing Filtering Done......')
#
#
# # Step 5. Auto White Balance and Gain Control (10pts)
# parameter = [r_gain, gr_gain, gb_gain, b_gain]
# awb = AWB(Bayer_blackLevelCompensation, parameter, bayer_pattern, awb_clip)
# Bayer_awb = awb.execute()
# print(50*'-' + '\n 1.5 White Balance Gain Done......')
# plt.imshow(Bayer_awb)
# plt.show()
#
# # Step 6. Chroma Noise Filtering (Extra 20pts)
# cnf = ChromaNoiseFiltering(Bayer_awb, bayer_pattern, 0, parameter, cfa_clip)
# Bayer_cnf = cnf.execute()
# print(50*'-' + '\n 1.6 Chroma Noise Filtering Done......')
#
# # Step 7. 'Color Filter Array Interpolation'  Malvar (20pts)
# cfa = CFA_Interpolation(Bayer_cnf, cfa_mode, bayer_pattern, cfa_clip)
# rgbimg_cfa = cfa.execute()
# print(50*'-' + '\n 1.7 Demosaicing Done......')
#
# #####################################  Bayer Domain Processing end    ###########################################
# #################################################################################################################
#
# # Convert RGB to YUV (5pts)
# YUV = RGB2YUV(rgbimg_cfa)
#
# #################################################################################################################
# #####################################    Part 2: YUV Domain Processing Steps  ###################################
# #################################################################################################################
#
# # Step Luma-2  Edge Enhancement  for Luma (20pts)
# ee = EdgeEnhancement(YUV[:,:,0], edge_filter, ee_gain, ee_thres, ee_emclip)
# yuvimg_ee = ee.execute()
# print(50*'-' + '\n 3.Luma.2  Edge Enhancement Done......')
#
#
# # Step Luma-3 Brightness/Contrast Control (5pts)
# contrast = contrast / pow(2,5)    #[-32,128]
# bcc = BrightnessContrastControl(yuvimg_ee, brightness, contrast, bcc_clip)
# yuvimg_bcc = bcc.execute()
# print(50*'-' + '\nBrightness/Contrast Adjustment Done......')
#
# # Step Chroma-1 False Color Suppresion (10pts)
# fcs = FalseColorSuppression(YUV[:,:,1:3], yuvimg_edgemap, fcs_edge, fcs_gain, fcs_intercept, fcs_slope)
# yuvimg_fcs = fcs.execute()
# print(50*'-' + '\n 3.Chroma.1 False Color Suppresion Done......')
#
# # Step Chroma-2 Hue/Saturation control (10pts)
# hsc = HueSaturationControl(yuvimg_fcs, hue, saturation, hsc_clip)
# yuvimg_hsc = hsc.execute()
# print(50*'-' + '\n 3.Chroma.2  Hue/Saturation Adjustment Done......')
#
#
# # Concate Y UV Channels
# yuvimg_out = np.zeros_like(YUV)
# yuvimg_out[:,:,0] = yuvimg_bcc
# yuvimg_out[:,:,1:3] = YUV[:,:,1:3]
#
# RGB = YUV2RGB(yuvimg_out) # Pay attention to the bits
#
# #####################################   End YUV Domain Processing Steps  ###################################
# #################################################################################################################
# print('ISP time cost : %.9f sec' %(time.time()-t_start))
#
#
# cv2.imwrite() # save as 8-bit JPG image as "results.jpg"
