# -*- coding: utf-8 -*-
"""
Created on Fri Aug 27 19:45:23 2021

@author: qilin sun
"""
import numpy as np


# Step 1. 'Color Correction Matrix'
class ColorCorrectionMatrix:
    def __init__(self, img, ccm):
        self.img = img
        self.ccm = ccm

    def execute(self):
        img_h = self.img.shape[0]
        img_w = self.img.shape[1]
        img_c = self.img.shape[2]
        ccm_img = np.empty((img_h, img_w, img_c), np.uint32)
        for y in range(img_h):
            for x in range(img_w):
                mulval = self.ccm[0:3, :] * self.img[y, x, :]
                ccm_img[y, x, 0] = np.sum(mulval[0]) + self.ccm[3, 0]
                ccm_img[y, x, 1] = np.sum(mulval[1]) + self.ccm[3, 1]
                ccm_img[y, x, 2] = np.sum(mulval[2]) + self.ccm[3, 2]
                ccm_img[y, x, :] = ccm_img[y, x, :]
        self.img = ccm_img
        return self.img


# Step 2. 'Gamma Correction'
class GammaCorrection:
    def __init__(self, img, lut, mode):
        self.img = img
        self.lut = lut
        # Here you can choose to cread a look up table(LUT) saved from Photoshop, you will get a better results and higher grades.
        # Or simply apply a gamma if LUT is too hard for you.
        self.mode = mode

    def execute(self):
        return

    # Step 3 . Color Space Conversion   RGB-->YUV


def RGB2YUV(img):
    # rgb2yuv = [0.299 0.587 0.114
    #            - 0.168736 - 0.331264 0.5
    #            0.5 - 0.418688 - 0.081312]   +[0 128 128]
    # JPEG 格式的YCbCr
    # img 输入值的范围为0-255
    img = img.copy()
    R = img[:, :, 0]
    G = img[:, :, 1]
    B = img[:, :, 2]
    # Your code here
    Y = 0.299 * R + 0.587 * G + 0.114 * B
    U = -0.168736 * R - 0.331264 * G + 0.5 * B + 128
    V = 0.5 * R - 0.418668 * G - 0.081312 * B + 128
    yuv = np.stack([Y, U, V], axis=2)
    # yuv = yuv.astype(np.uint8)
    return yuv


def YUV2RGB(img):
    # yuv2rgb = [1.0000 - 0.0000    1.4020
    #            1.0000 - 0.3441 - 0.7141
    #            1.0000    1.7720    0.0000]
    img = img.copy()
    Y = img[:, :, 0]
    U = img[:, :, 1] - 128
    V = img[:, :, 2] - 128
    # Your code here
    R = Y + 1.402 * V
    G = Y - 0.3441 * U - 0.7141 * V
    B = Y + 1.772 * U
    rgb = np.stack([R, G, B], axis=2)
    rgb = np.clip(rgb, 0, 255)
    rgb = rgb.astype(np.uint8)
    return rgb


@np.vectorize
def srgb_non_linear_trans(x):
    if x <= 0.0031308:
        return 12.92 * x
    elif x > 0.0031308:
        return (1 + 0.055) * np.power(x, 1 / 2.4) - 0.055
