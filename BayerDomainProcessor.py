# -*- coding: utf-8 -*-
"""
Created on Fri Aug 27 19:45:23 2021

@author: qilin sun
"""
import numpy as np
from scipy import signal
from scipy import interpolate


# For this file, use numpy only.  Try your best to get the best visual pleasent
# result, as well as the fasest speed and smallest memory consumption.

# Step 1. Dead Pixel Correction (10pts)
class deadPixelCorrection:

    def __init__(self, img, thres, mode, clip):
        self.img = img
        self.thres = thres
        self.mode = mode
        self.clip = clip

    def padding(self):
        # padding needed for avoid black boundry
        # Fill your code here
        # TODO padding  ignore first
        return

    def clipping(self):
        # clip needed for avoid values>maximum
        # Fill your code here
        self.img = np.clip(self.img, 0, self.clip)
        return

    def get_p1(self, i, j):
        if self.is_in_boundary(i, j):
            return self.img[i - 2, j - 2]
        else:
            return self.get_outside()

    def get_p2(self, i, j):
        if self.is_in_boundary(i, j):
            return self.img[i - 2, j]
        else:
            return self.get_outside()

    def get_p3(self, i, j):
        if self.is_in_boundary(i, j):
            return self.img[i - 2, j + 2]
        else:
            return self.get_outside()

    def get_p4(self, i, j):
        if self.is_in_boundary(i, j):
            return self.img[i, j - 2]
        else:
            return self.get_outside()

    def get_p5(self, i, j):
        if self.is_in_boundary(i, j):
            return self.img[i, j + 2]
        else:
            return self.get_outside()

    def get_p6(self, i, j):
        if self.is_in_boundary(i, j):
            return self.img[i + 2, j - 2]
        else:
            return self.get_outside()

    def get_p7(self, i, j):
        if self.is_in_boundary(i, j):
            return self.img[i + 2, j]
        else:
            return self.get_outside()

    def get_p8(self, i, j):
        if self.is_in_boundary(i, j):
            return self.img[i + 2, j + 2]
        else:
            return self.get_outside()

    def is_in_boundary(self, i, j):
        if i - 2 < 0 or i + 2 > self.img.shape[0] - 1 or j - 2 < 0 or j + 2 > self.img.shape[1] - 1:
            return False
        else:
            return True

    def get_outside(self):
        OUTSIDE_VALUE = 0
        return OUTSIDE_VALUE

    def dead_pixel_correction(self, p0, p1, p2, p3, p4, p5, p6, p7, p8):
        dv = np.abs(2 * p0 - p2 - p7)
        dh = np.abs(2 * p0 - p4 - p5)
        ddr = np.abs(2 * p0 - p1 - p8)
        ddl = np.abs(2 * p0 - p3 - p6)
        pixelValue = p0
        index = np.argmin([dv, dh, ddr, ddl])
        if index == 0:
            pixelValue = (p2 + p7 + 1) / 2
        elif index == 1:
            pixelValue = (p4 + p5 + 1) / 2
        elif index == 2:
            pixelValue = (p1 + p8 + 1) / 2
        elif index == 3:
            pixelValue = (p3 + p6 + 1) / 2
        return pixelValue

    def execute(self):
        # Fill your code here
        self.clipping()
        dpc_img = self.img
        # TODO color R
        HEIGHT = self.img.shape[0]
        WIDTH = self.img.shape[1]
        count = 0
        for i in range(0, HEIGHT):
            for j in range(0, WIDTH):
                p0 = self.img[i, j]
                p1 = self.get_p1(i, j)
                p2 = self.get_p2(i, j)
                p3 = self.get_p3(i, j)
                p4 = self.get_p4(i, j)
                p5 = self.get_p5(i, j)
                p6 = self.get_p6(i, j)
                p7 = self.get_p7(i, j)
                p8 = self.get_p8(i, j)
                if np.all(abs(np.array([p1, p2, p3, p4, p5, p6, p7, p8]) - p0) > self.thres):
                    # print("dead pixel: " + str(i) + " " + str(j))
                    self.img[i, j] = self.dead_pixel_correction(p0, p1, p2, p3, p4, p5, p6, p7, p8)
                    count = count + 1
        print("dead pixel # = " + str(count))
        return dpc_img


# Step 2.'Black Level Compensation'   (10pts)
class blackLevelCompensation:
    def __init__(self, img, parameter, bayer_pattern='rggb', clip=2 ** 14):
        self.img = img
        self.parameter = parameter
        self.bayer_pattern = bayer_pattern
        self.clip = clip

    def clipping(self):
        # clip needed for avoid values>maximum, find a proper value for 14bit raw input
        # Fill your code here
        self.img = np.clip(self.img, 0, self.clip)

        return

    def execute(self):
        bl_r = -self.parameter[0]
        bl_gr = -self.parameter[1]
        bl_gb = -self.parameter[2]
        bl_b = -self.parameter[3]
        alpha = self.parameter[4]
        beta = self.parameter[5]
        # Fill your code here
        HEIGHT = self.img.shape[0]
        WIDTH = self.img.shape[1]
        R = self.img[0::2, 0::2]
        GR = self.img[0::2, 1::2]
        GB = self.img[1::2, 0::2]
        B = self.img[1::2, 1::2]
        R = R + bl_r
        GR = GR + bl_gr + alpha * R
        GB = GB + bl_gb + beta * B
        B = B + bl_b
        tmpImg = np.zeros_like(self.img, dtype=np.int32)

        tmpImg[0::2, 0::2] = R[:, :]
        tmpImg[0::2, 1::2] = GR[:, :]
        tmpImg[1::2, 0::2] = GB[:, :]
        tmpImg[1::2, 1::2] = B[:, :]
        tmpImg = tmpImg.clip(0, self.clip)
        self.img = tmpImg.astype(np.uint16)
        self.clipping()
        return self.img

    # Step 3.'lens shading correction


# Skip this step

# Step 4. Anti Aliasing Filter (10pts)
class antiAliasingFilter:
    'Anti-aliasing Filter'

    def __init__(self, img):
        self.img = img

    def padding(self):
        # padding needed for avoid black boundry
        # Fill your code here

        return
        # Hint: "In bayer domain, the each ,R,G,G,B pixel is skipped by 2."

    def get_p1(self, i, j):
        if self.is_in_boundary(i, j):
            return self.img[i - 2, j - 2]
        else:
            return self.get_outside()

    def get_p2(self, i, j):
        if self.is_in_boundary(i, j):
            return self.img[i - 2, j]
        else:
            return self.get_outside()

    def get_p3(self, i, j):
        if self.is_in_boundary(i, j):
            return self.img[i - 2, j + 2]
        else:
            return self.get_outside()

    def get_p4(self, i, j):
        if self.is_in_boundary(i, j):
            return self.img[i, j - 2]
        else:
            return self.get_outside()

    def get_p5(self, i, j):
        if self.is_in_boundary(i, j):
            return self.img[i, j + 2]
        else:
            return self.get_outside()

    def get_p6(self, i, j):
        if self.is_in_boundary(i, j):
            return self.img[i + 2, j - 2]
        else:
            return self.get_outside()

    def get_p7(self, i, j):
        if self.is_in_boundary(i, j):
            return self.img[i + 2, j]
        else:
            return self.get_outside()

    def get_p8(self, i, j):
        if self.is_in_boundary(i, j):
            return self.img[i + 2, j + 2]
        else:
            return self.get_outside()

    def is_in_boundary(self, i, j):
        if i - 2 < 0 or i + 2 > self.img.shape[0] - 1 or j - 2 < 0 or j + 2 > self.img.shape[1] - 1:
            return False
        else:
            return True

    def get_outside(self):
        OUTSIDE_VALUE = 0
        return OUTSIDE_VALUE

    def execute(self):
        # Fill your code here
        HEIGHT = self.img.shape[0]
        WIDTH = self.img.shape[1]
        count = 0
        for i in range(0, HEIGHT):
            for j in range(0, WIDTH):
                p0 = self.img[i, j]
                p1 = self.get_p1(i, j)
                p2 = self.get_p2(i, j)
                p3 = self.get_p3(i, j)
                p4 = self.get_p4(i, j)
                p5 = self.get_p5(i, j)
                p6 = self.get_p6(i, j)
                p7 = self.get_p7(i, j)
                p8 = self.get_p8(i, j)
                self.img[i, j] = (2 * p0 + p1 + p2 + p3 + p4 + p5 + p6 + p7 + p8) / 10
        return self.img


# Step 5. Auto White Balance and Gain Control (10pts)
class AWB:
    def __init__(self, img, parameter, bayer_pattern, clip):
        self.img = img
        self.parameter = parameter
        self.bayer_pattern = bayer_pattern
        self.clip = clip

    def clipping(self):
        # clip needed for avoid values>maximum, find a proper value for 14bit raw input
        # Fill your code here
        MAXIMUM_VALUE = self.clip
        self.img[self.img > MAXIMUM_VALUE] = MAXIMUM_VALUE
        return self.img

    def execute_gaincontrol(self):
        R_gain, Gr_gain, Gb_gain, B_gain = self.parameter
        img = self.img.copy()
        R = img[0::2, 0::2]
        GR = img[0::2, 1::2]
        GB = img[1::2, 0::2]
        B = img[1::2, 1::2]
        img[0::2, 0::2] = R * R_gain
        img[0::2, 1::2] = GR * Gr_gain
        img[1::2, 0::2] = GB * Gb_gain
        img[1::2, 1::2] = B * B_gain
        img = np.clip(img, 0, 1)
        return img

    def execute(self):
        # calculate Gr_avg/R_avg, 1, Gr_avg/Gb_avg, Gr_avg/B_avg and apply to each channel
        # Fill your code here
        self.clipping()
        R = self.img[0::2, 0::2]
        GR = self.img[0::2, 1::2]
        GB = self.img[1::2, 0::2]
        B = self.img[1::2, 1::2]
        R_avg = np.mean(R)
        Gr_avg = np.mean(GR)
        GB_avg = np.mean(GB)
        B_avg = np.mean(GB)
        R_gain = Gr_avg / R_avg
        Gr_gain = 1
        Gb_gain = Gr_avg / GB_avg
        B_gain = Gr_avg / B_avg
        self.img[0::2, 0::2] = R * R_gain
        self.img[0::2, 1::2] = GR * Gr_gain
        self.img[1::2, 0::2] = GB * Gb_gain
        self.img[1::2, 1::2] = B * B_gain
        self.img = self.clipping()
        return self.img

    # Step 6. Chroma Noise Reduction (Additional 20pts)


# Ref: https://patentimages.storage.googleapis.com/a8/b7/82/ef9d61314d91f6/US20120237124A1.pdf

class ChromaNoiseFiltering:
    def __init__(self, img, bayer_pattern, thres, gain, clip):
        self.img = img
        self.bayer_pattern = bayer_pattern
        self.thres = thres
        self.gain = gain
        self.clip = clip

    def padding(self):
        # Fill your code here

        return

    def clipping(self):
        # Fill your code here

        return

    def cnc(self, is_color, center, avgG, avgC1, avgC2):
        'Chroma Noise Correction'
        r_gain, gr_gain, gb_gain, b_gain = self.gain[0], self.gain[1], self.gain[2], self.gain[3]
        dampFactor = 1.0
        signalGap = center - max(avgG, avgC2)
        # Fill your code here

        return

    def cnd(self, y, x, img):
        'Chroma Noise Detection'
        avgG = 0
        avgC1 = 0
        avgC2 = 0
        is_noise = 0
        # Fill your code here

        return is_noise, avgG, avgC1, avgC2

    def cnf(self, is_color, y, x, img):
        is_noise, avgG, avgC1, avgC2 = self.cnd(y, x, img)
        # Fill your code here

        return

    def execute(self):
        # Fill your code here

        return self.img

    # Step 7. 'Color Filter Array Interpolation'  with Malvar Algorithm ”High Quality Linear“ (20pts)


class CFA_Interpolation:
    def __init__(self, img, mode, bayer_pattern, clip):
        self.img = img
        self.mode = mode
        self.bayer_pattern = bayer_pattern
        self.clip = clip
        self.img = np.asarray(self.img, dtype=np.float64)
        self.img_pad = self.padding()

    def padding(self):
        # Fill your code here
        img_pad = np.pad(self.img, ((2, 2), (2, 2)), 'reflect')
        return img_pad

    def clipping(self):
        # Fill your code here

        return

    # def malvar(self, is_color, center, y, x, img):
    # Fill your code here

    # return [r, g, b]

    # bilinear interpolation
    def execute_bilinear(self):
        img = self.img.copy()
        HEIGHT = img.shape[0]
        WIDTH = img.shape[1]
        R = np.zeros_like(img)
        Gr = np.zeros_like(img)
        Gb = np.zeros_like(img)
        B = np.zeros_like(img)
        R[0::2, 0::2] = img[0::2, 0::2]
        Gr[0::2, 1::2] = img[0::2, 1::2]
        Gb[1::2, 0::2] = img[1::2, 0::2]
        B[1::2, 1::2] = img[1::2, 1::2]
        X_R = np.arange(0, HEIGHT, 2)
        Y_R = np.arange(0, WIDTH, 2)
        Z_R = R[0::2, 0::2]
        f_R = interpolate.interp2d(Y_R, X_R, Z_R, kind='linear')  # Y for cols X for rows
        X_B = np.arange(1, HEIGHT, 2)
        Y_B = np.arange(1, WIDTH, 2)
        Z_B = B[1::2, 1::2]
        f_B = interpolate.interp2d(Y_B, X_B, Z_B, kind='linear')
        X_Gr = np.arange(0, HEIGHT, 2)
        Y_Gr = np.arange(1, WIDTH, 2)
        Z_Gr = Gr[0::2, 1::2]
        f_Gr = interpolate.interp2d(Y_Gr, X_Gr, Z_Gr, kind='linear')
        X_Gb = np.arange(1, HEIGHT, 2)
        Y_Gb = np.arange(0, WIDTH, 2)
        Z_Gb = Gb[1::2, 0::2]
        f_Gb = interpolate.interp2d(Y_Gb, X_Gb, Z_Gb, kind='linear')
        rows = np.arange(0, HEIGHT, 1)
        cols = np.arange(0, WIDTH, 1)
        R_new = f_R(cols, rows)
        B_new = f_B(cols, rows)
        Gr_new = f_Gr(cols, rows)
        Gb_new = f_Gb(cols, rows)
        R = R_new
        G = (Gr_new + Gb_new) / 2
        B = B_new
        img = np.dstack((R, G, B))
        return img

    def execute(self):
        img_pad = self.padding()
        img_pad = img_pad.astype(np.int64)
        raw_h = self.img.shape[0]
        raw_w = self.img.shape[1]
        cfa_img = np.empty((raw_h, raw_w, 3), np.float64)
        # Fill your code here
        G00_filter = np.array([
            [0, 0, -1, 0, 0],
            [0, 0, 2, 0, 0],
            [-1, 2, 4, 2, -1],
            [0, 0, 2, 0, 0],
            [0, 0, -1, 0, 0]])
        G11_filter = G00_filter
        R01_filter = np.array([
            [0, 0, 1 / 2, 0, 0],
            [0, -1, 0, -1, 0],
            [-1, 4, 5, 4, -1],
            [0, -1, 0, -1, 0],
            [0, 0, 1 / 2, 0, 0]])
        R10_filter = np.array([
            [0, 0, -1, 0, 0],
            [0, -1, 4, -1, 0],
            [1 / 2, 0, 5, 0, 1 / 2],
            [0, -1, 4, -1, 0],
            [0, 0, -1, 0, 0]])

        R11_filter = np.array([
            [0, 0, -3 / 2, 0, 0],
            [0, 2, 0, 2, 0],
            [-3 / 2, 0, 6, 0, -3 / 2],
            [0, 2, 0, 2, 0],
            [0, 0, -3 / 2, 0, 0]])
        B10_filter = R01_filter
        B01_filter = R10_filter
        B00_filter = R11_filter
        R01 = signal.convolve2d(img_pad, R01_filter, 'valid')
        R10 = signal.convolve2d(img_pad, R10_filter, 'valid')
        R11 = signal.convolve2d(img_pad, R11_filter, 'valid')
        G00 = signal.convolve2d(img_pad, G00_filter, 'valid')
        G11 = signal.convolve2d(img_pad, G11_filter, 'valid')
        B00 = signal.convolve2d(img_pad, B00_filter, 'valid')
        B01 = signal.convolve2d(img_pad, B01_filter, 'valid')
        B10 = signal.convolve2d(img_pad, B10_filter, 'valid')
        R00_mask = np.zeros_like(self.img)
        R01_mask = np.zeros_like(self.img)
        R10_mask = np.zeros_like(self.img)
        R11_mask = np.zeros_like(self.img)
        G00_mask = np.zeros_like(self.img)
        G01_mask = np.zeros_like(self.img)
        G10_mask = np.zeros_like(self.img)
        G11_mask = np.zeros_like(self.img)
        B00_mask = np.zeros_like(self.img)
        B01_mask = np.zeros_like(self.img)
        B10_mask = np.zeros_like(self.img)
        B11_mask = np.zeros_like(self.img)
        R00_mask[0::2, 0::2] = 1
        R01_mask[0::2, 1::2] = 1
        R10_mask[1::2, 0::2] = 1
        R11_mask[1::2, 1::2] = 1
        G00_mask[0::2, 0::2] = 1
        G01_mask[0::2, 1::2] = 1
        G10_mask[1::2, 0::2] = 1
        G11_mask[1::2, 1::2] = 1
        B00_mask[0::2, 0::2] = 1
        B01_mask[0::2, 1::2] = 1
        B10_mask[1::2, 0::2] = 1
        B11_mask[1::2, 1::2] = 1
        R_channel = np.zeros_like(self.img, dtype=np.float64)
        G_channel = np.zeros_like(self.img, dtype=np.float64)
        B_channel = np.zeros_like(self.img, dtype=np.float64)
        R_channel = self.img * R00_mask + R01 * R01_mask + R10 * R10_mask + R11 * R11_mask
        G_channel = self.img * (G01_mask + G10_mask) + G00 * G00_mask + G11 * G11_mask
        B_channel = self.img * B11_mask + B00 * B00_mask + B01 * B01_mask + B10 * B10_mask
        cfa_img[:, :, 0] = R_channel
        cfa_img[:, :, 1] = G_channel
        cfa_img[:, :, 2] = B_channel
        cfa_img = np.clip(cfa_img, 0, self.clip)
        return cfa_img
