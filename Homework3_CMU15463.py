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

museum_ambient = io.imread('data/museum/museum_ambient.png')
museum_flash = io.imread('data/museum/museum_flash.png')
museum_ambient = museum_ambient[:, :, :-1]
museum_flash = museum_flash[:, :, :-1]
plt.figure()
plt.subplot(1, 2, 1)
plt.title('ambient')
plt.imshow(museum_ambient)
plt.subplot(1, 2, 2)
plt.imshow(museum_flash)
plt.title('flash')
plt.show()
N = 2
image_ambient_01 = museum_ambient / 255
image_ambient_01 = image_ambient_01[::N, ::N, :]
image_flash_01 = museum_flash / 255.0
image_flash_01 = image_flash_01[::N, ::N, :]
plt.figure()
plt.subplot(1, 2, 1)
plt.imshow(image_ambient_01)
plt.title(f'ambient image down sampling {N}')
plt.subplot(1, 2, 2)
plt.imshow(image_flash_01)
plt.title(f'flash image down sampling {N}')
plt.show()

# def bilateral2d_piecewise_linear(image_noise_01, range_sigma=0.4, space_sigma=0.02, kernel_size=

range_sigma = 0.1
space_sigma = 0.02
I = image_ambient_01.copy()
HEIGHT = I.shape[0]
WIDTH = I.shape[1]
lamda = range_sigma / 10
I_max = np.max(I)
I_min = np.min(I)
NB_SEGMENTS = np.ceil((I_max - I_min) / range_sigma)
NB_SEGMENTS = NB_SEGMENTS.astype(np.int32)
intensity_delta = (I_max - I_min) / NB_SEGMENTS
intensity = np.zeros((NB_SEGMENTS, 1))

J_list = []
J = np.zeros_like(I)
j = 0
for j in range(0, NB_SEGMENTS):
    intensity_j = I_min + j * intensity_delta
    G_j = g_sigma_r(I - intensity_j, range_sigma)
    space_sigma_image_size = space_sigma*np.min(I.shape[:-1])
    f_sigma_s = gaussian_kernel_for_piecewise_bilateral(space_sigma_image_size)
    K_j = Convolution2D(G_j, f_sigma_s, mode='same', boundary='fill', fillvalue=0)
    H_j = G_j * I
    H_star_j = Convolution2D(H_j,f_sigma_s, mode='same', boundary='fill', fillvalue=0)
    J_j = H_star_j/K_j
    J_list.append(J_j)
    J = J + J_j *get_hat_weights(I -intensity_j)

#I_intern = interpolate.interpn(J_list,I)


plt.imshow(K_j)
plt.show()

plt.imshow(G_j)
plt.show()

plt.imshow(J/np.max(J))
plt.show()
plt.imshow(np.clip(J,0,1))
plt.show()
####
#### part 2 gradient domain processing
####
# Itest = np.arange(1, 21).reshape(4, 5)
# #Itest = np.arange(1, 100).reshape(9, 11)
# Itest = np.stack((Itest, Itest, Itest), axis = 2)


# # I = Itest
# I = image_ambient_01.astype(np.int32)
# B = get_image_boundary(I)
# I_init_star = np.zeros_like(B)
# I_boundary_star = (1 - B) * I
# I_star = B * I_init_star + (1 - B) * I_boundary_star
# I_star_lap_fil = Laplacian(I)
# plt.figure()
# plt.subplot(1, 3, 1)
# plt.imshow(B)
# plt.title('B mask')
# plt.axis('off')
# plt.subplot(1, 3, 2)
# plt.imshow(np.abs(I_star_lap_fil))
# plt.title('laplacian filterig I ')
# plt.subplot(1, 3, 3)
# plt.imshow(I_star)
# plt.title('I*')
# plt.show()
#
# Ix_left, Iy_left = Gradient_Left(I)
# Ixx_right, Iyx_right = Gradient_Right(Ix_left)
# Ixy_right, Iyy_right = Gradient_Right(Iy_left)
# I_div = Divergence(I)
# Ilap = Laplacian(I)
# plt.figure()
# plt.subplot(1, 2, 1)
# plt.imshow(np.abs(Ix_left))
# plt.title('Ix')
# plt.subplot(1, 2, 2)
# plt.imshow(np.abs(Iy_left))
# plt.title('Iy')
# plt.show()
#
# plt.figure()
# plt.subplot(1, 2, 1)
# plt.imshow(np.abs(I_div))
# plt.title('divergence I ')
# plt.subplot(1, 2, 2)
# plt.imshow(np.abs(I_div))
# plt.title('laplacian I ')
# plt.show()
#
# ambient_x, ambient_y = Gradient_Left(image_ambient_01)
# flash_x, flash_y = Gradient_Left(image_flash_01)
# M = gradient_orientation_coherency_map(image_flash_01, image_ambient_01)
# weight_satuation_map = satuation_weight_map(image_flash_01)
# ws = weight_satuation_map
# ax = ambient_y
# ay = ambient_y
# phi_prime_x = flash_x
# phi_prime_y = flash_y
# phi_star_x = ws * ax + (1 - ws) * (M * phi_prime_x + (1 - M) * ax)
# phi_star_y = ws * ay + (1 - ws) * (M * phi_prime_y + (1 - M) * ay)
# plt.subplot(1, 4, 1)
# plt.imshow(ambient_x)
# plt.subplot(1, 4, 2)
# plt.imshow(ambient_x)
# plt.subplot(1, 4, 3)
# plt.imshow(M)
# plt.title('gradient coherency map')
# plt.subplot(1, 4, 4)
# plt.imshow(weight_satuation_map)
# plt.title('weight satuation map')
# plt.show()
#
# plt.subplot(1, 2, 1)
# plt.imshow(phi_star_x)
# plt.title(r'$\Phi^*_x$')
# plt.subplot(1, 2, 2)
# plt.imshow(phi_star_y)
# plt.title(r'$\Phi^*_y$')
# plt.show()
#
# # cgd to solve gradietn field integration
#
# I = image_ambient_01
# phi_prime = image_flash_01
# B = get_image_boundary(I)
# I_init_star = np.zeros_like(B)
# I_boundary_star = (1 - B) * I
# I_star = B * I_init_star + (1 - B) * I_boundary_star
# ax, ay = Gradient_Left(image_ambient_01)
# phi_prime_x, phi_prime_y = Gradient_Left(image_flash_01)
# M = gradient_orientation_coherency_map(image_flash_01, image_ambient_01)
# ws = satuation_weight_map(image_flash_01)
# phi_star_x = ws * ax + (1 - ws) * (M * phi_prime_x + (1 - M) * ax)
# phi_star_y = ws * ay + (1 - ws) * (M * phi_prime_y + (1 - M) * ay)
# D = Divergence_Gradient(phi_star_x, phi_star_y)
#
# r = B * (D - Laplacian(I_star))
# d = r
# delta_new = np.sum(r * r, axis=(0, 1))
# N = I.shape[0] * I.shape[1]
# I_star_list = []
# eps = 10 ** -6
# for n in range(0, N):
#     r_norm_square = np.sum(r * r, axis=(0, 1))
#     r_norm = np.sqrt(r_norm_square)
#     if np.any(r_norm < eps):
#         break
#     q = Laplacian(d)
#     d_q_dot = np.sum(d * q, axis=(0, 1))
#     eta = delta_new / d_q_dot
#     I_star = I_star + B * (eta * d)
#     r = B * (r - eta * q)
#     delta_old = delta_new
#     delta_new = np.sum(r * r, axis=(0, 1))
#     beta = delta_new / delta_old
#     d = r + beta * d
#     I_star_list.append(I_star)
#     # plt.imshow(I_star)
#     # plt.title(f'recostruction result {n}')
#     # plt.show()
#     print(f'iteration N : {n}')
#
# for i in range(0, len(I_star_list),10):
#     plt.imshow(np.clip(I_star_list[i], 0, 1))
#     plt.title(f'recostruction {i}')
#     plt.show()
#     time.sleep(1)
#
# plt.subplot(1, 2, 1)
# plt.imshow(np.abs(D))
# plt.title('Divergence ')
# plt.subplot(1, 2, 2)
# plt.imshow(np.abs(I_star))
# plt.title(r'$I^*$')
# plt.show()

# # part 1 bilateral filtering
# lamp_ambient = io.imread('data/lamp/lamp_ambient.tif')  # iso 1600
# lamp_flash = io.imread('data/lamp/lamp_flash.tif')  # iso 200
# plt.figure()
# plt.subplot(1, 2, 1)
# plt.title('ambient')
# plt.imshow(lamp_ambient)
# plt.subplot(1, 2, 2)
# plt.imshow(lamp_flash)
# plt.title('flash')
# plt.show()
#
# N = 2
# image_ambient_01 = lamp_ambient / 255.0
# image_ambient_01 = image_ambient_01[::N, ::N, :]
# image_flash_01 = lamp_flash / 255.0
# image_flash_01 = image_flash_01[::N, ::N, :]
# plt.figure()
# plt.subplot(1, 2, 1)
# plt.imshow(image_ambient_01)
# plt.title('ambient image down sampling 8')
# plt.subplot(1, 2, 2)
# plt.imshow(image_flash_01)
# plt.title('flash image down sampling 8')
# plt.show()
#
# # suggest space sigma  is 0.05 0.1 total image size
# range_sigma = 0.15 # sigma range suggest 0.05 0.25
# space_sigma = 3  # kernel size 4 *sigma + 1
#
# image_ambient_01_base = bilateral2d_color(image_ambient_01, range_sigma, space_sigma)
# image_flash_01_base = bilateral2d_color(image_flash_01, range_sigma, space_sigma)
# plt.figure()
# plt.subplot(1, 2, 1)
# plt.imshow(image_ambient_01_base)
# plt.title('ambient base')
# plt.subplot(1, 2, 2)
# plt.imshow(image_flash_01_base)
# plt.title('flash base')
# plt.show()
#
# A = image_ambient_01
# A_base = image_flash_01_base
# F = image_flash_01
# F_base = image_flash_01_base
# epsilon = 0.02  # papper suggest
# F_detail = (F + epsilon) / (F_base + epsilon)
# plt.figure()
# plt.imshow(F_detail)
# plt.title('F detail')
# plt.show()
#
# A_noise_reduction = bilateral2d_color_joint(A, F, range_sigma, space_sigma)
# plt.figure()
# plt.imshow(A_noise_reduction)
# plt.title('A_NR')
# plt.show()
# # compute A_NR  by joint bilateral filtering
#
#
# # compute A_detail
# A_detail = A_noise_reduction * F_detail
# plt.figure()
# plt.imshow(A_detail)
# plt.title('A_detail')
# plt.show()
#
# # to compute the
# F_linear_ISO200 = np.power(image_flash_01, 2.2)
# A_linear_ISO1600 = np.power(image_ambient_01, 2.2)
# A_linear_ISO200 = A_linear_ISO1600 * 200 / 1600
#
# F_linear = F_linear_ISO200
# A_linear = A_linear_ISO200
# tau_shadow = 0.01
# F_minus_A = F_linear - A_linear
# M_shadow = F_minus_A <= tau_shadow
# M_shadow_01 = M_shadow.astype(np.float64)
#
# plt.figure()
# plt.subplot(1, 3, 1)
# plt.imshow(A_linear)
# plt.title('A linear')
# plt.subplot(1, 3, 2)
# plt.imshow(F_linear)
# plt.title('F linear')
# plt.subplot(1, 3, 3)
# plt.imshow(M_shadow_01)
# plt.title('Mask shadow ')
# plt.show()
#
#
# # put all together
# A_final = (1-M_shadow)*A_noise_reduction*F_detail + M_shadow * A_base
# plt.figure()
# plt.imshow(A_final)
# plt.title('final result')
# plt.show()
#
# plt.figure()
# plt.subplot(1, 3, 1)
# plt.imshow(image_ambient_01)
# plt.title('ambient image down sampling 8')
# plt.subplot(1, 3, 2)
# plt.imshow(image_flash_01)
# plt.title('flash image down sampling 8')
# plt.subplot(1,3,3)
# plt.imshow(A_final)
# plt.title('A final')
# plt.show()
