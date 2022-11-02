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

# part 2 gradient domain processing
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
image_ambient_01 = museum_ambient / 255.0
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

I = image_ambient_01
B = get_image_boundary(image_ambient_01)
I_init_star = np.zeros_like(B)
I_boundary_star = (1 - B) * I
I_star = B * I_init_star + (1 - B) * I_boundary_star
I_star_lap_fil = Laplacian_Filtering(I)
plt.figure()
plt.subplot(1, 3, 1)
plt.imshow(B)
plt.title('B mask')
plt.axis('off')
plt.subplot(1, 3, 2)
plt.imshow(np.abs(I_star_lap_fil))
plt.title('laplacian filterig I ')
plt.subplot(1, 3, 3)
plt.imshow(I_star)
plt.title('I*')
plt.show()

Ix, Iy = Gradient(I)
plt.figure()
plt.subplot(1, 2, 1)
plt.imshow(np.abs(Ix))
plt.title('Ix')
plt.subplot(1, 2, 2)
plt.imshow(np.abs(Iy))
plt.title('Iy')
plt.show()

I_div = Divergence(I)
I_laplacian = Laplacian_Filtering(I)
plt.figure()
plt.subplot(1, 2, 1)
plt.imshow(np.abs(I_div))
plt.title('divergence I ')
plt.subplot(1, 2, 2)
plt.imshow(np.abs(I_laplacian))
plt.title('laplacian I ')
plt.show()

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
