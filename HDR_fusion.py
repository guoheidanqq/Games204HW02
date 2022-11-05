import rawpy
import numpy as np
import imageio
import cv2
import scipy
from scipy import signal


# step 1 merge ldr ra  into 32 bit hdr image

def raw_exposure_fuse(images, weights, exposure_times):
    fuse_image = np.zeros(images[0].raw_image.shape, dtype='float64')
    masks = compute_mask(images)
    fuse_weights = np.zeros(weights[0].shape)
    for i in range(0, len(images)):
        fuse_image = fuse_image + np.log(1 + (images[i].raw_image / exposure_times[i])) * weights[i] * masks[i]
        fuse_weights = fuse_weights + weights[i] * masks[i]
    eps = 10 ** -10
    fuse_weights[fuse_weights <= 0] = eps
    final_image = fuse_image / fuse_weights
    final_image = np.exp(final_image)
    return final_image


def raw_exposure_fuse_01(images, weights, exposure_times):
    fuse_image = np.zeros_like(images[0])
    masks = compute_mask_01(images)
    fuse_weights = np.zeros(weights[0].shape)
    for i in range(0, len(images)):
        fuse_image = fuse_image + (images[i] / exposure_times[i]) * weights[i] * masks[i]
        fuse_weights = fuse_weights + weights[i] * masks[i]
    eps = 10 ** -10
    fuse_weights[fuse_weights <= 0] = eps
    final_image = fuse_image / fuse_weights
    return final_image


def compute_mask(images):
    MIN = 0.05 * 2 ** 14
    MAX = 0.95 * 2 ** 14
    maskList = []
    for i in range(0, len(images)):
        image = images[i].raw_image
        maski = np.zeros_like(image, dtype=bool)
        maskaa = image <= MAX
        maskbb = image >= MIN
        maski = maskaa & maskbb
        maskList.append(maski)
    return maskList


def compute_mask_01(images):
    # input image should be in [0 1] of float
    MIN = 0.05
    MAX = 0.95
    maskList = []
    for i in range(0, len(images)):
        image = images[i]
        # maski = np.zeros_like(image, dtype=bool)
        maskaa = image <= MAX
        maskbb = image >= MIN
        maski = maskaa & maskbb  # & numpy logcial and
        maskList.append(maski)
    return maskList


def get_fusion_weight(images):
    # input data should be in 0 1
    MAX = 2 ** 14
    weights = []
    for i in range(0, len(images)):
        image = images[i].raw_image.copy()
        image = image.astype('float')
        image = image / MAX
        image = -4 * (image - 0.5) ** 2 / (0.5 ** 2)
        weight = np.exp(image)
        weights.append(weight)
    return weights


def get_uniform_weights(images):
    weights = []
    for i in range(0, len(images)):
        weight = np.ones_like(images[i])
        weights.append(weight)
    return weights


def get_tent_weights(images):
    # image value should be in  [0 1]
    weights = []
    for i in range(0, len(images)):
        z = images[i]
        weight = np.minimum(z, 1 - z)
        weights.append(weight)
    return weights


@np.vectorize
def w_tent(z):
    if 0 <= z <= 1:
        if z <= 1 - z:
            return z
        else:
            return 1 - z
    else:
        return 0


def g_solve(I, T, w, lamda=1):
    # I K row N columns , k images N pixels
    # T each pixels exposure time same dimensions as I
    # lamda the regulization term
    # w
    K = I.shape[0]
    N = I.shape[1]
    Z_levels = 256
    lamda = lamda
    A = np.zeros((K * N + Z_levels - 2, Z_levels + N))
    b = np.zeros((A.shape[0], 1))
    row = 0
    for z in range(0, 254):
        A[row, z] = 1 * w[z + 1] * lamda
        A[row, z + 1] = -2 * w[z + 1] * lamda
        A[row, z + 2] = 1 * w[z + 1] * lamda
        row = row + 1

    row = 254
    for k in range(0, K):
        for i in range(0, N):
            z = I[k, i].astype(np.int32)
            weight = w[z]
            ex_time = T[k, i]
            A[row, z] = weight
            A[row, i + 256] = -weight
            b[row] = weight * np.log(ex_time)
            row = row + 1

    X = np.linalg.lstsq(A, b, rcond=None)
    g = X[0][0:256]
    le = X[0][256:len(X[0])]
    return g, le


def get_gaussian_weights(images):
    # input value should be in  [0 1]
    weights = []
    for i in range(0, len(images)):
        image = images[i].copy()
        weight = np.zeros_like(images[i])
        weight = -4 * (image - 0.5) ** 2 / (0.5 ** 2)
        weight = np.exp(image)
        weights.append(weight)
    return weights


# step 2 demosaic the fused raw data

def CFA_Interpolation_function(fuse_image):
    r_channel = np.zeros(fuse_image.shape, dtype='uint32')
    gr_channel = np.zeros(fuse_image.shape, dtype=np.uint32)
    gb_channel = np.zeros(fuse_image.shape, dtype=np.uint32)
    b_channel = np.zeros(fuse_image.shape, dtype=np.uint32)
    r_channel[0::2, 0::2] = fuse_image[0::2, 0::2]
    gr_channel[0::2, 1::2] = fuse_image[0::2, 1::2]
    gb_channel[1::2, 0::2] = fuse_image[1::2, 0::2]
    b_channel[1::2, 1::2] = fuse_image[1::2, 1::2]
    color_image = np.zeros((*fuse_image.shape, 3), dtype=np.uint32)
    color_image[:, :, 0] = r_channel
    color_image[:, :, 1] = (gr_channel + gb_channel) / 2
    color_image[:, :, 2] = b_channel
    return color_image


def writeEXR(img_hdr, filename):
    cv2.imwrite(img_hdr, filename)


# step 3 tone mapping with bilateral filter


# fastbilaeral2d  input image 0 - 1 ,
def fastbilateral2d(HDR_image_gray, range_sigma=0.4, space_sigma=0.02, kernel_size=5):
    # image should be in float64 channel 1 , for example gray channel
    image = HDR_image_gray.copy()
    HEIGHT = image.shape[0]
    WIDTH = image.shape[1]
    # image = image.astype(np.float32)
    space_sigma = space_sigma * np.min(image.shape[:-1])
    range_sigma = range_sigma
    kernel_size = kernel_size
    kernel_size = np.int32(kernel_size)
    size = np.int32(kernel_size / 2)
    half_kernel_size = np.int32()
    smooth_kernel = gaussian_kernel(kernel_size, space_sigma)
    local_kernel = np.zeros((kernel_size, kernel_size), dtype=np.float64)
    image = np.pad(image, ((size, size), (size, size)), 'reflect')
    newimg = np.zeros_like(image)
    for x in range(0 + size, HEIGHT + size):
        for y in range(0 + size, WIDTH + size):
            local_img = image[x - size:x + size + 1, y - size:y + size + 1]
            local_kernel = value_kernel(kernel_size, range_sigma, local_img)
            kernel = smooth_kernel * local_kernel
            sum = kernel.sum()
            kernel_norm = kernel / sum
            value = kernel_norm * local_img
            newimg[x, y] = value.sum()
    # d = 2 * int(space_sigma) + 1
    # HDR_image_log_base = np.asarray(image)
    # HDR_image_log_base = cv2.bilateralFilter(image, d, range_sigma, space_sigma)
    bi_img = newimg[size:HEIGHT + size, size:WIDTH + size]
    return bi_img


def bilateral2d_color_joint(img_ambient, img_flash, range_sigma=0.4, space_sigma=5):
    # image should be in float64  channel 3 ,for example color image
    # space sigma should be something integer in 1-64,
    # kenel size = 4* space sigma + 1
    # space_sigma = space_sigma * np.min(image.shape[:-1])
    img_ambient = img_ambient.copy()
    img_flash = img_flash.copy()
    HEIGHT = img_ambient.shape[0]
    WIDTH = img_ambient.shape[1]

    range_sigma = range_sigma
    kernel_size = 4 * np.int(space_sigma) + 1
    kernel_size = np.int32(kernel_size)
    size = np.int32(kernel_size / 2)
    smooth_kernel = gaussian_kernel(kernel_size, space_sigma)
    smooth_kernel = np.stack((smooth_kernel, smooth_kernel, smooth_kernel), axis=2)
    local_kernel = np.zeros((kernel_size, kernel_size, 3), dtype=np.float64)
    img_ambient = np.pad(img_ambient, ((size, size), (size, size), (0, 0)), 'reflect')
    img_flash = np.pad(img_flash, ((size, size), (size, size), (0, 0)), 'reflect')
    newimg = np.zeros_like(img_ambient)

    for x in range(0 + size, HEIGHT + size):
        for y in range(0 + size, WIDTH + size):
            # x = 6
            # y = 378
            local_img = img_flash[x - size:x + size + 1, y - size:y + size + 1, :]
            local_kernel = value_kernel_color(kernel_size, range_sigma, local_img)
            kernel = smooth_kernel * local_kernel
            sum = kernel.sum(axis=(0, 1))
            kernel_norm = kernel / sum
            value = kernel_norm * local_img
            newimg[x, y, :] = value.sum(axis=(0, 1))
            # print(f'x{x} y{y}')

    bi_img = newimg[size:HEIGHT + size, size:WIDTH + size, :]
    return bi_img


def bilateral2d_color(HDR_image_gray, range_sigma=0.4, space_sigma=5):
    # image should be in float64  channel 3 ,for example color image
    # space sigma should be something integer in 1-64,
    # kenel size = 4* space sigma + 1
    # space_sigma = space_sigma * np.min(image.shape[:-1])
    image = HDR_image_gray.copy()
    HEIGHT = image.shape[0]
    WIDTH = image.shape[1]

    range_sigma = range_sigma
    kernel_size = 4 * np.int(space_sigma) + 1
    kernel_size = np.int32(kernel_size)
    size = np.int32(kernel_size / 2)
    smooth_kernel = gaussian_kernel(kernel_size, space_sigma)
    smooth_kernel = np.stack((smooth_kernel, smooth_kernel, smooth_kernel), axis=2)
    local_kernel = np.zeros((kernel_size, kernel_size, 3), dtype=np.float64)
    image = np.pad(image, ((size, size), (size, size), (0, 0)), 'reflect')
    newimg = np.zeros_like(image)

    for x in range(0 + size, HEIGHT + size):
        for y in range(0 + size, WIDTH + size):
            local_img = image[x - size:x + size + 1, y - size:y + size + 1, :]
            local_kernel = value_kernel_color(kernel_size, range_sigma, local_img)
            kernel = smooth_kernel * local_kernel
            sum = kernel.sum(axis=(0, 1))
            kernel_norm = kernel / sum
            value = kernel_norm * local_img
            newimg[x, y, :] = value.sum(axis=(0, 1))

    bi_img = newimg[size:HEIGHT + size, size:WIDTH + size, :]
    return bi_img


def value_kernel_color(kernel_size, range_sigma, local_img):
    kernel_size = np.int32(kernel_size)
    kernel_size = np.int32(np.int32(kernel_size / 2) * 2 + 1)
    size = np.int32(kernel_size / 2)
    x = np.arange(-size, size + 1)
    y = np.arange(-size, size + 1)
    [xx, yy] = np.meshgrid(x, y)
    f = local_img
    value = -(f[xx + size, yy + size, :] - f[size, size, :]) ** 2 / (2 * range_sigma ** 2)
    kernel = np.exp(value)
    norm_sum = np.sum(kernel, axis=(0, 1))
    norm_kernel = kernel / norm_sum
    return norm_kernel


def gaussian_kernel(kernel_size, space_sigma):
    # make sure your kernel size if odd
    kernel_size = np.int32(kernel_size)
    kernel_size = np.int32(np.int32(kernel_size / 2) * 2 + 1)
    size = np.int32(kernel_size / 2)
    x = np.arange(-size, size + 1)
    y = np.arange(-size, size + 1)
    [xx, yy] = np.meshgrid(x, y)
    value = -(xx * xx + yy * yy) / (2 * space_sigma ** 2)
    kernel = np.exp(value)
    norm_sum = np.sum(kernel)
    norm_kernel = kernel / norm_sum
    return norm_kernel


def value_kernel(kernel_size, range_sigma, local_img):
    kernel_size = np.int32(kernel_size)
    kernel_size = np.int32(np.int32(kernel_size / 2) * 2 + 1)
    size = np.int32(kernel_size / 2)
    x = np.arange(-size, size + 1)
    y = np.arange(-size, size + 1)
    [xx, yy] = np.meshgrid(x, y)
    f = local_img
    value = -(f[xx + size, yy + size] - f[size, size]) ** 2 / (2 * range_sigma ** 2)
    kernel = np.exp(value)
    norm_sum = np.sum(kernel)
    norm_kernel = kernel / norm_sum
    return norm_kernel


def compute_new_intensity(HDR_image_log_base, HDR_image_log_detail, gamma):
    pass
    return


# step 4 apply Gamma


def gamma_v709(image):
    # image should be in range [0 1 ]
    new_image = image.copy()
    new_image = v709(image)
    return new_image


@np.vectorize
def v709(x):
    if 0 <= x < 0.018:
        return 4.5 * x
    elif 0.018 <= x < 1:
        return 1.099 * x ** 0.45 - 0.099


def get_image_boundary(image):
    B = np.zeros_like(image)
    B[1:-1, 1:-1, :] = 1
    return B


def Gradient_Left(image):
    image_pad_x = np.pad(image, ((1, 0), (0, 0), (0, 0)))
    image_pad_y = np.pad(image, ((0, 0), (1, 0), (0, 0)))
    Ix = np.diff(image_pad_x, axis=0)
    Iy = np.diff(image_pad_y, axis=1)
    return Ix, Iy


def Gradient_Right(image):
    image_pad_x = np.pad(image, ((0, 1), (0, 0), (0, 0)))
    image_pad_y = np.pad(image, ((0, 0), (0, 1), (0, 0)))
    Ix = np.diff(image_pad_x, axis=0)
    Iy = np.diff(image_pad_y, axis=1)
    return Ix, Iy


def Gradient(image):
    Ix_left, Iy_left = Gradient_Left(image)
    Ix_right, Iy_right = Gradient_Right(image)
    Ix = (Ix_left + Ix_right) / 2
    Iy = (Iy_left + Iy_right) / 2
    return Ix, Iy


def Divergence(image):
    image_pad_x = np.pad(image, ((1, 0), (0, 0), (0, 0)))
    image_pad_y = np.pad(image, ((0, 0), (1, 0), (0, 0)))
    Ix = np.diff(image_pad_x, axis=0)
    Iy = np.diff(image_pad_y, axis=1)
    Ix_pad_x = np.pad(Ix, ((0, 1), (0, 0), (0, 0)))
    Iy_pad_y = np.pad(Iy, ((0, 0), (0, 1), (0, 0)))
    Ix_x = np.diff(Ix_pad_x, axis=0)
    Iy_y = np.diff(Iy_pad_y, axis=1)
    div = Ix_x + Iy_y
    return div


def gradient_orientation_coherency_map(ambient_image, flash_image):
    phi = flash_image
    a = ambient_image
    phi_x, phi_y = Gradient_Left(phi)
    a_x, a_y = Gradient_Left(a)
    eps = 10 ** -6
    phi_norm = np.sqrt(phi_x ** 2 + phi_y ** 2)
    a_norm = np.sqrt(a_x ** 2 + phi_y ** 2)
    denominator = phi_norm * a_norm + eps
    numerator = np.abs(phi_x * a_x + phi_y * a_y)
    map = numerator / denominator
    return np.clip(map, 0, 1)


def satuation_weight_map(flash_image):
    tau_s = 0.9
    sigma = 40
    weight_map = np.tanh(sigma * (flash_image - 0.9))
    return weight_map


def Laplacian(image):
    R = image[:, :, 0]
    G = image[:, :, 1]
    B = image[:, :, 2]
    lap_filter = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]])
    R_lap_fil = signal.convolve2d(R, lap_filter, mode='same', boundary='fill', fillvalue='0')
    G_lap_fil = signal.convolve2d(G, lap_filter, mode='same', boundary='fill', fillvalue=0)
    B_lap_fil = signal.convolve2d(B, lap_filter, mode='same', boundary='fill', fillvalue=0)
    image_lap_fil = np.stack((R_lap_fil, G_lap_fil, B_lap_fil), axis=2)
    return image_lap_fil
