import rawpy
import numpy as np
import imageio
import cv2


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


def get_fusion_weight(images):
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
    cv2.imwrite(img_hdr,filename)



# step 3 tone mapping with bilateral filter


# fastbilaeral2d  input image 0 - 1 ,
def fastbilateral2d(HDR_image_gray, range_sigma=0.4, space_sigma=0.02, kernel_size=5):
    # image should be in float64
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
