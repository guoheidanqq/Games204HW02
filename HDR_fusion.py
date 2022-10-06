import rawpy
import numpy as np
import imageio


# step 1 merge ldr ra  into 32 bit hdr image

def raw_exposure_fuse(images, weights, exposure_times):
    fuse_image = np.zeros(images[0].raw_image.shape, dtype='float64')
    masks = compute_mask(images)
    fuse_weights = np.zeros(weights[0].shape)
    for i in range(0, len(images)):
        fuse_image = fuse_image + np.log(1+(images[i].raw_image / exposure_times[i])) * weights[i] * masks[i]
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


def writeEXR(filename):
    pass
    return


# step 3 tone mapping with bilateral filter

def fastbilateral2d(HDR_image, space_sigma=0.02, range_sigma=0.4):
    # image should be in float64
    image = HDR_image.astype(np.float64)
    log_image = np.log2(image + 1)
    space_sigma = 0.02 * np.min(image.shape[:-1])
    range_sigma = 0.4
    HDR_image_log_base = log_image
    return HDR_image_log_base


def compute_new_intensity(HDR_image_log_base, HDR_image_log_detail, gamma):
    pass
    return


# step 4 apply Gamma


def gamma_v709(image):
    # image should be in range [0 1 ]
    new_image = image.copy()
    for i in range(0, new_image.shape[0]):
        for j in range(0, new_image.shape[1]):
            new_image[i, j] = v709(new_image[i, j])
    return new_image


def v709(x):
    if 0 <= x < 0.018:
        return 4 * x
    elif 0.018 <= x < 1:
        return 1.099 * x ** 0.45 - 0.099
