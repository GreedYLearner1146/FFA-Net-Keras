# Compute PSNR and SSIM.
from math import log10, sqrt
import numpy as np

from skimage.metrics import structural_similarity

# PSNR

def PSNR(clean_image, predicted_image):
    mse = np.mean((clean_image - predicted_image)**2)
    if (mse ==0):  # MSE = 0 => no noise present in image, PSNR has no importance.
        return 100
    max_pixel = 1   # Because we normalized our images.
    psnr = 20*log10(max_pixel/sqrt(mse))
    return psnr

# Fill in your dehazed image and clear image array as below.

PSNR('dehazed image', 'clean image')

#Use compare_ssim built in functions. Insert array of dehazed (predicted) images and clean images.

score = tf.image.ssim('dehazed image', 'clean image', max_val=2.0)
