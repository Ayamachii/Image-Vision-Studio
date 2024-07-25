import numpy as np
import cv2
import matplotlib.pyplot as plt

import numpy as np



def  global_spectral_thresholding(img):
    image = img.copy()
    # Check if input image is grayscale
    if len(img.shape) != 2:
        raise ValueError("Input image must be grayscale")

    # Compute histogram and cdf
    hist, bins = np.histogram(img.flatten(), bins=256, range=[0,256])
    cdf = hist.cumsum()

    # Compute image mean
    img_mean = np.mean(image)

    # Initialize variables
    max_value = -1
    optimal_low = 0
    optimal_high = 0

    # Loop over possible thresholds
    for k1 in range(1, 254):
        for k2 in range(k1 + 1, 255):
            # Compute weights and means of classes
            w0 = cdf[k1]
            w1 = cdf[k2] - cdf[k1]
            w2 = cdf[255] - cdf[k2]
            mean0 = np.sum(hist[:k1] * np.arange(k1)) / w0 if w0 > 0 else 0
            mean1 = np.sum(hist[k1:k2] * np.arange(k1, k2)) / w1 if w1 > 0 else 0
            mean2 = np.sum(hist[k2:255] * np.arange(k2, 255)) / w2 if w2 > 0 else 0

            # Calculate variance
            sigma2 = w0 * (mean0 - img_mean)**2 + w1 * (mean1 - img_mean)**2 + w2 * (mean2 - img_mean)**2

            # Update thresholds if variance is greater
            if sigma2 > max_value:
                max_value = sigma2
                optimal_low = k1
                optimal_high = k2

    # Threshold the image
    result = np.zeros_like(img)
    result[img < optimal_low] = 0
    result[(img >= optimal_low) & (img < optimal_high)] = 128
    result[img >= optimal_high] = 255

    return optimal_low, optimal_high, result


def local_spectral_thresholding(img):

    image = img.copy()
 
    height = image.shape[0]
    width = image.shape[1]
    half_height = height//2 
    half_width = width//2

    # Getting the four section of the 2x2 image
    section_1 = image[:half_height, :half_width]
    section_2 = image[:half_height, half_width:]
    section_3 = image[half_height:, :half_width]
    section_4 = image[half_height:, half_width:]

    # Check if the threshold is calculated through Otsu's method(depending on mode) or given by the user
    _, _, section_1 = global_spectral_thresholding(section_1)
    _, _,section_2 = global_spectral_thresholding(section_2)
    _, _,section_3 = global_spectral_thresholding(section_3)
    _, _,section_4 = global_spectral_thresholding(section_4)

    # Regroup the sections to form the final image
    top_section = np.concatenate((section_1, section_2), axis = 1)
    bottom_section = np.concatenate((section_3, section_4), axis = 1)
    final_img = np.concatenate((top_section, bottom_section), axis=0)

    return final_img

# image = cv2.imread('cameraman.jpg', cv2.IMREAD_GRAYSCALE)
# segmented_image = local_spectral_thresholding(image)

# plt.imshow(segmented_image, cmap='gray')
# plt.show()