import numpy as np
import matplotlib.pyplot as plt
import cv2

def normalization(img):
    """
    Normalize the input image array.

    Parameters:
        img (numpy.ndarray): Input image array.

    Returns:
        numpy.ndarray: Normalized image array.
    """
    # normalized_img = img.copy()
    # if img.min() != 0:
    #     img_shifted_to_0 = img - img.min()
    #     normalization_scale = 1 / (img.max() - img.min())
    #     normalized_img = img_shifted_to_0 * normalization_scale 
    # else:
    #     normalized_img = img.astype(float)/ img.max()
    # final_img = (normalized_img * 255).astype('uint8')
    normalized_img = img.copy()
    if img.max() != 0:
        normalized_img = img.astype(float) / img.max()
    else:
        normalized_img = img.astype(float)
    final_img = (normalized_img * 255).astype('uint8')
    return final_img


def otsu_threshold(img):
    """
    Apply Otsu's thresholding method to the input image.

    Parameters:
        img (numpy.ndarray): Input image array.

    Returns:
        tuple: Tuple containing the thresholded image array and the calculated threshold value.
    """
    image = img.copy()
    image = normalization(img)
    # calculate_Histogram
    pixel_counts = [np.sum(image == i) for i in range(256)]
    max_variance = (0,-np.inf)
    
    for threshold in range(256):

        n1 = float(sum(pixel_counts[:threshold]))
        n2 = float(sum(pixel_counts[threshold:]))

        mu_1 = sum([i * pixel_counts[i] for i in range(0,threshold)]) / n1 if n1 > 0 else 0       
        mu_2 = sum([i * pixel_counts[i] for i in range(threshold, 256)]) / n2 if n2 > 0 else 0

        # calculate between-variance
        variance = n1 * n2 * (mu_1 - mu_2) ** 2

        if variance > max_variance[1]:
            max_variance = (threshold, variance)
    
    t = (max_variance[0]/255)*(img.max()-img.min()) + img.min()
    final_img = img.copy()
    final_img[img > t] = 255
    final_img[img < t] = 0 
    
    return final_img, t

def local_otsu_thresholding(img, mode = 1,  t1 = 0, t2 = 64, t3 = 128, t4 = 192):
    """
    Apply local Otsu thresholding to the input image.

    Parameters:
        image (numpy.ndarray): Input image array.
        mode (int): Mode flag. If mode=1, calculate threshold using Otsu's method; otherwise, use provided thresholds.
        t1, t2, t3, t4 (int): Threshold values for the four sections of the image (default values are for demonstration).

    Returns:
        numpy.ndarray: Thresholded image array.
    """
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
    if (mode == 1): 
        _, t1 = otsu_threshold(section_1)
        _, t2 = otsu_threshold(section_2)
        _, t3 = otsu_threshold(section_3)
        _, t4 = otsu_threshold(section_4)

    # Applying the threshold of each section on its corresponding section
    section_1[section_1 > t1] = 255
    section_1[section_1 < t1] = 0

    section_2[section_2 > t2] = 255
    section_2[section_2 < t2] = 0

    section_3[section_3 > t3] = 255
    section_3[section_3 < t3] = 0

    section_4[section_4 > t4] = 255
    section_4[section_4 < t4] = 0
    print(t1, t2, t3, t4)

    # Regroup the sections to form the final image
    top_section = np.concatenate((section_1, section_2), axis = 1)
    bottom_section = np.concatenate((section_3, section_4), axis = 1)
    final_img = np.concatenate((top_section, bottom_section), axis=0)

    return final_img


def global_otsu_thresholding(image, mode = 1, t= 128):
    """
    Apply global Otsu thresholding to the input image.

    Parameters:
        image (numpy.ndarray): Input image array.
        mode (int): Mode flag. If mode=1, calculate threshold using Otsu's method; otherwise, use provided threshold.
        t (int): Threshold value for the whole image (default value is for demonstration).

    Returns:
        numpy.ndarray: Thresholded image array.
    """
    # Check if the threshold is calculated through Otsu's method or the threshold is given by the user
    if (mode == 1): 
       _, t = otsu_threshold(image)

    final_img = image.copy()
    final_img[image > t] = 255
    final_img[image < t] = 0

    return final_img

# image = cv2.imread('cameraman.jpg', cv2.COLOR_RGB2GRAY)
# final_img= local_otsu_thresholding(image)
# plt.imshow(final_img)
# plt.show()

