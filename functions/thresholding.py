import numpy as np
from math import ceil

def local_thresholding(img, region_size=3):
    """
    Apply local_thresholding to the image, takes a greyscale image, the threshold is the mean value.
    Parameters:
        img (np.ndarray): The input image.
        region_size (int): The size of the square local region
    Returns:
        thresholded_image (np.ndarray): The result image after thresholding.
    """
    result = np.zeros_like(img)

    row_regions_count = ceil(img.shape[0] / region_size)
    col_regions_count = ceil(img.shape[1] / region_size)

    for row_index in range(row_regions_count):
        for col_index in range(col_regions_count):

            # check that the index has not gotten out of bound
            row_region_end_index = (row_index + 1) * region_size
            col_region_end_index = (col_index + 1) * region_size
            if row_region_end_index > img.shape[0] - 1:
                row_region_end_index = img.shape[0] - 1
            if col_region_end_index > img.shape[1] - 1:
                col_region_end_index = img.shape[1] - 1
            local_region = img[row_index * region_size:row_region_end_index, col_index * region_size:col_region_end_index].copy()

            local_threshold = local_region.flatten().mean()
            result[row_index * region_size : row_region_end_index, col_index * region_size : col_region_end_index] = apply_threshold(local_region, local_threshold)
    
    return result
        
def apply_threshold(img, threshold, low_value = 0, high_value = 255):
    """
    Apply thresholding (binary) to the image, takes a greyscale image.
    Parameters:
        img (np.ndarray): The input image.
        threshold (int): The required threshold value.
    Returns:
        thresholded_image (np.ndarray): The result image after thresholding.
    """
    thresholded_image = np.ones(img.shape) * low_value
    higherRow, higherCol = np.where(img >= threshold)
    thresholded_image[higherRow, higherCol] = high_value
    return thresholded_image

