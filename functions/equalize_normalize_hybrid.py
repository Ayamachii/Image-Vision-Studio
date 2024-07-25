
import numpy as np
import pyqtgraph as pg
import cv2
from functions.main_tab_functionality import get_cumulative_frequencies
from functions.filters_edges import gaussian_filter

def histogram_equalization(image):
    """
    Perform histogram equalization on an input image.

    Args:
        image (numpy.ndarray): Input image (grayscale or colored).

    Returns:
        numpy.ndarray: Equalized image.
    """
    # convert to hsv
    hsv_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)

    # Compute histogram for V channel
    hist_original, bin_edges_original = np.histogram(hsv_image[:, :, 2].flatten(), bins=256, range=[0, 256])

    # Compute cumulative distribution function (CDF)
    cdf = get_cumulative_frequencies(hist_original)

    # Normalize CDF
    cdf_normalized = cdf / cdf.max()

    # Create mapping function
    mapping= np.uint8(255 * cdf_normalized)

    # Apply mapping to each pixel in the image for V channel
    equalized_image = mapping[image[:, :, 2]]

    # Merge equalized Value (V) channel back into the HSV image
    hsv_image[:, :, 2] = equalized_image

    hist_eq, bin_edges_eq = np.histogram(hsv_image[:, :, 2].flatten(), bins=256, range=[0, 256])
    
    # Convert to original color space
    equalized_image = cv2.cvtColor(hsv_image, cv2.COLOR_HSV2RGB)

    return equalized_image, hist_original, hist_eq, bin_edges_original, bin_edges_eq

def equalize_image(image):
    """
    Perform histogram equalization on an input image using OpenCV.

    Args:
        image (numpy.ndarray): Input image (grayscale or colored).

    Returns:
        numpy.ndarray: Equalized image.
    """
    # Convert BGR image to HSV
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Equalize the value (V) channel
    hsv_image[:, :, 2] = cv2.equalizeHist(hsv_image[:, :, 2])

    # Convert to RGB
    equalized_image = cv2.cvtColor(hsv_image, cv2.COLOR_HSV2RGB)

    return equalized_image

def normalize(image):
    ''' 
    Function to normalize an input image

    Args:
        image (numpy.ndarray): Input image (grayscale or colored).

    Returns:
        numpy.ndarray: Normalized image.
    '''
    # convert to hsv
    image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)

    # Extract the V channel (brightness) from the input image
    v_channel = image[:, :, 2]

    # Calculate the minimum and maximum pixel values in the V channel
    v_min = v_channel.min()
    v_max = v_channel.max()

    # Normalize the pixel values of the V channel to the range [0, 255]
    normalized_v = ((v_channel - v_min) * (255.0 / (v_max - v_min))).astype(np.uint8)

    # Replace the original V channel with the normalized V channel
    normalized_image = np.copy(image)
    normalized_image[:, :, 2] = normalized_v

    # Convert to original color space
    normalized_image = cv2.cvtColor(image, cv2.COLOR_HSV2RGB)

    return normalized_image

def generate_hybrid_image(image1, image2, segma1=20, segma2=1, kernel_size1=7, kernel_size2=3, is_grey1=True, is_grey2=True):
    '''
    Generates a hybrid image of two input images
    Args:
        image1, image2 (numpy.ndarray): Input images (grayscale or colored).

    Returns:
        numpy.ndarray: Hybrid image.
    '''

    # Apply Gaussian filter to image1
    blurred1 = gaussian_filter(image1, kernel_size1, segma1, is_grey1)

    # Apply high-pass filtering to image2
    blurred2 = gaussian_filter(image2, kernel_size2, segma2, is_grey2)
    high_freq_image = image2 - blurred2

    # Combine both images
    hybrid_image = blurred1 + high_freq_image

    hybrid_image = np.clip(hybrid_image, 0, 255).astype(np.uint8)

    return hybrid_image

def resize_images(image1, image2):
    """
    Resize the input images to the smaller size.
    Args:
        image1, image2 (numpy.ndarray): Input images.
    Returns:
        numpy.ndarray: The resized images.
    """
    # Determine which image is larger
    if image1.shape[0] * image1.shape[1] > image2.shape[0] * image2.shape[1]:
        # Resize image2 to match the shape of image1
        resized_image1 = cv2.resize(image1, (image2.shape[1], image2.shape[0]))
        resized_image2 = image2
    else:
        # Resize image1 to match the shape of image2
        resized_image2 = cv2.resize(image2, (image1.shape[1], image1.shape[0]))
        resized_image1 = image1

    return resized_image1, resized_image2