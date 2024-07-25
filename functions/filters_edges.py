import numpy as np
import cv2
from scipy.signal import convolve2d
from PyQt5 import QtGui
QPixmap = QtGui.QPixmap


def convv(img,kernel_size,kernel_type):
    """
    Apply convolution operation to the image.
    Args:
        img (numpy.ndarray): The input image.
        kernel_size (int): The size of the kernel.
        kernel_type (numpy.ndarray): The convolution kernel.
    Returns:
        filtered_img(numpy.ndarray): The final filtered image.
    """
    padded_img = np.pad(img, ((kernel_size//2, kernel_size//2), (kernel_size//2, kernel_size//2)), mode='constant')
    filtered_img = np.zeros_like(img, dtype=np.float32)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            filtered_img[i, j] = np.sum(kernel_type * padded_img[i:i+kernel_size, j:j+kernel_size])
    return filtered_img
    
# -----------------------------------------------------------------Gaussian Filter-------------------------------------------------------------------------
def gaussian_kernel(kernel_size, sigma):
    """
    Generate a Gaussian kernel.
    Args:
        kernel_size (int): The size of the kernel.
        sigma (float): The standard deviation of the Gaussian distribution.
    Returns:
        numpy.ndarray: The Gaussian kernel.
    """
    gaussian =np.zeros((kernel_size,kernel_size))
    kernel_size=kernel_size//2
    for x in range(-kernel_size,kernel_size+1):
        for y in range(-kernel_size,kernel_size+1): 
            gaussian[x+kernel_size,y+kernel_size] = (1 / (2 * np.pi * sigma)) * np.exp(-(np.square(x) + np.square(y)) / (2 * np.square(sigma)))
    return gaussian

def gaussian_filter(img, kernel_size, sigma, is_gray=True):
    """
    Apply Gaussian filter to the image.
    Args:
        img (numpy.ndarray): The input image.
        kernel_size (int): The size of the Gaussian kernel.
        sigma (float): The standard deviation of the Gaussian distribution.
        is_gray (bool, optional): Whether the image is grayscale. Defaults to True.
    Returns:
        filtered_img (numpy.ndarray): The gaussian filtered image.
    """
    if is_gray:
        filtered_img = convv(img, kernel_size, gaussian_kernel(kernel_size, sigma))
        return filtered_img
    else:
        b, g, r = cv2.split(img)
        b_filtered = convv(b, kernel_size, gaussian_kernel(kernel_size, sigma))
        g_filtered = convv(g, kernel_size, gaussian_kernel(kernel_size, sigma))
        r_filtered = convv(r, kernel_size, gaussian_kernel(kernel_size, sigma))
        filtered_img = cv2.merge((b_filtered, g_filtered, r_filtered))
        return filtered_img.astype(np.uint8)
# -----------------------------------------------------------------Average Filter-------------------------------------------------------------------------

def avg_filter(img, kernel_size, is_gray=True):
    """
    Apply average filter to the image.
    Args:
        img (numpy.ndarray): The input image.
        kernel_size (int): The size of the average filter kernel.
        is_gray (bool, optional): Whether the image is grayscale. Defaults to True.
    Returns:
        avg_img (numpy.ndarray): The average filtered image.
    """
    if is_gray:  
        avg_kernel = np.ones((kernel_size, kernel_size), dtype=np.float32) / (kernel_size * kernel_size)
        avg_img = convv(img, kernel_size, avg_kernel)
        return avg_img
    else: 
        b, g, r = cv2.split(img)
        b_filtered = convv(b, kernel_size, np.ones((kernel_size, kernel_size), dtype=np.float32) / (kernel_size * kernel_size))
        g_filtered = convv(g, kernel_size, np.ones((kernel_size, kernel_size), dtype=np.float32) / (kernel_size * kernel_size))
        r_filtered = convv(r, kernel_size, np.ones((kernel_size, kernel_size), dtype=np.float32) / (kernel_size * kernel_size))
        avg_img = cv2.merge((b_filtered, g_filtered, r_filtered))
        return avg_img.astype(np.uint8)
# -----------------------------------------------------------------Median Filter-------------------------------------------------------------------------
    
def median_filter(img, kernel_size, is_gray=True):
    """
    Apply median filter to the image.
    Args:
        img (numpy.ndarray): The input image.
        kernel_size (int): The size of the median filter kernel.
        is_gray (bool, optional): Whether the image is grayscale. Defaults to True.
    Returns:
        filtered_image (numpy.ndarray): The median filtered image.
    """
    if is_gray:  
        w, h = img.shape
        filtered_image = np.zeros_like(img,  dtype=np.float32)  
        pad_width = kernel_size // 2
        padded_img = np.pad(img, ((pad_width, pad_width), (pad_width, pad_width)), mode='constant')
        for i in range(pad_width, w + pad_width):
            for j in range(pad_width, h + pad_width):
                mask = padded_img[i-pad_width:i+pad_width+1, j-pad_width:j+pad_width+1]
                filtered_image[i-pad_width, j-pad_width] = np.median(mask)
        return filtered_image
    else: 
        b, g, r = cv2.split(img)
        b_filtered = median_filter(b, kernel_size, is_gray=True)
        g_filtered = median_filter(g, kernel_size, is_gray=True)
        r_filtered = median_filter(r, kernel_size, is_gray=True)
        filtered_image = cv2.merge((b_filtered, g_filtered, r_filtered))
        return filtered_image.astype(np.uint8)
# ----------------------------------------------------------------------------------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------Edges----------------------------------------------------------------------------------------
    
def apply_kernel(img: np.ndarray, horizontal_kernel: np.ndarray, vertical_kernel: np.ndarray, is_gray: bool = True, return_edge: bool = False):
    """
    Apply the given horizontal and vertical kernels to the image for edge detection.
    Parameters:
        img (np.ndarray): The input image.
        horizontal_kernel (np.ndarray): The horizontal kernel for convolution.
        vertical_kernel (np.ndarray): The vertical kernel for convolution.
        is_gray (bool, optional): Indicates whether the image is grayscale. Defaults to True.
        return_edge (bool, optional): Whether to return the edges of image with magnitude of image. Defaults to False.
    Returns:
        mag (np.ndarray): The edge image after applying the kernels.
    """
    if is_gray:
        horizontal_edge = convolve2d(img, horizontal_kernel, mode='same', boundary='symm')
        vertical_edge = convolve2d(img, vertical_kernel, mode='same', boundary='symm')
    else:
        b, g, r = cv2.split(img)
        b_horizontal_edge = convolve2d(b, horizontal_kernel, mode='same', boundary='symm')
        b_vertical_edge = convolve2d(b, vertical_kernel, mode='same', boundary='symm')
        g_horizontal_edge = convolve2d(g, horizontal_kernel, mode='same', boundary='symm')
        g_vertical_edge = convolve2d(g, vertical_kernel, mode='same', boundary='symm')
        r_horizontal_edge = convolve2d(r, horizontal_kernel, mode='same', boundary='symm')
        r_vertical_edge = convolve2d(r, vertical_kernel, mode='same', boundary='symm')
        horizontal_edge = cv2.merge((b_horizontal_edge, g_horizontal_edge, r_horizontal_edge))
        vertical_edge = cv2.merge((b_vertical_edge, g_vertical_edge, r_vertical_edge))
    mag = np.sqrt(np.power(horizontal_edge, 2.0) + np.power(vertical_edge, 2.0))
    if return_edge:
        return mag, horizontal_edge, vertical_edge
    return mag

# -------------------------------------------------------------------------Prewitt Edges----------------------------------------------------------------------------------
def prewitt_edge(img: np.ndarray, is_gray: bool = False):
    """
    Apply Prewitt edge detection to the image.
    Parameters:
        img (np.ndarray): The input image.
        is_gray (bool, optional): Indicates whether the image is grayscale. Defaults to False.
    Returns:
        prewitt_edges (np.ndarray): The edge image after applying Prewitt edge detection.
    """
    vertical = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]])
    horizontal = np.flip(vertical.T)
    prewitt_edges = apply_kernel(img, horizontal, vertical, is_gray)
    return prewitt_edges

# -------------------------------------------------------------------------Roberts Edges----------------------------------------------------------------------------------
def roberts_edge(img: np.ndarray, is_gray: bool = False):
    """
    Apply Roberts edge detection to the image.
    Parameters:
        img (np.ndarray): The input image.
        is_gray (bool, optional): Indicates whether the image is grayscale. Defaults to False.
    Returns:
        RobertsEdges (np.ndarray): The edge image after applying Roberts edge detection.
    """
    vertical = np.array([[0, 1], [-1, 0]])
    horizontal = np.flip(vertical.T)
    roberts_edges = apply_kernel(img, horizontal, vertical, is_gray)
    return roberts_edges

# -------------------------------------------------------------------------Sobel Edges----------------------------------------------------------------------------------
def sobel_edge(img: np.ndarray, is_gray: bool = True, get_magnitude: bool = True, get_direction: bool = False):
    """
    Apply Sobel edge detection to the image.
    Parameters:
        img (np.ndarray): The input image.
        is_gray (bool, optional): Indicates whether the image is grayscale. Defaults to True.
        get_magnitude (bool, optional): Whether to return the magnitude of gradients. Defaults to True.
        get_direction (bool, optional): Whether to return the direction of gradients. Defaults to False.
    Returns:
        Union[SobelEdges (np.ndarray), Tuple[np.ndarray, np.ndarray])]: The edge image or tuple of magnitude and direction.
    """
    horizontal = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    vertical = np.flip(horizontal.T)
    sobel_edges, horizontal_edge, vertical_edge = apply_kernel(img, horizontal, vertical, is_gray, True)
    horizontal_edge = horizontal_edge[:-2, :-2]
    vertical_edge = vertical_edge[:-2, :-2]
    if not get_magnitude:
        return horizontal_edge, vertical_edge
    if get_direction:
        direction = np.arctan2(vertical_edge, horizontal_edge)
        return sobel_edges, direction
    return sobel_edges

# -------------------------------------------------------------------------Canny Edges----------------------------------------------------------------------------------
def canny_edge(img: np.ndarray,is_gray: bool = True):
    """
    Apply Canny edge detection to the image.
    Parameters:
        img (np.ndarray): The input image.
        is_gray (bool, optional): Indicates whether the image is grayscale. Defaults to True.
    Returns:
        CannyEdges (np.ndarray): The edge image after applying Canny edge detection.
    Note:
        The Canny edge detection algorithm is typically applied to grayscale
        images. If the input image is colored and 'is_gray' is set to False,
        it will be converted to grayscale internally before processing.
        The low and high thresholds for double thresholding are determined based
        on the maximum pixel intensity of the image. The low threshold is set to 
        a percentage (0.02 by default) of the maximum intensity, while the high
        threshold is set to a percentage (0.5 by default) of the maximum intensity.
    """
    if not is_gray:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # First Apply Gaussian Filter
    filtered_image = gaussian_filter(img, kernel_size=3, sigma=9)
    # Second Get Gradient Magnitude & Direction
    gradient_magnitude, gradient_direction = sobel_edge(filtered_image, get_magnitude=True, get_direction=True)
    # Third Apply Non-Maximum Suppression
    suppressed_image = non_maximum_suppression(gradient_magnitude, gradient_direction)
    low_ratio = 0.02
    high_ratio = 0.5
    high_threshold = img.max() * low_ratio
    low_threshold = high_threshold * high_ratio
    # Fourth Apply Double Thresholding
    thresholded_image = double_threshold(suppressed_image, low_threshold, high_threshold)
    # Fifth Apply hysteresis
    canny_edges = hysteresis(thresholded_image, 70, 255)
    return canny_edges


def non_maximum_suppression(gradient_magnitude: np.ndarray, gradient_direction: np.ndarray):
    """
    Perform non-maximum suppression on the gradient magnitude image.
    Parameters:
        gradient_magnitude (np.ndarray): The gradient magnitude image.
        gradient_direction (np.ndarray): The gradient direction image(in rad).
    Returns:
        suppressed_image (np.ndarray): The result image after non-maximum suppression.
    Note:
        It's the third step in canny edge detection algorithm
    """
    M, N = gradient_magnitude.shape
    suppressed_image = np.zeros(gradient_magnitude.shape)
    gradient_direction = np.rad2deg(gradient_direction)
    gradient_direction += 180
    for row in range(M):
        for col in range(N):
            try:
                direction = gradient_direction[row, col]
                # 0°
                if (0 <= direction < 22.5) or (337.5 <= direction <= 360):
                    before_pixel = gradient_magnitude[row, col - 1]
                    after_pixel = gradient_magnitude[row, col + 1]
                # 45°
                elif (22.5 <= direction < 67.5) or (202.5 <= direction < 247.5):
                    before_pixel = gradient_magnitude[row + 1, col - 1]
                    after_pixel = gradient_magnitude[row - 1, col + 1]
                # 90°
                elif (67.5 <= direction < 112.5) or (247.5 <= direction < 292.5):
                    before_pixel = gradient_magnitude[row - 1, col]
                    after_pixel = gradient_magnitude[row + 1, col]
                # 135°
                else:
                    before_pixel = gradient_magnitude[row - 1, col - 1]
                    after_pixel = gradient_magnitude[row + 1, col + 1]
                if gradient_magnitude[row, col]>= max(before_pixel,after_pixel):
                    suppressed_image[row, col] = gradient_magnitude[row, col]
            except IndexError as e:
                pass
    return suppressed_image

def double_threshold(image, LowThreshold, high_threshold):
    """
    Apply double thresholding to the image.
    Parameters:
        image (np.ndarray): The input image.
        LowThreshold (int): The low threshold value.
        high_threshold (int): The high threshold value.
    Returns:
        thresholded_image (np.ndarray): The result image after double thresholding.
    Note:
        It's the fourth step in canny edge detection algorithm
    """
    thresholded_image = np.zeros(image.shape)
    weak =70
    strong = 255
    strong_row, strong_col = np.where(image >= high_threshold)
    weakRow, weakCol = np.where((image <= high_threshold) & (image >= LowThreshold))
    thresholded_image[strong_row, strong_col] = strong
    thresholded_image[weakRow, weakCol] = weak
    return thresholded_image
    

def hysteresis(image, weak=70, strong=255):
    """
    Apply hysteresis thresholding to the image.
    Parameters:
        image (np.ndarray): The input image.
        weak (int, optional): The weak threshold value. Defaults to 70.
        strong (int, optional): The strong threshold value. Defaults to 255.
    Returns:
        image (np.ndarray): The result after hysteresis thresholding.
    Note:
        It's the final step in canny edge detection algorithm
    """
    M, N = image.shape
    for i in range(M):
        for j in range(N):
            if image[i, j]== weak:
                try:
                    if ((image[i-1,j] == strong) or (image[i + 1, j] == strong) or (
                            image[i, j - 1] == strong)
                            or (image[i, j + 1] == strong) or (image[i+1, j + 1] == strong)
                            or (image[i - 1, j - 1] == strong) or (image[i - 1, j-1] == strong) or (
                                    image[i - 1, j + 1] == strong)):
                        image[i, j] = strong
                    else:
                        image[i, j] = 0
                except IndexError as e:
                    pass
    return image



