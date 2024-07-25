import numpy as np
import cv2 
from scipy.ndimage import maximum_filter
import matplotlib.pyplot as plt

def line_detection(image: np.ndarray,th_low,th_high)-> tuple:
    """
    Detects lines in an image using the Hough Transform.
    Args:
        image (numpy.ndarray): Input image, should be in BGR format.
        th_low (int): Lower threshold for the Canny edge detector.
        th_high (int): Upper threshold for the Canny edge detector.
    Returns:
        tuple: A tuple containing:
            - accumulator (numpy.ndarray): The Hough accumulator array.
            - thetas (numpy.ndarray): Array of theta values.
            - rhos (numpy.ndarray): Array of rho values.
    """
    gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur_img = cv2.GaussianBlur(gray_img, (3,3), 1.5)
    edge_img = cv2.Canny(blur_img, th_low, th_high)
    height, width = edge_img.shape
    max_dist = int(np.sqrt(height**2 + width**2)) 
    thetas = np.deg2rad(np.arange(-90, 90))
    # Generate array of evenly spaced numbers
    rhos = np.linspace(-max_dist, max_dist, 2*max_dist)
    # 2d matrix to fill all votes of all rhos and thier corresponding thetas
    accumulator = np.zeros((2 * max_dist +1, len(thetas)))
    for h in range(height):
        for w in range(width):
            if edge_img[h,w] > 0:
                for phi in range(len(thetas)):
                    r = w * np.cos(thetas[phi]) + h * np.sin(thetas[phi])
                    # to ensure that r is non-negative and fit within the range of the accumulator array
                    # shift the origin of the rho values to the center of the accumulator array.
                    accumulator[int(r) + max_dist, phi] += 1
    return accumulator, thetas, rhos

def hough_peaks(acc, peaks, neighborhood_size=3)-> tuple:
    """
    Finds peaks in the Hough accumulator array.
    Args:
        acc (numpy.ndarray): The Hough accumulator array.
        peaks (int): Number of peaks to find.
        neighborhood_size (int): Size of the neighborhood around each peak.
    Returns:
        tuple: A tuple containing:
            - peaks_indices (list): List of indices corresponding to the peaks.
            - acc (numpy.ndarray): The modified Hough accumulator array.
    """
    peaks_indices = []
    acc_copy = np.copy(acc)  
    for _ in range(peaks):
        # Find the maximum value index in the accumulator array
        max_index = np.argmax(acc_copy)
        max_y, max_x = np.unravel_index(max_index, acc_copy.shape)  # Convert index to 2D coordinates
        peaks_indices.append((max_y, max_x))
        # Update the accumulator array to remove the neighborhood around the peak
        # This ensures that peaks within this neighborhood will not be detected again in subsequent iterations.
        min_x = max(0, max_x - (neighborhood_size // 2))
        max_x = min(acc.shape[1], max_x + (neighborhood_size // 2))
        min_y = max(0, max_y - (neighborhood_size // 2))
        max_y = min(acc.shape[0], max_y + (neighborhood_size // 2))
        acc_copy[min_y:max_y, min_x:max_x] = 0  # Set neighborhood values to 0
        # Highlight the peak in the original accumulator array
        acc[max_y, max_x] = 255
    return peaks_indices, acc


def hough_lines_draw(img, peaks_indices, rhos, thetas)-> None:
    """
    Draws lines on an image based on the Hough peaks.
    Args:
        img (numpy.ndarray): Input image.
        peaks_indices (list): List of peak indices.
        rhos (numpy.ndarray): Array of rho values.
        thetas (numpy.ndarray): Array of theta values.
    Returns:
        None
    """
    for i in range(len(peaks_indices)):
        rho = rhos[peaks_indices[i][0]]
        theta = thetas[peaks_indices[i][1]]
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a * rho
        y0 = b * rho
        # these are then scaled so that the lines go off the edges of the image
        x1 = int(x0 - 1000 * (b))
        y1 = int(y0 + 1000 * (a))
        x2 = int(x0 + 1000 * (b))
        y2 = int(y0 - 1000 * (a))
        cv2.line(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

def hough_lines(img: np.ndarray,peaks: int = 10,neighborhood_size : int = 3,th_low: int=50,th_high : int = 150)-> np.ndarray:
    """
    Detects lines in an image using the Hough Transform and draws them on the image.
    Args:
        img (numpy.ndarray): Input image.
        peaks (int): Number of peaks to find in the Hough accumulator array.
        neighborhood_size (int): Size of the neighborhood around each peak.
        th_low (int): Lower threshold for the Canny edge detector.
        th_high (int): Upper threshold for the Canny edge detector.
    Returns:
        img_copy (numpy.ndarray): Image with detected lines drawn on it.
    """
    img_copy = np.copy(img)
    acc, thetas, rhos = line_detection(img_copy,th_low,th_high)
    indicies, acc = hough_peaks(acc, peaks, neighborhood_size) 
    hough_lines_draw(img_copy, indicies, rhos, thetas)
    return img_copy
