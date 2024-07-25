import cv2
import matplotlib.pyplot as plt
import numpy as np
import timeit


def apply_harris_operator(img: np.ndarray, k: float = 0.05) -> np.ndarray:
    """
    Apply the Harris corner detection algorithm to an input image.
    Parameters:
        img (np.ndarray): The input image to which the Harris operator will be applied.
        k (float): Sensitivity factor to separate corners from edges.
                            Smaller values of k result in detection of sharper corners. Default is 0.05.
    Returns:
        np.ndarray: An array representing the Harris response of the input image.
    """
    img_copy = np.copy(img)
    img = cv2.cvtColor(img_copy, cv2.COLOR_BGR2GRAY)
    I_x = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=5)
    I_y = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=5)
    # The weights Wp should be circularly symmetric (for rotation invariance)
    Sxx = cv2.GaussianBlur(src=I_x ** 2, ksize=(5, 5), sigmaX=0)
    Syy = cv2.GaussianBlur(src=I_y ** 2, ksize=(5, 5), sigmaX=0)
    # Sxy = Syx
    Sxy = cv2.GaussianBlur(src=I_x * I_y, ksize=(5, 5), sigmaX=0)
    # H Matrix is 
    # [ Ix^2        Ix * Iy ]
    # [ Iy * Ix     Iy^2    ]
    # Harris Response R
    det_H = Sxx * Syy - Sxy ** 2
    trace_H = Sxx + Syy
    # corner strength function
    harris_response = det_H - k * (trace_H ** 2)
    return harris_response


def apply_lamda_minus_operator(img: np.ndarray) -> np.ndarray:
    """
    Apply the λ- corner detection operator to an input image. 
    Parameters:
        img (np.ndarray): The input image to which the λ- operator will be applied.
    Returns:
        Lamda_ (np.ndarray): An array representing the λ- corner response of the input image.
    """
    img_copy = np.copy(img)
    img = cv2.cvtColor(img_copy, cv2.COLOR_BGR2GRAY)
    I_x = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=5)
    I_y = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=5)
    Sxx = cv2.GaussianBlur(src=I_x ** 2, ksize=(5, 5), sigmaX=0)
    Sxy = cv2.GaussianBlur(src=I_x * I_y, ksize=(5, 5), sigmaX=0)
    Syy = cv2.GaussianBlur(src=I_y ** 2, ksize=(5, 5), sigmaX=0)
    det_M = (Sxx * Syy) - (Sxy ** 2)
    trace_M = Sxx + Syy
    Lamda_ = 0.5 * (trace_M - np.sqrt(trace_M**2 - 4 * det_M))
    return Lamda_

def get_operator_indices(operator_response: np.ndarray, threshold: float = 0.01) -> tuple:
    """
    Compute local maxima of Operator  response R to identify corners, edges, and flat areas.
    Parameters:
        operator_response (np.ndarray): An array representing the Operator  response of an image.
        threshold (float, optional): Threshold value used for computing local maxima. Default is 0.01.
    Returns:
        tuple: A tuple of three numpy arrays representing indices of corners, edges, and flat areas respectively.
    Note:
        We can use these peak values to isolate corners and edges
        - Corner : R > max_response * threshold (point of intereset)
        - Edge   : R < 0
        - Flat   : R = 0
    """
    operator_copy = np.copy(operator_response)
    # Dilate the points to be more clear
    operator_matrix = cv2.dilate(operator_copy, None)
    max_corner_response = np.max(operator_matrix)
    # Indices of each corner, edges and flat area
    # Threshold for an optimal value, it may vary depending on the image.
    corner_indices = np.array(operator_matrix > (max_corner_response * threshold), dtype="int8")
    edges_indices = np.array(operator_matrix < 0, dtype="int8")
    flat_indices = np.array(operator_matrix == 0, dtype="int8")
    return corner_indices, edges_indices, flat_indices

def map_indices_to_image(img: np.ndarray, indices: np.ndarray, color: list):
    """
    Draw dots on the input image based on provided indices.
    Parameters:
        img (np.ndarray): The input image on which dots will be drawn.
        indices (np.ndarray): An array of 0's and 1's representing indices of interest.
                              1 indicates an index of interest where a dot will be drawn.
        color (list): Color to draw the dots with, specified as a list, e.g., [0, 0, 225] for blue.
    Returns:
        np.ndarray: A new image with dots drawn on it based on provided indices.
    """
    img_copy = np.copy(img)
    # Make sure that the original source shape == indices shape
    img = img_copy[:indices.shape[0], :indices.shape[1]]
    # Mark each index with dot
    img[indices == 1] = color
    return img

