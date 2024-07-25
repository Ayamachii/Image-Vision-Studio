import numpy as np
import pyqtgraph as pg
from PyQt5.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget, QHBoxLayout, QPushButton
import cv2
from functions.filters_edges import gaussian_filter, sobel_edge

# define 3*3 window
global window 
window = np.array([[0, 0], [0, 1], [0, -1], [1, 0],[-1, 0], [1, 1], [1, -1], [-1, 1], [-1, -1]])

def create_initial_contour(source):
    num_points = 60
    """
    Creates an initial contour for active contour models.

    Args:
        source (numpy.ndarray): Input image.

    Returns:
        initial_contour (numpy.ndarray): Initial contour coordinates.
    """
    num_points = 70
    angles = np.linspace(0, 2*np.pi, num_points)

    radius = 0.4 * min(source.shape[0], source.shape[1])
    center_x, center_y = source.shape[1]/2, source.shape[0]/2

    x = (center_x + radius * np.cos(angles)).astype(int)
    y = (center_y + radius * np.sin(angles)).astype(int)
    return np.array([x, y]).T

def internal_energy(contour_points, alpha, beta):
    """
    Calculates the internal energy of a contour defined by a set of points.
    Internal Energy = alpha * sum((sqrt((x_i+1 - x_i)^2 + (y_i+1 - y_i)^2)) - mean(sqrt( (x_i+1 - x_i)^2 + (y_i+1 - y_i)^2)) ))^2)
                    + beta * (sum((x_i+1 - 2*x_i + x_i-1)^2 + (y_i+1 - 2*y_i + y_i-1)^2))

    Args:
        contour_points (ndarray): Array of shape (N, 2) containing the coordinates of N points defining the contour.
        alpha (float): Weight parameter controlling the contribution of elasticity to the internal energy.
        beta (float): Weight parameter controlling the contribution of stiffness (curvature) to the internal energy.

    Returns:
        internal_energy (float): The total internal energy of the contour.
    """
    # Shift the contour points to get previous and next points
    previous_points = np.roll(contour_points, 1)
    next_points = np.roll(contour_points, -1)

    # Calculate displacement between consecutive points
    displacement = next_points - contour_points
    # Calculate Euclidean distance between consecutive points
    euclidean_distance = np.sqrt(displacement[:, 0]**2 + displacement[:, 1]**2)
    # Calculate elasticity energy
    elasticity = alpha * np.sum((euclidean_distance - np.mean(euclidean_distance))**2)
    
    # Calculate curvature energy
    curvature = next_points - 2 * contour_points + previous_points
    final_curvature = np.sum(curvature[:, 0]**2 + curvature[:, 1]**2)
    stiffness = beta * final_curvature
    # Compute the total internal energy
    internal_energy = elasticity + stiffness

    return internal_energy

def prepare_external_energy(image):
    """
    Prepares external energy components for active contours.

    Args:
        image (numpy.ndarray): Input image.

    Returns:
        blurred_image (numpy.ndarray): Blurred version of the input image.
        edges (numpy.ndarray): Edge map of the input image.
    """
    # Apply Gaussian blur to the input image
    blurred_image = gaussian_filter(image, kernel_size=5, sigma=3, is_gray=True)
    # Compute edges using Sobel filter
    edges = sobel_edge(blurred_image, is_gray=True, get_magnitude=True, get_direction=False)
    return blurred_image, edges

def external_energy(point, intensity, gradient, w_line, w_edge, gamma):
    """
    Computes external energy at a given point based on intensity and gradient information.
    External energy = gamma * (w_line * E_line + w_edge * E_edg) 

    Args:
        point (tuple): Coordinates of the point (x, y).
        intensity (numpy.ndarray): Intensity values of the image.
        gradient (numpy.ndarray): Gradient magnitude of the image.
        w_line (float): Weight for intensity term.
        w_edge (float): Weight for gradient term.
        gamma (float): Scaling factor.

    Returns:
        external_energy (float): External energy at the given point.
    """

    e_line = 0
    e_edge = 0

    # print(point[0])
    e_line = intensity[point[1], point[0]]
    e_edge = gradient[point[1], point[0]]

    # Compute external energy as a combination of intensity and gradient energies
    external_energy = gamma * (w_line * e_line + w_edge * e_edge)

    return external_energy
