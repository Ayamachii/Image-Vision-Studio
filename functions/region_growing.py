import sys
import numpy as np
import pyqtgraph as pg
from pyqtgraph import ImageView
from PyQt5.QtCore import Qt, pyqtSignal
import cv2

def get_8neighbours(x, y, max_x, max_y):
    """
    Get 8-neighbour coordinates of a given point (x, y) within the image of size (N=max_x, M=max_y).

    Parameters:
        x: x-coordinate of the point
        y: y-coordinate of the point
        max_x: Maximum value of x-coordinate
        max_y: Maximum value of y-coordinate

    Returns:
        List of 8-neighbour coordinates
    """
    neighbours = []
    # Check if the neighbour is within the image boundaries
    if (x + 1) < max_x:
        neighbours.append((x + 1, y))
    if (x - 1) >= 0:
        neighbours.append((x - 1, y))
    if (y + 1) < max_y:
        neighbours.append((x, y + 1))
    if (y - 1) >= 0:
        neighbours.append((x, y - 1))
    if (x + 1) < max_x and (y + 1) < max_y:
        neighbours.append((x + 1, y + 1))
    if (x - 1) >= 0 and (y - 1) >= 0:
        neighbours.append((x - 1, y - 1))
    if (x - 1) >= 0 and (y + 1) < max_y:
        neighbours.append((x - 1, y + 1))
    if (x + 1) < max_x and (y - 1) >= 0:
        neighbours.append((x + 1, y - 1))
    return neighbours

def get_similarity(image, x_seed, y_seed, x_neighbour, y_neighbour):
    """
    Calculate similarity between a seed point and its neighbour using intensity difference.

    Args:
        image(numpy.ndarray): Input image
        x_seed: x-coordinate of the seed point
        y_seed: y-coordinate of the seed point
        x_neighbour: x-coordinate of the neighbour
        y_neighbour: y-coordinate of the neighbour

    Returns:
        Absolute difference between intensities of the seed and neighbour pixels
    """
    return abs(int(image[x_seed, y_seed]) - int(image[x_neighbour, y_neighbour]))

def grow_region(image, threshold, seed_points):
    """
    Perform region growing on the input image.

    Parameters:
        image (numpy.ndarray): Input RGB image
        threshold: Similarity threshold for region growing

    Returns: None
    """
    # Convert RGB image to Luv color space
    luv_image = cv2.cvtColor(image, cv2.COLOR_RGB2Luv)
    height, width,_ = luv_image.shape
    # Extract L channel from Luv image
    L_channel = luv_image[:, :, 0]
    # Create a mask to mark seed points
    seed_marked = np.array(L_channel)
    # Create an array to keep track of visited points
    visited = np.zeros(L_channel.shape)
    # Loop until there are no seed points (stack is empty)
    while(len(seed_points) > 0 ):
        x_seed, y_seed = seed_points.pop()
        # Mark the seed point
        seed_marked[x_seed, y_seed] = 255
        visited[x_seed, y_seed] = 1
        # Iterate through 8-neighbours of the seed point
        for x_neighbour, y_neighbour in get_8neighbours(x=x_seed, y=y_seed, max_x=height, max_y=width):
            # Check if neighbour is already visited
            if visited[x_neighbour, y_neighbour] == 1:
                continue
            # Check if similarity is within threshold and neighbour is not marked as seed point
            if get_similarity(L_channel, x_seed, y_seed, x_neighbour, y_neighbour) <= threshold and seed_marked[x_neighbour, y_neighbour] != 255:
                seed_marked[x_neighbour, y_neighbour] = 255
                seed_points.append((x_neighbour, y_neighbour))

            visited[x_neighbour, y_neighbour] == 1

    print("DONE!")

    seed_points = []

    # Create a new array with three channels
    output_image = np.zeros_like(luv_image)

    # Copy L channel values
    output_image[:, :, 0] = seed_marked

    # Assign U and V channel values from luv_image
    output_image[:, :, 1:] = luv_image[:, :, 1:]

    output_image = cv2.cvtColor(output_image, cv2.COLOR_Luv2RGB)

    return output_image

