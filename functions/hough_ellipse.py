import cv2
import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt


def find_hough_ellipses(image, edge_image, a_min, a_max, b_min, b_max, delta_a, delta_b, num_thetas, bin_threshold, post_process=True):
    """
    Detect ellipses using Hough Transform.

    Parameters:
        image (numpy.ndarray): Input image.
        edge_image (numpy.ndarray): Edge-detected image.
        a_min (int): Minimum value of semi-major axis.
        a_max (int): Maximum value of semi-major axis.
        b_min (int): Minimum value of semi-minor axis.
        b_max (int): Maximum value of semi-minor axis.
        delta_a (int): Step size for semi-major axis.
        delta_b (int): Step size for semi-minor axis.
        num_thetas (int): Number of steps for theta.
        bin_threshold (float): Threshold for voting percentage.
        post_process (bool, optional): Perform post-processing. Defaults to True.

    Returns:
        tuple: A tuple containing the output image with detected ellipses and a list of detected ellipses.
    """


    img_height, img_width = edge_image.shape[:2]
  
    dtheta = int(360 / num_thetas)
    thetas = np.arange(0, 360, step=dtheta)
    
    a_values = np.arange(a_min, a_max, step=delta_a)
    b_values = np.arange(b_min, b_max, step=delta_b)
    
    cos_thetas = np.cos(np.deg2rad(thetas))
    sin_thetas = np.sin(np.deg2rad(thetas))
    
    ellipse_candidates = []
    for a in a_values:
        for b in b_values:
            for theta in range(num_thetas):
                ellipse_candidates.append((a, b, int(a * cos_thetas[theta]), int(b * sin_thetas[theta])))
        
    
    accumulator = defaultdict(int)
    
    for edge_pixel in np.argwhere(edge_image != 0):
        y, x = edge_pixel[0], edge_pixel[1]
        for a, b, acos_t, asin_t in ellipse_candidates:
            x_center = x - acos_t
            y_center = y - asin_t
            accumulator[(x_center, y_center, a, b)] += 1

    
    output_img = image.copy()
    out_ellipses = []
    
    # for candidate_ellipse, votes in sorted(accumulator.items(), key=lambda i: -i[1]):
    for candidate_ellipse, votes in accumulator.items():
        x, y, a, b = candidate_ellipse
        current_vote_percentage = votes / num_thetas
        if current_vote_percentage >= bin_threshold: 
            out_ellipses.append((x, y, a, b, current_vote_percentage))
            print(x, y, a, b, current_vote_percentage)
      
      
    if post_process:
        pixel_threshold = 1
        postprocess_ellipses = []
        for x, y, a, b, v in out_ellipses:
            if all(abs(x - xc) > pixel_threshold or abs(y - yc) > pixel_threshold or abs(a - ac) > pixel_threshold or abs(b - bc) > pixel_threshold for xc, yc, ac, bc, v in postprocess_ellipses):
                postprocess_ellipses.append((x, y, a, b, v))
        out_ellipses = postprocess_ellipses
    
    for x, y, a, b, v in out_ellipses:
        output_img = cv2.ellipse(output_img, (x, y), (a, b), 0, 0, 360, (0, 255, 0), 2)
  
    return output_img, out_ellipses

