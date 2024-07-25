import cv2
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
# from filters_edges import canny_edge

def find_hough_circles(image, edge_image, r_min, r_max, delta_r, num_thetas, bin_threshold, post_process = True):


  img_height, img_width = edge_image.shape[:2]
  
  # Calculate the step size for theta (dtheta) based on the number of steps (num_thetas).
  dtheta = int(360 / num_thetas)

  # Generate an array of theta values (thetas) from 0 to 360 degrees with the specified step size.
  thetas = np.arange(0, 360, step=dtheta)
  
  ## Generate an array of radius values (radius_valus) from r_min to r_max with the specified step size 
  radius_valus = np.arange(r_min, r_max, step=delta_r)
  
  # Calculate Cos(theta) and Sin(theta) it will be required later
  cos_thetas = np.cos(np.deg2rad(thetas))
  sin_thetas = np.sin(np.deg2rad(thetas))
  

  # as the parametric equation of the circle is : x = x_center + r.cos(theta) & y = y_center + r.sin(theta)
  # we create candidate circles, a set of possible circle centers (x_center, y_center) and radii r.
  circle_candidates = []

  for r in radius_valus:  # The loop iterates over each radius value r specified within the range from r_min to r_max.
    for theta in range(num_thetas):  # For each radius, it iterates over a predefined number of theta values num_thetas, covering the entire circle.

      #For each combination of r and theta, we calculate the corresponding (x, y) coordinates using the parametric equations.
      # The candidate circles are represented as tuples (r, x, y).
      circle_candidates.append((r, int(r * cos_thetas[theta]), int(r * sin_thetas[theta])))
  


  # Hough Accumulator, we are using defaultdic instead of standard dict as this will initialize for key which is not 
  # aready present in the dictionary instead of throwing exception.
  accumulator = defaultdict(int)

  for edge_pixel in np.argwhere(edge_image != 0):
    y, x = edge_pixel[0], edge_pixel[1]

    # For each detected edge pixel, the algorithm proceeds to vote for potential circle candidates that pass through this pixel.
    for r, rcos_t, rsin_t in circle_candidates: # It iterates over the list of circle_candidates, which contains tuples representing (r, rcos_t, rsin_t) for each candidate circle.
        # For each candidate circle, it calculates the circle center (x_center, y_center) based on the parametric equations derived earlier.
        # The (x_center, y_center) is calculated by subtracting the pre-calculated offsets rcos_t and rsin_t from the current pixel (x, y).
        x_center = x - rcos_t
        y_center = y - rsin_t

        # After determining the circle center for each candidate, 
        # the algorithm increments the vote count in the accumulator for that specific circle center (x_center, y_center) and radius r.
        accumulator[(x_center, y_center, r)] += 1

  print("first loop ended")

  output_img = image.copy()


  # Output list of detected circles. A single circle would be a tuple of (x,y,r,threshold) 
  out_circles = []
  
  # Sort the accumulator based on the votes for the candidate circles 
  for candidate_circle, votes in sorted(accumulator.items(), key=lambda i: -i[1]):
    x, y, r = candidate_circle
    # For each candidate circle, we need the percentage of votes relative to the total number of theta values (num_thetas).
    # This percentage represents how many votes out of the total possible votes the candidate circle received.
    current_vote_percentage = votes / num_thetas
    if current_vote_percentage >= bin_threshold: 

      out_circles.append((x, y, r, current_vote_percentage))
      print(x, y, r, current_vote_percentage)
      
  
  # Post process the results, can add more post processing later.
  if post_process :
    pixel_threshold = 5
    postprocess_circles = []
    for x, y, r, v in out_circles:
      # Exclude circles that are too close of each other
      # all((x - xc) ** 2 + (y - yc) ** 2 > rc ** 2 for xc, yc, rc, v in postprocess_circles)
      # Remove nearby duplicate circles based on pixel_threshold
      if all(abs(x - xc) > pixel_threshold or abs(y - yc) > pixel_threshold or abs(r - rc) > pixel_threshold for xc, yc, rc, v in postprocess_circles):
        postprocess_circles.append((x, y, r, v))
    out_circles = postprocess_circles
  
    
  # Draw shortlisted circles on the output image
  for x, y, r, v in out_circles:
    output_img = cv2.circle(output_img, (x,y), r, (0,255,0), 2)
  
  return output_img, out_circles
