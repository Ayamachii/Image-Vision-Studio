import numpy as np
import cv2

def optimal_threshold(image):
    """
    Apply global thresholding To The given grayscale image
    Parameters:
    image: NumPy Array of The Source Grayscale Image
    Return: NumPy Array binary thresholded image
    """
    enhanced_image = cv2.convertScaleAbs(image, alpha=1, beta=0)  # needed to be adjusted for some images
    filtered_image = cv2.GaussianBlur(enhanced_image, (3, 3), 0)
    # Get the height and width of the image
    height, width = filtered_image.shape
    # Get the indices of the four corners
    corners = [(0, 0), (0, width - 1), (height - 1, 0), (height - 1, width - 1)]
    # Initialize the lists for pixel values
    background = [filtered_image[y, x] for (y, x) in corners]
    objects = [filtered_image[y, x] for y in range(height) for x in range(width) if (y, x) not in corners]
    u_b = np.mean(background)
    u_o = np.mean(objects)
    # Initial threshold
    T = ((u_b + u_o) / 2)
    print(f"initial T = {T} ")
    while True:
        background = filtered_image[filtered_image < T]
        objects = filtered_image[filtered_image > T]
        if len(background) == 0:
            print(f"Empty background!")
            break
        if len(objects) == 0:
            print(f"Empty objects!")
            break
        u_b = np.mean(background)
        u_o = np.mean(objects)
        T_new = (u_b + u_o) / 2
        print(f"T_new = {T_new}")
        if T_new == T:
            break
        T = T_new
    print(f"final T = {T}")
    # Apply thresholding using numpy indexing
    binary_image = np.where(filtered_image < T, 0, 255).astype(np.uint8)

    return binary_image

def local_optimal_thresholding(image, num_x_sections, num_y_sections):
    """
    Apply Local Thresholding To The Given Grayscale Image
    Parameters:
    image: NumPy Array of The Source Grayscale Image
    num_x_sections, num_y_sections: Number of Regions To Divide The Image To
    Return: NumPy Array Thresholded Image
    """
    src = np.copy(image)
    if len(src.shape) > 2:
        src = cv2.cvtColor(src, cv2.COLOR_RGB2GRAY)
    else:
        pass

    max_y, max_x = src.shape
    output_image = np.zeros((max_y, max_x))
    y_step = max_y // num_x_sections
    x_step = max_x // num_y_sections
    x_range = []
    y_range = []
    for i in range(0, num_x_sections):
        x_range.append(x_step * i)

    for i in range(0, num_y_sections):
        y_range.append(y_step * i)

    x_range.append(max_x)
    y_range.append(max_y)
    for x in range(0, num_x_sections):
        for y in range(0, num_y_sections):
            output_image[y_range[y]:y_range[y + 1], x_range[x]:x_range[x + 1]] = optimal_threshold(src[y_range[y]:y_range[y + 1], x_range[x]:x_range[x + 1]])
    return output_image

def builtin_optimal_threshold(image_path):
    # Load the image in grayscale mode
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    # filtered_image = cv2.medianBlur(image, 5)
    filtered_image = cv2.GaussianBlur(image, (3, 3), 0)
    ret, th1 = cv2.threshold(filtered_image, 148, 255, cv2.THRESH_BINARY)
    return th1

def builtin_local_thresholding(self, image_path):
    # Load the image in grayscale mode
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    # filtered_image = cv2.medianBlur(image, 3)
    th2 = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 45, 2)
    return th2
