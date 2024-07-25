import cv2
import numpy as np
import matplotlib.pyplot as plt


# ---------------------------------------------------------------- Gaussian Noise -----------------------------------------------------------------------------------
def apply_Gassian_noise(image, scale):
    """Return image with noise.
    Parameters
    ----------
    image : ndarray, image to which noise is applied.
    scale : positive float, standard deviation of Gaussian 
        noise distribution.
    Returns:
    - noisy: ndarray, image with Gaussian noise.
    """

    # scale = scale /50 #come from the slider

    image = np.asarray(image)
    # Create a Gaussian noise array.
    noise = np.random.normal(loc=0, scale=scale, size=image.shape)
    # Apply the noise array.
    noisy = image + noise
    # Transform to integer type.
    noisy = noisy.astype(np.int32)
    # Clip the values to RGB bounds.
    noisy = noisy.clip(0, 255)


    return noisy

# -----------------------------------------------------------------------------Salt and Pepper Noise ---------------------------------------------------------
def apply_Salt_Pepper_noise(img, noise_percentage):
    """
    Apply salt-and-pepper noise to an image.

    Parameters:
    - img: ndarray, input image (2D or 3D array).
    - noise_percentage: float, percentage of pixels to be affected by noise.

    Returns:
    - img_noised: ndarray, image with salt-and-pepper noise.
    """
    noise_percentage = noise_percentage / 100  # come from the slider

    # Get the image size (number of pixels in the image).
    img_size = img.size

    # Determine the size of the noise based on the noise percentage
    noise_size = int(noise_percentage*img_size)

    # Randomly select indices for adding noise.
    random_indices = np.random.choice(img_size, noise_size)

    # Create a copy of the original image that serves as a template for the noised image.
    img_noised = img.copy()

    # Create a noise list with random placements of min and max values of the image pixels.
    noise = np.random.choice([img.min(), img.max()], noise_size)

    # Replace the values of the templated noised image at random indices with the noise,
    # to obtain the final noised image.
    img_noised.flat[random_indices] = noise

    return img_noised



# ---------------------------------------------------------------------------- Uinform Noise -----------------------------------------------------------------
def apply_Uniform_noise(img, noise_percentage):
    """
    Applies uniform noise to the input image.

    Parameters:
    - img (numpy.ndarray): Input image (grayscale or color).

    Returns:
    - numpy.ndarray: Noisy image with uniform noise added.
    """
    # Ensure the image is in the correct data type
    img = img.astype(np.float64)

    start_range_of_noise = -2 * noise_percentage  # the range for the noise
    end_range_of_noise = 2 * noise_percentage

    # Generate uniform noise in the specified range
    noise = np.random.uniform(start_range_of_noise, end_range_of_noise, size=img.shape)
    
    # Apply the noise to the image
    noisy = img + noise

    # Clip the values to the valid range [0, 255]
    noisy = np.clip(noisy, 0, 255).astype(np.uint8)

    return noisy


# img = cv2.imread("./Hager/images/1.jpg")
# print(img.shape)
# noisey = apply_Gassian_noise(img, 100)
# plt.imshow( noisey)
# plt.show()