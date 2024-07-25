import cv2
import numpy as np
import matplotlib.pyplot as plt
from functions.add_noise import *


def convert_to_ferq_domain(img: np.ndarray):
    """
    Convert the image to the frequency domain using the 2D Fast Fourier Transform (FFT).

    Parameters:
        img (np.ndarray): Input image.

    Returns:
        Tuple[np.ndarray, np.ndarray]: Tuple containing the frequency spectrum (F) and the shifted spectrum (Fshift).
    """

    F = np.fft.fft2(img, axes=(0, 1))

    Fshift = np.fft.fftshift(F, axes=(0, 1))

    return (F, Fshift)


def calc_filter_H(img):
    """
    Calculate the filter H for low-pass filtering.

    Parameters:
        img (np.ndarray): Input image.

    Returns:
        np.ndarray: Filter H for low-pass filtering.
    """

    rows, columns = img.shape
    H = np.zeros((rows, columns), dtype=np.float32)
    # sets a cutoff frequency D0
    D0 = 50
    for u in range(rows):
        for v in range(columns):
            D = np.sqrt((u - rows / 2) ** 2 + (v - columns / 2) ** 2)
            if D <= D0:
                H[u, v] = 1
            else:
                H[u, v] = 0
    return H

# ---------------------------------------------------------------- Low Pass Filter ----------------------------------------------------------------------------
def LPF(img, is_gray):
    """
    Apply a low-pass filter to the input image.

    Parameters:
        img (np.ndarray): Input image.
        is_gray (bool): Flag indicating whether the image is grayscale.

    Returns:
        np.ndarray: Low-pass filtered image.
    """

    if not is_gray:
        H = np.stack([calc_filter_H(img[:,:,0])] * 3, axis=-1)  # Duplicate H for each color channel
    else:
        H = calc_filter_H(img)

    # Ideal Low Pass Filtering
    F, Fshift = convert_to_ferq_domain(img)
    Gshift = Fshift * H
    
    # Inverse Fourier Transform
    G = np.fft.ifftshift(Gshift, axes=(0, 1))
    g = np.abs(np.fft.ifft2(G, axes=(0, 1)))

    # Normalize pixel values to range [0, 255]
    g_normalized = (g - np.min(g)) / (np.max(g) - np.min(g)) * 255
    # g_normalized = cv2.normalize(g, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    return g_normalized.astype(np.uint8)
    


# -------------------------------------------------------------------- High Pass Filter --------------------------------------------------------------------------
def HPF(img, is_gray):
    """
    Apply a high-pass filter to the input image.

    Parameters:
        img (np.ndarray): Input image.
        is_gray (bool): Flag indicating whether the image is grayscale.

    Returns:
        np.ndarray: High-pass filtered image.
    """

    if not is_gray:
        H = 1 - np.stack([calc_filter_H(img[:,:,0])] * 3, axis=-1)  # Duplicate H for each color channel
    else:
        H = 1 - calc_filter_H(img)

    # Ideal High Pass Filtering
    F, Fshift = convert_to_ferq_domain(img)
    Gshift = Fshift * H

    # Inverse Fourier Transform
    G = np.fft.ifftshift(Gshift, axes=(0, 1))
    g = np.abs(np.fft.ifft2(G, axes=(0, 1)))

    # Normalize pixel values to range [0, 255]
    g_normalized = (g - np.min(g)) / (np.max(g) - np.min(g)) * 255
    # g_normalized = cv2.normalize(g, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    return g_normalized.astype(np.uint8)


# fig, axes = plt.subplots(1, 3, figsize=(15, 5))
# img = cv2.imread("./Hager/images/6.jpg")
# print(img.shape)
# noisey = apply_Gassian_noise(img, 50)
# lpf_filterd = LPF(noisey)
# hpf_filterd = HPF(noisey)
# axes[0].imshow(noisey)
# axes[1].imshow(lpf_filterd)
# axes[2].imshow(hpf_filterd)

# plt.show()
