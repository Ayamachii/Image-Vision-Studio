import numpy as np
import pyqtgraph as pg

# BGR
def get_channels(img):
    """"
    Gets the image as an ndarray in the format BGR and separates its channels.

    Args:
        img (ndarray): image in the format BGR of dimension (img_height, img_width, 3)

    Returns:
        A tuple of the three channels separated
    """
    return(img[:, :, 0], img[:, :, 1], img[:, :, 2])
    
def get_frequencies(channel_array):
    """"
    Gets a channel of the image and returns the frequency 
    of each pixel values and the bin labels to draw its histogram.

    Args:
        channel_array (ndarray): a 2d numpy array containing pixel values

    Returns:
        A tuple of the three channels separated
    """
    # flatten 2d array to a 1d array of values
    channel_array = channel_array.flatten()
    return np.histogram(channel_array, bins=256)
    
def get_cumulative_frequencies(frequencies):
    """"
    Gets a channel of the image and returns the frequency 
    of each pixel values and the bin labels to draw its histogram.

    Args:
        frequencies (ndarray): a 1d numpy array containing the frequency of each pixel value

    Returns:
        A numpy array of the cumulative frequencies for every pixel frequency and the previous pixel frequencies
    """
    cum_frequencies = [frequencies[0]]
    for index in range(1, len(frequencies)):
        cum_frequencies.append(frequencies[index] + cum_frequencies[index - 1])
    return np.array(cum_frequencies)
    
def to_greyscale(red_ch, green_ch, blue_ch):
    """"
    Gets the 3 channels of a colored image and converts it to grey scale

    Args:
        red_ch (ndarray): a 2d numpy array containing pixel values of the red channel (img_width, img_height)
        green_ch (ndarray): a 2d numpy array containing pixel values of the green channel (img_width, img_height)
        blue_ch (ndarray): a 2d numpy array containing pixel values of the blue channel (img_width, img_height)

    Returns:
        grey_ch (ndarray): A 2d array of one channel of pixel values 
    """
    grey_ch = 0.2989 * red_ch + 0.5870 * green_ch + 0.1140 * blue_ch
    return grey_ch

