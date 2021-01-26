import math
import scipy
from scipy.signal import convolve
import numpy as np
import scipy.ndimage as ndimager


def calc_find_tfl(c_image, kernal, min_threshold=0, max_threshold=math.inf):
    convoloved = convolve(c_image, kernal, 'same')
    max_filtered_img = scipy.ndimage.maximum_filter(convoloved, size=100)
    max_filtered_copy = max_filtered_img.copy()
    mask = ((max_filtered_copy == convoloved) & (max_filtered_copy > min_threshold) & (
                max_filtered_copy < max_threshold))
    positions = np.where(mask == True)
    return positions[1], positions[0]


def find_tfl_lights(c_image: np.ndarray, **kwargs):
    """
    Detect candidates for TFL lights. Use c_image, kwargs and you imagination to implement
    :param c_image: The image itself as np.uint8, shape of (H, W, 3)
    :param kwargs: Whatever config you want to pass in here
    :return: 4-tuple of x_red, y_red, x_green, y_green
    """
    threshold = 652350
    kernel = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                       [0, 0, 0, 0, -255, -255, -255, -255, 0, 0, 0, 0],
                       [0, 0, 0, -255, 255, 255, 255, 255, -255, 0, 0, 0],
                       [0, 0, -255, 255, 255, 255, 255, 255, 255, -255, 0, 0],
                       [0, -255, 255, 255, 255, -255, -255, 255, 255, 255, -255, 0],
                       [0, -255, 255, 255, 255, -255, -255, -255, 255, 255, -255, 0],
                       [0, -255, 255, 255, -255, -255, -255, 255, 255, 255, -255, 0],
                       [0, 0, -255, 255, 255, 255, 255, 255, 255, 255, -255, 0],
                       [0, 0, -255, 255, 255, 255, 255, 255, 255, -255, 0, 0],
                       [0, 0, 0, -255, -255, 255, 255, 255, -255, 0, 0, 0],
                       [0, 0, 0, 0, 0, 0, -255, -255, -255, 0, 0, 0]])  # 12x11
    red_img = c_image[:, :, 0]
    red_x, red_y = calc_find_tfl(red_img, kernel, threshold)
    green_img = c_image[:, :, 1]
    threshold = 842350
    green_x, green_y = calc_find_tfl(green_img, kernel, threshold)

    return red_x, red_y, green_x, green_y