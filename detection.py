import math
import keras
import numpy as np


def crop_tlf(img, col, row, size):
    '''
    :param img: img to crop
    :param col: the column to crop around
    :param row: the row to crop around
    :param size: the size of the cropped image
    :return: a cropped image around one tfl
    '''
    start_range_col = col - math.floor(size / 2) if col - math.floor(size / 2) > 0 else 0
    end_range_col = start_range_col + size if start_range_col + size < img.shape[1] else img.shape[1]
    if end_range_col == img.shape[1]:
        start_range_col = end_range_col - size

    start_range_row = row - math.floor(size / 2) if row - math.floor(size / 2) > 0 else 0
    end_range_row = start_range_row + size if start_range_row + size < img.shape[0] else img.shape[0]
    if end_range_row == img.shape[0]:
        start_range_row = end_range_row - size

    return img[start_range_row:end_range_row, start_range_col: end_range_col]


def crop_img(img, pos, size):
    '''
    :param img: img to crop
    :param pos: list of points to crop around
    :param size: the size of the cropped image
    :return: a list of cropped images
    '''
    tlf_imgs = []
    for tlf in pos:
        cp = crop_tlf(img, tlf[0], tlf[1], size)
        # print(cp.shape)
        tlf_imgs.append(crop_tlf(img, tlf[0], tlf[1], size))
    return tlf_imgs


def open_model(json_filename, h5_filename):
    with open(json_filename, 'r') as j:
        loaded_json = j.read()

    # load the model architecture:
    loaded_model = keras.models.model_from_json(loaded_json)
    # load the weights:
    loaded_model.load_weights(h5_filename)
    return loaded_model


def predict(positions, loaded_model, points):
    traffic_lights = []
    for i, img in enumerate(positions):
        img = img.reshape([-1] + [81, 81] + [3])
        l_predictions = loaded_model.predict(img)
        l_predicted_label = np.argmax(l_predictions, axis=-1)

        if l_predicted_label[0] == 1:
            traffic_lights.append(points[i])

    return traffic_lights


def crop_and_predict(img, candidates, model):
    croped_img = crop_img(img, candidates, 81)
    return predict(croped_img, model, candidates)
