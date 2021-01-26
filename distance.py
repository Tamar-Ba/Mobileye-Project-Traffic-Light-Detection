import numpy as np
import SFM
from PIL import Image


def calc_tfl_distances(prev_container, curr_container, focal, pp, EM):

    curr_container = SFM.calc_TFL_dist(prev_container, curr_container, focal, pp, EM)
    return curr_container.distances


class FrameContainer(object):
    def __init__(self, img_path):
        self.img = np.array(Image.open(img_path))
        self.traffic_light = []
        self.traffic_lights_3d_location = []
        self.EM = []
        self.corresponding_ind = []
        self.valid = []
        self.traffic_distance = []


