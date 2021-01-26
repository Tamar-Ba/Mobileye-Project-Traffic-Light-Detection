import matplotlib.pyplot as plt
import attention
import detection
import distance
from container import Container
from frame import Frame


class TflMan:

    def __init__(self, pp, focal, json_filename, h5_filename):
        self.pp = pp
        self.focal = focal
        self.model = detection.open_model(json_filename, h5_filename)
        self.container = Container()

    def attention(self):
        red_x, red_y, green_x, green_y = attention.find_tfl_lights(self.container.current_frame.image)
        self.container.current_frame.candidates = list(zip(red_x, red_y))

    def detection(self):
        self.container.current_frame.traffic_lights = detection.crop_and_predict(self.container.current_frame.image,
                                                                                 self.container.current_frame.candidates,
                                                                                 self.model)

    def calc_distance(self, EM):
        if self.container.prev_frame is not None:
            self.container.current_frame.distances = distance.calc_tfl_distances(self.container.prev_frame,
                                                                                 self.container.current_frame,
                                                                                 self.focal, self.pp, EM)

    def run_frame(self, image, EM):
        frame = Frame(image)
        self.container.set_frame(frame)
        # run attention on image- part 1
        self.attention()
        # run detection tfl based on neural net- part 2
        self.detection()
        # run distance - part 3
        self.calc_distance(EM)
        self.visualize()


    def visualize(self, frame):

        fig, (sec_part_1, sec_part_2, sec_part_3) = plt.subplots(3, sharex=True, sharey=True, figsize=(12, 6))
        fig.suptitle(f'Frame #{frame.frame_id}: image name')

        sec_part_1.imshow(frame.image)
        x = [pt[0] for pt in frame.candidates]
        y = [pt[1] for pt in frame.candidates]
        sec_part_1.plot(x, y, 'ro', marker='o', color='r', markersize=2)

        sec_part_2.imshow(frame.image)
        x = [pt[0] for pt in frame.traffic_lights]
        y = [pt[1] for pt in frame.traffic_lights]
        sec_part_2.plot(x, y, 'ro', marker='o', color='r', markersize=2)

        sec_part_3.imshow(frame.image)
        sec_part_3.plot(x, y, 'ro', marker='o', color='r', markersize=2)

        if frame.valid != []:
            for i in range(len(frame.traffic_lights)):

                if frame.valid[i]:
                    sec_part_3.text(frame.traffic_lights[i][0], frame.traffic_lights[i][1],
                                    r'{0:.1f}'.format(frame.distances[i]), color='r')

        plt.show()
