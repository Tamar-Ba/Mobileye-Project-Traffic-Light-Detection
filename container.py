from frame import Frame


class Container:
    def __init__(self, current_frame: Frame = None, prev_frame: Frame = None):
        self.current_frame = current_frame
        self.prev_frame = prev_frame

    def set_frame(self, frame):
        self.prev_frame = self.current_frame
        self.current_frame = frame

