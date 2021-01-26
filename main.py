from controller import Controller

if __name__ == '__main__':
    controller = Controller('frames lists/listOfFrames_dusseldorf_000049.pls.txt')
    # load the playlist file
    controller.run_all_frames()


