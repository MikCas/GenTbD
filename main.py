import logging
import os

from Detector import Detector
from Tracker import Tracker
from VideoProcessor import VideoProcessor

from Assignment import linear_assignment

def setup_logger():
    # Set up loggingc
    logger = logging.getLogger('System_Logger')
    logger.setLevel(logging.DEBUG) 

    # Define the format of log messages
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

    # StreamHandler sends log messages to the console
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    return logger

if __name__ == '__main__':  
    
    # LOGGER
    logger = setup_logger()

    # VIDEO FILE PATH
    video_file = 'TownCent.mp4'
    video_path = os.path.join(os.getcwd(), 'data', video_file)

    # DETECTOR 
    model_name = 'yolo11n'
    detector = Detector(model_name, logger)

    # TRACKER 
    tracker = Tracker(logger=logger)

    # PROCESS VIDEO AND START TRACKING
    videoProcessor = VideoProcessor(detector, tracker, video_path=video_path, logger=logger)
    videoProcessor.process()

    # streamProcessor = VideoProcessor(isStream=True)
    # streamProcessor.process()
