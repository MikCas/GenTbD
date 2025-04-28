"""
Main module for processing videos using object detection and tracking.

This script initializes the logger, sets up the detector and tracker, and processes
a video file to detect and track objects.
"""

from detecting.detectors.Detector_ONNX_YOLO7 import Detector
from tracking.Tracker import Tracker
from Temporal.VideoProcessor import SimpleVideoProcessor

import logging
import os

def setup_logger():
    """
    Sets up the logger for the application.

    Returns:
        logging.Logger: Configured logger instance.
    """
    logger = logging.getLogger('System_Logger')
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')  # Log message format
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    return logger

if __name__ == '__main__':
    """
    Main entry point of the application.

    This script performs the following:
    - Initializes the logger.
    - Sets up the video file path.
    - Configures the object detector and tracker.
    - Processes the video file to detect and track objects.
    """

    logger = setup_logger()

    vp = SimpleVideoProcessor(video_path='data/video1.mp4', logger=logger)
    vp.process()
    
    # # LOGGER
    # logger = setup_logger()

    # # VIDEO FILE PATH
    # video_file = 'video1.mp4'
    # video_path = os.path.join(os.getcwd(), 'data', video_file)

    # # DETECTOR
    # model_name = 'yolov7_640x640.onnx'
    # confidence_threshold = 0.1
    # iou_threshold = 0.5
    # classes = [0]
    # detector = Detector(
    #     model_name,
    #     confidence_threshold=confidence_threshold,
    #     iou_threshold=iou_threshold,
    #     classes=classes,
    #     logger=logger
    # )

    # # TRACKER
    # detection_threshold = 0.3
    # activation_threshold = 0.3
    # match_threshold = 0.2
    # tracker = Tracker(
    #     detection_threshold=detection_threshold,
    #     activation_threshold=activation_threshold,
    #     match_threshold=match_threshold,
    #     logger=logger
    # )

    # # PROCESS VIDEO AND START TRACKING
    # display_tracks_mode = 'id'  # Display the unique ID of each track
    # videoProcessor = VideoProcessor(
    #     detector,
    #     tracker,
    #     video_path=video_path,
    #     continuous_mode=False,
    #     display_tracks_mode=display_tracks_mode,
    #     logger=logger
    # )
    # videoProcessor.process()


