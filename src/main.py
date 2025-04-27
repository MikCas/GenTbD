from detecting.detectors.Detector_ONNX_YOLO7 import Detector
from tracking.Tracker import Tracker
from VideoProcessor import VideoProcessor

import logging
import os

def setup_logger():
    logger = logging.getLogger('System_Logger')
    logger.setLevel(logging.DEBUG) 
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s') # LOG MESSAGE FORMAT
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    return logger

if __name__ == '__main__':  
    
    # LOGGER
    logger = setup_logger()

    # VIDEO FILE PATH
    video_file = 'video1.mp4'
    video_path = os.path.join(os.getcwd(), 'data', video_file)

    # DETECTOR 
    model_name = 'yolov7_640x640.onnx'
    confidence_threshold = 0.1
    iou_threshold = 0.5
    classes = [0]
    detector = Detector(model_name, confidence_threshold=confidence_threshold, iou_threshold=iou_threshold, classes=classes, logger=logger)

    # TRACK PARAMETERS
    # Track.set_max_lost_count(15)
    # Track.set_trajectory_max_size(20)
    
    # TRACKER
    detection_threshold = 0.3
    activation_threshold = 0.3
    match_threshold = 0.2
    tracker = Tracker(detection_threshold=detection_threshold, activation_threshold=activation_threshold, match_threshold=match_threshold, logger=logger)

    # PROCESS VIDEO AND START TRACKING
    # display_tracks_mode = 'state' # DISPLAY THE STATE OF EACH TRACK WITH A CORRESPONDING COLOUR - MATCHED, LOST, NEW
    display_tracks_mode = 'id' # DISPLAY THE UNIQUE ID OF EACH TRACK
    videoProcessor = VideoProcessor(detector, tracker, video_path=video_path, continuous_mode=False, display_tracks_mode=display_tracks_mode, logger=logger)
    videoProcessor.process()

    # NOT USING STREAM PROCESSING FOR NOW
    # streamProcessor = VideoProcessor(isStream=True)
    # streamProcessor.process()


