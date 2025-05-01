from Detecting.Detectors.Detector_ONNX_YOLO7 import YOLOv7ONNX
from Tracking.Trackers.SimpleTracker import SimpleTracker
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
    1. Initialises the logger.
    2. Configures the object detector.
    3. Configures the tracker.
    4. Sets up the video processor.
    5. Processes the video file to detect and track objects.q
    """

    ##### 1. LOGGER #####
    logger = setup_logger()

    ##### 3. DETECTOR #####
    model_path = 'models/yolov7_640x640.onnx'
    confidence_threshold = 0.1
    iou_threshold = 0.5
    classes = [0]
    detector = YOLOv7ONNX(
        model_path,
        confidence_threshold=confidence_threshold,
        iou_threshold=iou_threshold,
        classes=classes,
        logger=logger
    )

    ##### 4. TRACKER #####
    detection_threshold = 0.3
    creation_threshold = 0.3
    match_thresholds = [0.4, 0.2, 0.2]
    activation_threshold = 10
    deactivation_threshold = 15

    tracker = SimpleTracker( 
        detection_threshold=detection_threshold,
        creation_threshold=creation_threshold,
        activation_threshold=activation_threshold, 
        deactivation_threshold=deactivation_threshold,
        match_thresholds=match_thresholds,
        logger=logger
    )

    ##### 4. VIDEO SETUP #####
    video_file = 'video1.mp4'
    video_path = os.path.join(os.getcwd(), 'data', video_file)
    draw_mode = 'state'  # Options: 'state', 'id', 'none'   

    vp = SimpleVideoProcessor(video_path=video_path, 
                              detector=detector, 
                              tracker=tracker, 
                              draw_mode=draw_mode,
                              logger=logger)
    
    vp.process()


