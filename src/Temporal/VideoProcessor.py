from Temporal.Temporal import Temporal
from Detecting.Detectors.Detector import Detector
from Tracking.Trackers.Tracker import Tracker

import logging
import cv2

class SimpleVideoProcessor(Temporal):
    """
    A simple video processor that perform detection and tracking on each frame of a video.

    Inherits from the Temporal class.

    Attributes:
        - detector (Detector): The object detector to use.
        - tracker (Tracker): The object tracker to use.
        - draw_mode (str): The mode for drawing the detections and tracks. Options are 'state' or 'id'.
    """

    ##### SETUP #####
    def __init__(self, *args, detector: Detector, tracker: Tracker, draw_mode: str = 'state', **kwargs):
        super().__init__(*args, **kwargs)
        self._detector: Detector = detector
        self._tracker: Tracker = tracker
        self._draw_mode: str = draw_mode
        self._cap: cv2.VideoCapture = self.create_video_capture(self._video_path)
    
    ##### VIDEO PROCESSING #####
    def process_image(self, image: cv2.Mat) -> None:
        """
        Performs detection followed by tracking on the given image. The image with detections/tracks is then displayed. 

        Args:
            image (cv2.Mat): The image to process.
        """
        
        # Perform detection 
        self.log(logging.INFO, "|| DETECTION")  
        detections = self._detector.detect(image)
        # self._detector.draw_detections(detections, image)

        # Perform tracking
        self.log(logging.INFO, "|| TRACKING")
        self._tracker.update(self._timestep, detections)
        self._tracker.draw_tracks(image, mode=self._draw_mode)

        # Display image
        cv2.imshow('Processed Frame', image)

    def process(self) -> None:
        """
        Processes the video file frame-by-frame, at each frame applying the 'process_image' method, while also handling user input events.
        """
        if self._logger: print()
        self.log(logging.INFO, f"|| BEGIN VIDEO PROCESSING")

        # Iterate through thte vide, reading a frame at each iteration
        while True:
            image = self.continue_video()
            if image is None: break        # End of video or error reading frame

            # Perform detection and tracking on the image 
            self.process_image(image)

            # Handle key events
            if self._continuous_mode:
                key = cv2.waitKey(1) & 0xFF
            else:
                key = cv2.waitKey(0) & 0xFF

            video_name = self._video_path.split('/')[-1]      # Extract video file name
            self.save_event(key, image, video_name)           # Press 's'
            self.toggle_event(key)                            # Press 'c' 
            if self.quit_event(key): break                    # Press 'q'   
        
            if self._logger: print() # Skip line for better readability

        self.terminate()
