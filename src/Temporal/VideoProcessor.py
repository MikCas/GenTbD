from Temporal.Temporal import Temporal
from Detection.Detectors.Detector import Detector
from Tracking.Tracker import Tracker

from typing import Optional
import cv2

class SimpleVideoProcessor(Temporal):
    """
    A simple video processor based on the AbstractTemporalSystem class.
    """

    def __init__(self, *args, detector: Detector, tracker: Tracker, **kwargs):
        super().__init__(*args, **kwargs)
        self._detector: Detector = detector
        self._tracker: Tracker = tracker
        self._cap: cv2.VideoCapture = self.create_video_capture(self._video_path)
    
    def process_image(self, image: cv2.Mat) -> None:
        """
        Processes an image.

        Args:
            image (cv2.Mat): The image to process.
        """

        # Perform detection 
        detections = self._detector.detect(image)
        # self._detector.display_detections(detections, image)

        # Perform tracking
        self._tracker.update(self._timestep, detections)
        self._tracker.display_tracks(image, mode='id')

        # Display image
        cv2.imshow('Processed Frame', image)

    def process(self) -> None:
        """
        Processes the video file frame-by-frame, at each frame applying the process_frame method. 
        """
        while True:
            image = self.continue_video()
            if image is None: break        # End of video or error reading frame

            self.update_timestep()
            self.process_image(image)

            # Handle key events
            if self._continuous_mode:
                key = cv2.waitKey(1) & 0xFF
            else:
                key = cv2.waitKey(0) & 0xFF

            self.save_event(key, image)     # Press 's'
            self.toggle_event(key)          # Press 'c' 
            if self.quit_event(key): break  # Press 'q'     
