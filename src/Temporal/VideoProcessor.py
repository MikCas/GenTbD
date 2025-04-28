from Temporal.AbstractTemporalSystem import AbstractTemporalSystem
import cv2
from typing import Optional

class SimpleVideoProcessor(AbstractTemporalSystem):
    """
    A subclass of the AbstractTemporalSystem for processing video files. 
    A simple video processor which reads a video file path and processes each frame.
    The video file can be processed in continuous mode or step-by-step mode.
    """
        
    def __init__(self, *args, **kwargs):
        """
        Initializes the SimpleVideoProcessor.
       """
        super().__init__(*args, **kwargs)
    
    def process_frame(self, frame: cv2.Mat) -> None:
        """
        Processes a single frame by applying a simple operation.

        Args:
            frame (cv2.Mat): The video frame to process.
        """
        processed_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        cv2.imshow('Processed Frame', processed_frame)

    def process(self) -> None:
        """
        Processes the video file frame-by-frame, at each frame applying the process_frame method. 
        """
        while True:
            result: bool
            frame: Optional[cv2.Mat]
            result, frame = self._cap.read()

            # Validate the frame
            if not self.validate_frame(result, frame):
                break

            # Update the timestep
            self.update_timestep()

            # Process the frame
            self.process_frame(frame)

            # Handle key events
            if self._continuous_mode:
                key = cv2.waitKey(1) & 0xFF
            else:
                key = cv2.waitKey(0) & 0xFF

            self.save_event(key, frame)     # Press 's'
            self.toggle_event(key)          # Press 'c' 
            if self.quit_event(key): break  # Press 'q'     
