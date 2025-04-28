from abc import ABC, abstractmethod
import cv2
import logging
from typing import Optional

class AbstractTemporalSystem(ABC):
    """
    Abstract base class for processing temporal data (videos). 
        - Provides framework for processing videos frame-by-frame  
        - At each frame, applies the process_frame method.
        - Provides event handlers for saving frames, toggling between continuous and step-by-step modes, and quitting the video.

    Attributes:
        video_path (str): Path to the video file.
        timestep (int): Current timestep of temporal data. 
        continuous_mode (bool): Flag to indicate if the video is processed in continuous mode (True) or step-by-step mode (False).
        cap (cv2.VideoCapture): Video capture object for reading video frames.
        logger (logging.Logger): Logging messages
    """

    ### ATTRIBUTES
    @property
    def timestep(self) -> int: return self._timestep

    @timestep.setter
    def timestep(self, value: int) -> None:
        self._timestep = value
    
    def update_timestep(self) -> None:
        """
        Updates the current timestep based on the video capture position.
        """
        self._timestep: int = int(self._cap.get(cv2.CAP_PROP_POS_FRAMES))
        self._logger.info(f"TIMESTEP {self._timestep}")

    ### SETUP METHODS
    def create_video_capture(self, video_path: str) -> cv2.VideoCapture:
        cap: cv2.VideoCapture = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            self.logger.error(f"----ERROR COULD NOT OPEN VIDEO CAPTURE: {video_path}")
            raise ValueError("Video source could not be opened.")
        return cap
    
    def create_source_capture(self) -> cv2.VideoCapture:
        cap: cv2.VideoCapture = cv2.VideoCapture(0)
        if not cap.isOpened():
            self.logger.error("----ERROR COULD NOT OPEN STREAM")
            raise ValueError("Stream source could not be opened.")
        return cap

    def __init__(self, video_path: str, logger: Optional[logging.Logger] = None):
        """
        Args:
            video_path (str): Path to the video file.
            logger (Optional[logging.Logger]): Logger instance for logging messages (optional).
        """
        self._video_path: str = video_path
        self._timestep: int = 0
        self._continuous_mode: bool = False # Start in step-by-step mode
        self._cap: cv2.VideoCapture = self.create_video_capture(video_path)
        self._logger: logging.Logger = logger if logger else logging.getLogger(__name__)

    def cleanup(self) -> None:
        if self._cap is not None:
            self._cap.release()
        cv2.destroyAllWindows()

    ### EVENT HANDLERS
    def save_event(self, key: int, frame: cv2.Mat) -> bool:
        """
        Event handler for saving the current frame to a file.

        Args:
            key (int): The key code of the pressed key.
            frame (cv2.Mat): The current video frame.

        Returns:
            bool: True if the save event is triggered, False otherwise.
        """
        if key == ord('s'):
            filename = f"frame_{self._timestep}.jpg"
            cv2.imwrite(filename, frame)
            self._logger.info(f"SAVED FRAME {self._timestep} AS {filename}")
            return True
        return False
    
    def toggle_event(self, key: int) -> bool:
        """
        Event handler for toggling between continuous and step-by-step modes.
        continuous mode - the video are automatically processed without user input.
        step-by-step mode - the video is processed frame by frame, waiting for user input to advance to the next frame.

        Args:
            key (int): The key code of the pressed key.

        Returns:
            bool: True if the toggle event is triggered, False otherwise.
        """
        if key == ord('c'):
            self._continuous_mode = not self._continuous_mode
            mode = "CONTINUOUS" if self._continuous_mode else "STEPPED"
            self._logger.info(f"TOGGLED TO {mode} MODE")
            return True
        return False
    
    def quit_event(self, key: int) -> bool:
        """
        Event handler for quitting the video processing.

        Args:
            key (int): The key code of the pressed key.

        Returns:
            bool: True if the quit event is triggered, False otherwise.
        """
        if key == ord('q'):
            self._logger.info("QUIT VIDEO")
            return True 
        return False
    
    ### FRAME VALIDATION
    def validate_frame(self, result: bool, frame: Optional[cv2.Mat]) -> bool:
        """
        Validates if a frame was successfully read from the video source.

        Args:
            result (bool): Indicates if the frame was successfully read.
            frame (Optional[cv2.Mat]): The frame read from the video source.

        Returns:
            bool: True if the frame is valid, False if the video has ended.

        Raises:
            RuntimeError: If there is an error reading the frame.
        """
        if not result or frame is None:
            if self._cap.get(cv2.CAP_PROP_POS_FRAMES) >= self._cap.get(cv2.CAP_PROP_FRAME_COUNT):
                self._logger.info("VIDEO FINISHED")
                return False
            else:
                self._logger.error("ERROR READING FRAME")
                raise RuntimeError("Error reading frame from video source.")
        return True

    ### ABSTRACT METHODS (TO BE IMPLEMENTED BY SUBCLASSES)
    @abstractmethod
    def process_frame(self, frame: cv2.Mat) -> None:
        """
        Abstract method to process a single frame. Must be implemented by subclasses.

        Args:
            frame (cv2.Mat): The video frame to process.
        """
        pass

    @abstractmethod
    def process(self) -> None:
        """
        Abstract method to process the video. Must be implemented by subclasses.
        """
        pass


