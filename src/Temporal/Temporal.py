from abc import ABC, abstractmethod
import cv2
import logging
from typing import Optional

class Temporal(ABC):
    """
    Abstract base class for processing temporal data (videos). 
        - Provides framework for processing videos image-by-image  
        - At each image, applies the process_image method.
        - Provides event handlers for saving images, toggling between continuous and step-by-step modes, and quitting the video.

    Attributes:
        video_path (str): Path to the video file.
        timestep (int): Current timestep of temporal data. 
        continuous_mode (bool): Flag to indicate if the video is processed in continuous mode (True) or step-by-step mode (False).
        cap (cv2.VideoCapture): Video capture object for reading video images.
        logger (logging.Logger): Logging messages
    """

    ##### ATTRIBUTES #####
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

    ##### SETUP #####
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
        self._logger: logging.Logger = logger if logger else logging.getLogger(__name__)

    def cleanup(self) -> None:
        if self._cap is not None:
            self._cap.release()
        cv2.destroyAllWindows()

    ##### FUNCTIONS #####
    def save_event(self, key: int, image: cv2.Mat) -> bool:
        """
        Event handler for saving the current image to a file.

        Args:
            key (int): The key code of the pressed key.
            image (cv2.Mat): The current video image.

        Returns:
            bool: True if the save event is triggered, False otherwise.
        """
        if key == ord('s'):
            filename = f"image_{self._timestep}.jpg"
            cv2.imwrite(filename, image)
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
    
    def continue_video(self) -> Optional[cv2.Mat]:
        """
        Reads the next frame from the video source.

        Returns:
            Optional[cv2.Mat]: The next frame read from the video source.
        """

        # Read the next frame from the video source
        result: bool
        image: Optional[cv2.Mat]
        result, image = self._cap.read()

        if not result or image is None:
            if self._cap.get(cv2.CAP_PROP_POS_FRAMES) >= self._cap.get(cv2.CAP_PROP_FRAME_COUNT):
                self._logger.info("VIDEO FINISHED")
                return None
            else:
                self._logger.error("ERROR READING FRAME")
                raise RuntimeError("Error reading frame from video source.")
        return image
    
    ### ABSTRACT METHODS
    @abstractmethod
    def process_image(self, image: cv2.Mat) -> None:
        """
        Abstract method to process a single image. Must be implemented by subclasses.

        Args:
            image (cv2.Mat): The video image to process.
        """
        pass

    @abstractmethod
    def process(self) -> None:
        """
        Abstract method to process the video. Must be implemented by subclasses.
        """
        pass

    ##### DISPLAY #####
    def log(self, level: int, message: str) -> None:
        """
        Log a message at the specified logging level.
        Args:
            level (int): Logging level (e.g., logging.INFO, logging.ERROR).
            message (str): Message to log.
        """
        if self._logger:
            self._logger.log(level, message)

