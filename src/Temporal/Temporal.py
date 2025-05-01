from abc import ABC, abstractmethod
import cv2
import logging
from typing import Optional

class Temporal(ABC):
    """
    Abstract base class for processing temporal data (videos). 
    
    Processes videos frame-by-frame and applies the `process_image()` method to each frame.
    Handles event-based user input like saving frames, quitting, and toggling playback modes.

    Attributes:
        video_path (str): Path to the video file.
        timestep (int): Current timestep of temporal data. 
        continuous_mode (bool): Flag to indicate if the video is processed in continuous mode (True) or step-by-step mode (False).
        cap (cv2.VideoCapture): Video capture object for reading video frames.
        logger (logging.Logger): Logger instance for logging messages.
    """

    __slots__ = ("_video_path", "_timestep", "_continuous_mode", "_cap", "_logger")

    ##### PROPERTIES #####
    @property
    def timestep(self) -> int:
        return self._timestep

    @timestep.setter
    def timestep(self, value: int) -> None:
        self._timestep = value

    def update_timestep(self) -> None:
        """
        Updates the current timestep based on the video capture position.
        """
        self._timestep = int(self._cap.get(cv2.CAP_PROP_POS_FRAMES))

    ##### SETUP #####
    def __init__(self, video_path: str, logger: Optional[logging.Logger] = None):
        """
            Args:
            video_path (str): Path to the video file.
            logger (Optional[logging.Logger]): Logger instance for logging messages (optional).
        """
        self._video_path = video_path
        self._timestep = 0
        self._continuous_mode = False  # Start in step-by-step mode
        self._logger = logger if logger else logging.getLogger(__name__)
        self._cap = None

        self.log(logging.INFO, f"|| TEMPORAL INITIALISED - VIDEO: {self._video_path}")

    def create_video_capture(self, video_path: str) -> cv2.VideoCapture:
        """
        Creates a video capture object for the specified video file.

        Args:
            video_path (str): Path to the video file.

        Returns:
            cv2.VideoCapture: Video capture object.

        Raises:
            ValueError: If the video source cannot be opened.
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            self.log(logging.ERROR, f"COULD NOT OPEN VIDEO CAPTURE FOR: {video_path}")
            raise ValueError(f"Video source could not be opened: {video_path}")
        return cap

    def create_source_capture(self) -> cv2.VideoCapture:
        """
        Creates a video capture object for the default camera.

        Returns:
            cv2.VideoCapture: Video capture object.

        Raises:
            ValueError: If the stream source cannot be opened.
        """
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            self.log(logging.ERROR, "COULD NOT OPEN STREAM CAPTURE")
            raise ValueError("Stream source could not be opened.")
        return cap

    def terminate(self) -> None:
        """
        Releases the video capture object and closes all OpenCV windows.
        """
        if self._cap is not None:
            self._cap.release()
            self.log(logging.INFO, "|| TERMINATED TEMPORAL COMPONENT")
        cv2.destroyAllWindows()

    ##### EVENT HANDLERS #####
    def save_event(self, key: int, image: cv2.Mat, name: str) -> bool:
        """
        Event handler for saving the current image to a file.

        Args:
            key (int): The key code of the pressed key.
            image (cv2.Mat): The current video frame.
            name (str): The name of the file to save the image.

        Returns:
            bool: True if the save event is triggered, False otherwise.
        """
        if key == ord('s'):
            filename = f"{name}_{self._timestep}.jpg"
            if cv2.imwrite(filename, image):
                self.log(logging.INFO, f"// EVENT: SAVE FRAME {self._timestep} as {filename}")
                return True
            else:
                self.log(logging.ERROR, f"FAILED TO SAVE FRAME {self._timestep}")
        return False

    def toggle_event(self, key: int) -> bool:
        """
        Event handler for toggling between continuous and step-by-step modes.

        Args:
            key (int): The key code of the pressed key.

        Returns:
            bool: True if the toggle event is triggered, False otherwise.
        """
        if key == ord('c'):
            self._continuous_mode = not self._continuous_mode
            mode = "CONTINUOUS" if self._continuous_mode else "STEPPED"
            self.log(logging.INFO, f"// EVENT: TOGGLED TO {mode} mode")
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
            self.log(logging.INFO, "// EVENT: QUIT")
            return True
        return False

    ##### VIDEO PROCESSING #####
    def update_timestep(self) -> None:
        """
        Updates the current timestep based on the video capture position.
        """
        self._timestep = int(self._cap.get(cv2.CAP_PROP_POS_FRAMES))

    def continue_video(self) -> Optional[cv2.Mat]:
        """
        Reads the next frame from the video source.

        Returns:
            Optional[cv2.Mat]: The next frame read from the video source, or None if the video has ended.

        Raises:
            RuntimeError: If there is an error reading the frame from the video source.
        """

        # Read the next frame from the video source
        result, image = self._cap.read()

        # Check if the frame was read successfully
        if not result or image is None:
            if self._cap.get(cv2.CAP_PROP_POS_FRAMES) >= self._cap.get(cv2.CAP_PROP_FRAME_COUNT):
                self.log(logging.INFO, "|| END OF VIDEO")
                return None
            else:
                self.log(logging.ERROR, "FRAME IS NOT READ SUCCESSFULLY")
                raise RuntimeError("Error reading frame from video source.")
            
        # Update the current timestep
        self.update_timestep()

        self.log(logging.INFO, f"// FRAME: {self._timestep}")
        # if self._logger: print("") # Skip line for readability while using logger
        return image

    @abstractmethod
    def process_image(self, image: cv2.Mat) -> None:
        """
        Abstract method to process a single image. Must be implemented by subclasses.

        Args:
            image (cv2.Mat): The video frame to process.
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

