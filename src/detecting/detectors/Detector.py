from detecting.detections.Detection import Detection
from utils.geometry import compute_iou_array, nms

from abc import ABC, abstractmethod
import cv2
import numpy as np
from typing import Any, Optional
import logging

class Detector(ABC): 
    """
    Abstract base class for a detector. Detectors can be object detectors, face detectors, keypoint estimators, re-id systems etc..

    Attributes:
        model_path (str): Path to the model file.
        confidence_threshold (float): Confidence threshold for detection.
        logger (Optional[logging.Logger]): Logger instance for logging messages (optional).
    """

    ##### SETUP #####
    def __init__(self, 
                model_path:str, 
                confidence_threshold:float = 0.1, 
                logger: Optional[logging.Logger] = None):
        self._model_path = model_path
        self._confidence_threshold = confidence_threshold
        self._logger = logger

    ##### DETECTION #####
    @abstractmethod
    def create_detections(self, data: tuple) -> list:
        """
        Create Detection objects from the model output.

        Args:
            data (tuple): The model output.

        Returns:
            list: List of Detection objects.
        """
        pass

    @abstractmethod
    def preprocess(self, image: cv2.Mat) -> np.ndarray:
        """
        Preprocess the input image for the detector.

        Args:
            image (cv2.Mat): The input image.

        Returns:
            Any: The preprocessed image blob.
        """
        pass

    @abstractmethod
    def inference(self, blob:np.ndarray) -> list:
        """
        Perform inference on the preprocessed image blob.

        Args:
            blob (np.ndarray): The preprocessed image blob.

        Returns:
            Any: The model output.
        """
        pass

    @abstractmethod
    def postprocess(self, output: list) -> Any:
        """
        Postprocess the model output.

        Args:
            output (Any): The model output.

        Returns:
            Any: The postprocessed output.
        """
        
        pass

    def detect(self, image:cv2.Mat) -> list[Detection]:
        """
        Detect objects in the image.

        Args:
            image (cv2.Mat): Input image.

        Returns:
            list[Detection]: List of detected objects.
        """
        
        blob = self.preprocess(image)                          # PREPROCESS
        output = self.inference(blob)                          # INFERENCE
        output_data = self.postprocess(output)                 # POSTPROCESS
        detections = self.create_detections(output_data)       # CREATE DETECTIONS
        return detections
    
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

    def draw_detections(self, detections: list[Detection], image: cv2.Mat) -> None:
        """
        Draw detections on the given image.

        Args:
            detections (list[Detection]): List of detections to display.
            image (cv2.Mat): The image on which to display the detections.
        """

        tab = "\n\t\t\t\t\t"
        if not detections:
            if self._logger:
                self.log(logging.INFO, "\t// DETECTIONS: None")
            return

        detections_log = []

        for detection in detections:
            detection.draw(image, colour=(255, 255, 255))
            if self._logger:
                detections_log.append(str(detection))

        if self._logger:
            log_output = tab.join(detections_log)
            self.log(logging.DEBUG, f"\t// DETECTIONS:{tab}{log_output}")
