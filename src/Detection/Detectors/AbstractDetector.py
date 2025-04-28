from Properties.BoundingBox import BoundingBox
from Detection.Detection import Detection

from abc import ABC, abstractmethod
import cv2
import onnxruntime as ort
import numpy as np
from typing import Any, Optional
import logging

class AbstractDetector(ABC): 
    """
    Abstract base class for an ONNX detector.

    Attributes:
        model_path (str): Path to the model file.
        confidence_threshold (float): Confidence threshold for detection.
        logger (Optional[logging.Logger]): Logger instance for logging messages (optional).
    
        session (ort.InferenceSession): ONNX inference session.
        input_names (list): List of input names for the model.
        input_shape (tuple): Shape of the model input.
        input_height (int): Height of the model input.
        input_width (int): Width of the model input.
        output_names (list): List of output names for the model.
    """

    ### SETUP 
    def create_onnx_model(self, model_path: str) -> None:
        """
        Set up the ONNX model.

        Args:
            model_path (str): Path to the ONNX model file.

        Returns:
            ort.InferenceSession: Inference session for the ONNX model.
        """
        try:
            # Create the ONNX inference session
            self._session = ort.InferenceSession(model_path, providers=['CoreMLExecutionProvider', 'CPUExecutionProvider']) # Apple Silicon
            # session = ort.InferenceSession(model_path, providers=['CUDAExecutionProvider', 'CPUExecutionProvider']) # NVIDIA GPU
            # session = ort.InferenceSession(model_path, providers=['OpenVINOExecutionProvider', 'CPUExecutionProvider']) # Intel CPU
            # session = ort.InferenceSession(model_path, providers=['DmlExecutionProvider', 'CPUExecutionProvider']) # Windows GPU

            # Model input details   
            model_inputs = self._session.get_inputs()
            self._input_names = [input.name for input in model_inputs]     # Names
            self._input_shape = model_inputs[0].shape                      # Shape Ex. (1, 3, 640, 640)
            self._input_height = self._input_shape[2]                      # Height
            self._input_width = self._input_shape[3]                       # Width

            # Model output details
            model_outputs = self._session.get_outputs()
            self._output_names = [output.name for output in model_outputs] # Names

            if self._logger: self._logger.info(f"DETECTOR INITIALISED - MODEL {model_path}")

        except Exception as e:
            self._logger.error(f"COULD NOT LOAD ONNX MODEL: {e}")
            raise

    def __init__(self, 
                model_path:str, 
                confidence_threshold:float = 0.1, 
                logger: Optional[logging.Logger] = None):
        self._model_path = model_path
        self._confidence_threshold = confidence_threshold
        self._logger: logging.Logger = logger if logger else logging.getLogger(__name__)

        self.create_onnx_model(model_path)
    
        self._classes = []
        self._iou_threshold = 0.5
        self._confidence_threshold = confidence_threshold

    ### FUNCTIONS 
    @abstractmethod
    def create_detections(self, data: tuple) -> list[Detection]:
        """
        Create Detection objects from the model output.

        Args:
            data (tuple): The model output.

        Returns:
            list[Detection]: List of Detection objects.
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

    @abstractmethod
    def preprocess(self, image: cv2.Mat) -> np.ndarray:
        """
        Preprocess the input image for the detector.

        Args:
            image (cv2.Mat): The input image.

    #     Returns:
    #         Any: The preprocessed image blob.
    #     """
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
    
    ### REPRESENTATION
    def display_detections(self, detections: list[Detection], image: cv2.Mat) -> None:
        """
        Display detections on the given image.

        Args:
            detections (list[Detection]): List of detections to display.
            image (cv2.Mat): The image on which to display the detections.
        """
        for detection in detections:
            detection.display(image, colour=(255, 255, 255))
            if self._logger:
                self._logger.info(f"DETECTED: {detection}")