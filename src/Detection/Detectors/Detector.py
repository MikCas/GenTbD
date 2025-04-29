from Detection.Detection import Detection

from abc import ABC, abstractmethod
import cv2
import onnxruntime as ort
import numpy as np
from typing import Any, Optional
import logging

class Detector(ABC): 
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

    ### STATIC METHODS
    @staticmethod
    def compute_iou(box: np.ndarray, boxes: np.ndarray) -> np.ndarray:
        """
        Compute the Intersection over Union (IoU) between a given box and a set of boxes.

        Args:
            box (np.ndarray): A single bounding box in corner format (x1, y1, x2, y2).
            boxes (np.ndarray): An array of bounding boxes in corner format (x1, y1, x2, y2).

        Returns:
            np.ndarray: An array of IoU values between the given box and each box in the input array.
        """
        # Calculate the coordinates of the intersection rectangle
        xmin = np.maximum(box[0], boxes[:, 0])
        ymin = np.maximum(box[1], boxes[:, 1])
        xmax = np.minimum(box[2], boxes[:, 2])
        ymax = np.minimum(box[3], boxes[:, 3])

        # Compute the area of the intersection rectangle
        intersection_width = np.maximum(0, xmax - xmin)
        intersection_height = np.maximum(0, ymax - ymin)
        intersection_area = intersection_width * intersection_height

        # Compute the area of the union
        box_area = (box[2] - box[0]) * (box[3] - box[1])
        boxes_area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        union_area = box_area + boxes_area - intersection_area

        # Compute the IoU
        iou = intersection_area / np.maximum(union_area, 1e-6)  # Avoid division by zero

        return iou
    
    @staticmethod
    def nms(boxes: np.ndarray, scores: np.ndarray, iou_threshold: float) -> list:
        """
        Perform Non-Maximum Suppression (NMS) to filter overlapping bounding boxes.

        Args:
            boxes (np.ndarray): Array of bounding boxes in corner format (x1, y1, x2, y2).
            scores (np.ndarray): Array of confidence scores corresponding to the bounding boxes.
            iou_threshold (float): IoU threshold for suppressing overlapping boxes.

        Returns:
            list: Indices of the bounding boxes to keep after applying NMS.
        """
        # Sort indices of boxes by scores in descending order
        sorted_indices = np.argsort(scores)[::-1]

        keep_boxes = []
        while sorted_indices.size > 0:
            # Select the box with the highest score
            current_box_index = sorted_indices[0]
            keep_boxes.append(current_box_index)

            # Compute IoU of the selected box with the remaining boxes
            ious = Detector.compute_iou(boxes[current_box_index], boxes[sorted_indices[1:]])

            # Filter out boxes with IoU above the threshold
            remaining_indices = np.where(ious < iou_threshold)[0]

            # Update the sorted indices to exclude suppressed boxes
            sorted_indices = sorted_indices[remaining_indices + 1]

        return keep_boxes

    ### ABSTRACT METHODS 
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

    ### FUNCTIONALITY
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