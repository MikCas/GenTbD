"""
PyTorch YOLO detector using Ultralytics.

This detector uses the Ultralytics YOLO framework, which makes detection MUCH easier:
- Auto-downloads models
- Handles preprocessing internally
- Returns results in a clean format
- ~50 lines vs 263 for ONNX!

Supported Models (all auto-download on first use):
Detection:
  - yolov8n (nano): 3.2M params, fastest
  - yolov8s (small): 11.2M params
  - yolov8m (medium): 25.9M params
  - yolov8l (large): 43.7M params
  - yolov8x (extra large): 68.2M params

Pose (for future):
  - yolov8n-pose, yolov8s-pose, etc.

Segmentation (for future):
  - yolov8n-seg, yolov8s-seg, etc.
"""

from properties.BoundingBox import BoundingBox
from detecting.detections.ObjectDetection import ObjectDetection as Detection
from detecting.detectors.Detector import Detector
from detecting.backends.PyTorchBackend import PyTorchBackend

import cv2
import numpy as np
from typing import List, Optional
import logging


class YOLOv8PyTorch(Detector):
    """
    YOLOv8 detector using PyTorch backend.

    This is MUCH simpler than the ONNX version because Ultralytics handles:
    - Preprocessing (resizing, padding, normalization)
    - NMS (non-maximum suppression)
    - Postprocessing (extracting boxes, scores, classes)

    You just pass an image and get detections!

    Example:
        >>> detector = YOLOv8PyTorch(model_name='yolov8n')
        >>> detections = detector.detect(image)
    """

    def __init__(self,
                 model_name: str = 'yolov8n.pt',
                 confidence_threshold: float = 0.1,
                 iou_threshold: float = 0.5,
                 classes: Optional[List[int]] = None,
                 device: Optional[str] = None,
                 logger: Optional[logging.Logger] = None):
        """
        Initialize YOLOv8 detector.

        Args:
            model_name: Model to use. Options:
                       'yolov8n', 'yolov8s', 'yolov8m', 'yolov8l', 'yolov8x'
                       Or path to custom .pt file
            confidence_threshold: Minimum confidence for detections
            iou_threshold: IoU threshold for NMS
            classes: List of class IDs to detect (None = all classes)
                    Example: [0] for person only
            device: Device to use ('cpu', 'cuda', 'mps', or None for auto)
            logger: Logger instance

        Learning Point:
            Compare this __init__ to YOLOv7ONNX:
            - No create_onnx_model() method needed
            - No manual input/output name extraction
            - Backend handles all the complexity
        """
        # We don't pass model_path to parent since we use backend
        # Initialize parent with dummy path (backend handles actual loading)
        super().__init__('', confidence_threshold, logger)

        self._iou_threshold = iou_threshold
        self._classes = classes if classes is not None else [0]  # Default to person

        # Create PyTorch backend
        self._backend = PyTorchBackend(device=device, logger=logger)

        # Load model through backend
        self._backend.load_model(model_name)

        # Store reference to actual model for Ultralytics-specific methods
        self._model = self._backend._model

        self.log(logging.INFO, f"|| YOLOv8 DETECTOR INITIALISED\n"
                              f"\t\t\t\t    - MODEL: {model_name}\n"
                              f"\t\t\t\t    - DEVICE: {self._backend.get_device()}\n"
                              f"\t\t\t\t    - CONFIDENCE: {confidence_threshold}\n"
                              f"\t\t\t\t    - IOU: {iou_threshold}")

    def preprocess(self, image: cv2.Mat) -> np.ndarray:
        """
        Preprocess image for YOLO.

        Args:
            image: Input image (BGR format from OpenCV)

        Returns:
            Image (Ultralytics handles preprocessing internally!)

        Learning Point:
            Unlike ONNX version (132 lines of preprocessing),
            Ultralytics does ALL preprocessing internally:
            - RGB conversion
            - Resizing with aspect ratio
            - Padding
            - Normalization
            - Tensor conversion

            We just return the image as-is!
        """
        self.log(logging.INFO, "\t|| PREPROCESSING (handled by Ultralytics)")
        return image

    def inference(self, image: np.ndarray) -> List:
        """
        Run inference using backend.

        Args:
            image: Input image

        Returns:
            Ultralytics Results object

        Learning Point:
            Ultralytics Results object contains:
            - results[0].boxes.xyxy: Bounding boxes in xyxy format
            - results[0].boxes.conf: Confidence scores
            - results[0].boxes.cls: Class IDs
            - And much more!
        """
        self.log(logging.INFO, "\t|| INFERENCE")

        # Ultralytics YOLO.predict() parameters:
        # - conf: confidence threshold
        # - iou: NMS IoU threshold
        # - classes: filter by class IDs
        # - verbose: suppress output
        results = self._model.predict(
            image,
            conf=self._confidence_threshold,
            iou=self._iou_threshold,
            classes=self._classes,
            verbose=False
        )

        return results

    def postprocess(self, results: List) -> tuple:
        """
        Extract boxes, classes, and scores from Ultralytics results.

        Args:
            results: Ultralytics Results objects

        Returns:
            Tuple of (boxes, class_ids, scores)

        Learning Point:
            Ultralytics already did:
            - Filtering by confidence threshold
            - NMS (non-maximum suppression)
            - Coordinate conversion

            We just extract the data!

            Compare to ONNX version (56 lines of postprocessing):
            - No manual confidence filtering
            - No manual NMS application
            - No manual coordinate conversion
        """
        self.log(logging.INFO, "\t|| POSTPROCESS")

        # Extract first result (batch size = 1)
        result = results[0]

        # Check if any detections
        if len(result.boxes) == 0:
            return np.array([]), np.array([]), np.array([])

        # Extract data from Results object
        # Learning Point: .cpu().numpy() converts PyTorch tensors to numpy
        boxes = result.boxes.xyxy.cpu().numpy()  # Shape: (N, 4)
        scores = result.boxes.conf.cpu().numpy()  # Shape: (N,)
        class_ids = result.boxes.cls.cpu().numpy().astype(int)  # Shape: (N,)

        return boxes, class_ids, scores

    def create_detections(self, data: tuple) -> List[Detection]:
        """
        Create Detection objects from postprocessed data.

        Args:
            data: Tuple of (boxes, class_ids, scores)

        Returns:
            List of Detection objects

        Learning Point:
            This is identical to the ONNX version!
            Same interface, different backend.
        """
        boxes, class_ids, scores = data

        return [
            Detection(
                class_id=int(class_id),
                bounding_box=BoundingBox.from_corners(*box),
                confidence_score=float(score)
            )
            for box, class_id, score in zip(boxes, class_ids, scores)
        ]
