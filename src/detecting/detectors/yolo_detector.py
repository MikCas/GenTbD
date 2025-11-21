import torch
import numpy as np
from typing import List, Optional
from ultralytics import YOLO
from ..detector import Detector
from ..detection import Detection
from ...core.properties import BoundingBox


class YOLODetector(Detector):
    """YOLO-based object detector for COCO classes.

    Uses Ultralytics YOLO models (YOLOv8/YOLOv11) which have better MPS
    support and avoid the aten::nonzero performance bug that affects FasterRCNN.
    """

    MODELS = {
        'v8n': 'yolov8n.pt',         # Nano - fastest, ~6.3M params
        'v8s': 'yolov8s.pt',         # Small - ~11.2M params
        'v8m': 'yolov8m.pt',         # Medium - ~25.9M params
        'v8l': 'yolov8l.pt',         # Large - ~43.7M params
        'v8x': 'yolov8x.pt',         # Extra large - ~68.2M params
        'v11n': 'yolo11n.pt',        # YOLOv11 nano - latest
        'v11s': 'yolo11s.pt',        # YOLOv11 small - latest
        'v11m': 'yolo11m.pt',        # YOLOv11 medium - latest
    }

    def __init__(self, model='v8n', device='cpu', conf_threshold=0.5, classes=None):
        """Initialize YOLO detector with specified model.

        Args:
            model: Model name - 'v8n', 'v8s', 'v8m', 'v8l', 'v8x',
                   'v11n', 'v11s', 'v11m'
            device: 'cpu', 'mps' or 'cuda'
            conf_threshold: Minimum confidence for detections
            classes: List of class IDs to filter (None = all classes)

        Raises:
            ValueError: If model name is not recognized
        """
        if model not in self.MODELS:
            valid_models = ', '.join(self.MODELS.keys())
            raise ValueError(f"Unknown model '{model}'. Valid options: {valid_models}")

        # Load YOLO model
        model_path = self.MODELS[model]
        yolo_model = YOLO(model_path)

        # YOLO models don't need explicit .to(device) - handled in predict()
        # But we need a torch.nn.Module for the parent class
        # Use the underlying model
        loaded_model = yolo_model.model

        # Store the YOLO wrapper for prediction
        self._yolo = yolo_model

        # Initialize parent class
        super().__init__(loaded_model, device, conf_threshold)

        # Store class filter
        self.classes = classes

    def preprocess(self, image: np.ndarray) -> np.ndarray:
        """YOLO handles preprocessing internally, just return the image.

        The Ultralytics YOLO library handles all preprocessing internally
        including BGR→RGB conversion, normalization, and resizing.

        Args:
            image: OpenCV image in BGR format (H, W, 3)

        Returns:
            Original image (preprocessing happens in inference)
        """
        # YOLO expects BGR images directly from OpenCV
        return image

    def inference(self, input_image: np.ndarray) -> list:
        """Run YOLO inference.

        Uses the Ultralytics YOLO predict() method which handles all
        preprocessing, device transfer, and inference internally.

        Args:
            input_image: BGR image from OpenCV (H, W, 3)

        Returns:
            YOLO Results object
        """
        # YOLO handles everything: preprocessing, device transfer, inference
        # verbose=False suppresses progress bars and logs
        results = self._yolo.predict(
            input_image,
            device=self.device,
            conf=self.conf_threshold,
            classes=self.classes,  # Filter classes at inference time
            verbose=False
        )
        return results

    def postprocess(self, output: list, image_shape: tuple) -> List[Detection]:
        """Convert YOLO results to Detection objects.

        Args:
            output: YOLO Results object from predict()
            image_shape: (height, width) of original image (unused, YOLO preserves coords)

        Returns:
            List of Detection objects with 'bbox', 'class_id', 'confidence'
        """
        # YOLO returns a list of Results objects (one per image)
        # Since we process one image at a time, take the first result
        result = output[0]

        # Extract boxes, classes, and confidences
        # result.boxes is a Boxes object with xyxy, cls, conf attributes
        boxes = result.boxes

        detections = []
        for i in range(len(boxes)):
            # Get bounding box coordinates (already in xyxy format)
            x1, y1, x2, y2 = boxes.xyxy[i].cpu().tolist()

            # Get class ID and confidence
            class_id = int(boxes.cls[i].cpu().item())
            confidence = float(boxes.conf[i].cpu().item())

            # Create Detection object
            detection = Detection({
                'bbox': BoundingBox(x1, y1, x2, y2),
                'class_id': class_id,
                'confidence': confidence
            })
            detections.append(detection)

        return detections

    def detect(self, image: np.ndarray) -> List[Detection]:
        """Run full detection pipeline.

        Override the parent detect() to use YOLO's optimized pipeline
        instead of the three-stage preprocess→inference→postprocess.

        Args:
            image: Input image in BGR format (H, W, 3)

        Returns:
            List of Detection objects
        """
        # YOLO's predict() handles everything internally (much more efficient)
        results = self.inference(image)
        return self.postprocess(results, image.shape[:2])
