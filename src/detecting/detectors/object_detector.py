import torch
import numpy as np
from typing import List, Optional
from torchvision.models.detection import (
    fasterrcnn_resnet50_fpn,
    fasterrcnn_mobilenet_v3_large_fpn,
    retinanet_resnet50_fpn,
)
import cv2
from ..detector import Detector
from ..detection import Detection
from ...core.properties import BoundingBox


class ObjectDetector(Detector):
    """FasterRCNN-based object detector for COCO classes."""

    MODELS = {
        'resnet50': fasterrcnn_resnet50_fpn,
        'mobilenet': fasterrcnn_mobilenet_v3_large_fpn,
        'retinanet': retinanet_resnet50_fpn,
    }

    def __init__(self, model='mobilenet', device='cpu', conf_threshold=0.5, classes=None,
                 min_size=None, max_size=None):
        """Initialize object detector with specified model.

        Args:
            model: Model name - 'resnet50', 'mobilenet', or 'retinanet'
            device: 'cpu', 'mps' or 'cuda'
            conf_threshold: Minimum confidence for detections
            classes: List of class IDs to filter (None = all classes)
            min_size: Minimum image size for model (None = use model default 800)
            max_size: Maximum image size for model (None = use model default 1333)

        Raises:
            ValueError: If model name is not recognized
        """
        if model not in self.MODELS:
            valid_models = ', '.join(self.MODELS.keys())
            raise ValueError(f"Unknown model '{model}'. Valid options: {valid_models}")

        # Load the model
        model_fn = self.MODELS[model]
        loaded_model = model_fn(weights='DEFAULT')

        # Override internal resize if specified
        if min_size is not None or max_size is not None:
            # Access the transform inside the model
            if hasattr(loaded_model, 'transform'):
                if min_size is not None:
                    loaded_model.transform.min_size = (min_size,)
                if max_size is not None:
                    loaded_model.transform.max_size = max_size

        # Initialize parent class
        super().__init__(loaded_model, device, conf_threshold)

        # Store class filter
        self.classes = classes
        self.min_size = min_size
        self.max_size = max_size 

    def preprocess(self, image: np.ndarray) -> torch.Tensor:
        """Convert BGR image to RGB tensor normalized to [0, 1].

        Uses the default PyTorch preprocessing: BGR → RGB, normalize to [0, 1],
        and convert to CHW format.

        Args:
            image: OpenCV image in BGR format (H, W, 3)

        Returns:
            Tensor with shape (3, H, W) and values in [0, 1]
        """
        return self._default_preprocess_pytorch(image)

    def inference(self, input_tensor: torch.Tensor) -> dict:
        """Run FasterRCNN inference.

        Uses the default PyTorch inference: add batch dimension, move to device,
        run model, and return first output.

        Args:
            input_tensor: Preprocessed image tensor (3, H, W)

        Returns:
            FasterRCNN output dict with 'boxes', 'labels', 'scores'
        """
        return self._default_inference_pytorch(input_tensor)

    def postprocess(self, output: dict, image_shape: tuple) -> List[Detection]:
        """Filter detections and convert to Detection objects.

        Args:
            output: FasterRCNN output with 'boxes', 'labels', 'scores'
            image_shape: (height, width) of original image

        Returns:
            List of Detection objects with 'bbox', 'class_id', 'confidence'
        """

        if len(output['scores']) == 0:
            return []


        boxes = output['boxes'].cpu()
        labels = output['labels'].cpu()
        scores = output['scores'].cpu()

        # Filter by confidence threshold
        mask = scores >= self.conf_threshold
        boxes = boxes[mask]
        labels = labels[mask]
        scores = scores[mask]

        # Filter by class IDs if specified
        # if self.classes is not None:
        #     class_mask = torch.zeros(len(labels), dtype=torch.bool)
        #     for class_id in self.classes:
        #         class_mask |= (labels == class_id)
        #     boxes = boxes[class_mask]
        #     labels = labels[class_mask]
        #     scores = scores[class_mask]

        if self.classes is not None:
            class_mask = torch.isin(labels, torch.tensor(self.classes))

        # Create Detection objects
        detections = [
            Detection({
                'bbox': BoundingBox(*box.tolist()),
                'class_id': int(label),
                'confidence': float(score)
            })
            for box, label, score in zip(boxes, labels, scores)
        ]


        return detections
