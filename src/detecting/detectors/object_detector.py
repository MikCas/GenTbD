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
from ..properties.bounding_box import BoundingBox


class ObjectDetector(Detector):
    """FasterRCNN-based object detector for COCO classes."""

    MODELS = {
        'resnet50': fasterrcnn_resnet50_fpn,
        'mobilenet': fasterrcnn_mobilenet_v3_large_fpn,
        'retinanet': retinanet_resnet50_fpn,
    }

    def __init__(self, model='mobilenet', device='cpu', conf_threshold=0.5, classes=None):
        """Initialize object detector with specified model.

        Args:
            model: Model name - 'resnet50', 'mobilenet', or 'retinanet'
            device: 'cpu', 'mps' or 'cuda'
            conf_threshold: Minimum confidence for detections
            classes: List of class IDs to filter (None = all classes)

        Raises:
            ValueError: If model name is not recognized
        """
        if model not in self.MODELS:
            valid_models = ', '.join(self.MODELS.keys())
            raise ValueError(f"Unknown model '{model}'. Valid options: {valid_models}")

        # Load the model
        model_fn = self.MODELS[model]
        loaded_model = model_fn(weights='DEFAULT')

        # Initialize parent class
        super().__init__(loaded_model, device, conf_threshold)

        # Store class filter
        self.classes = classes 

    def preprocess(self, image: np.ndarray) -> torch.Tensor:
        """Convert BGR image to RGB tensor normalized to [0, 1].

        Args:
            image: OpenCV image in BGR format (H, W, 3)

        Returns:
            Tensor with shape (3, H, W) and values in [0, 1]
        """
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        tensor = torch.from_numpy(image_rgb).float() / 255.0
        tensor = tensor.permute(2, 0, 1)
        return tensor

    def inference(self, input_tensor: torch.Tensor) -> dict:
        """Run FasterRCNN inference.

        Args:
            input_tensor: Preprocessed image tensor (3, H, W)

        Returns:
            FasterRCNN output dict with 'boxes', 'labels', 'scores'
        """
        # Add batch dimension and move to device
        input_batch = input_tensor.unsqueeze(0).to(self.device)
        # Run model (returns list of dicts, one per image)
        outputs = self.model(input_batch)
        return outputs[0]

    def postprocess(self, output: dict, image_shape: tuple) -> List[Detection]:
        """Filter detections and convert to Detection objects.

        Args:
            output: FasterRCNN output with 'boxes', 'labels', 'scores'
            image_shape: (height, width) of original image

        Returns:
            List of Detection objects with 'bbox', 'class_id', 'confidence'
        """
        boxes = output['boxes'].cpu()
        labels = output['labels'].cpu()
        scores = output['scores'].cpu()

        # Filter by confidence threshold
        mask = scores >= self.conf_threshold
        boxes = boxes[mask]
        labels = labels[mask]
        scores = scores[mask]

        # Filter by class IDs if specified
        if self.classes is not None:
            class_mask = torch.zeros(len(labels), dtype=torch.bool)
            for class_id in self.classes:
                class_mask |= (labels == class_id)
            boxes = boxes[class_mask]
            labels = labels[class_mask]
            scores = scores[class_mask]

        # Create Detection objects
        detections = []
        for box, label, score in zip(boxes, labels, scores):
            x1, y1, x2, y2 = box.tolist()
            detection = Detection({
                'bbox': BoundingBox(x1, y1, x2, y2),
                'class_id': int(label),
                'confidence': float(score)
            })
            detections.append(detection)

        return detections
