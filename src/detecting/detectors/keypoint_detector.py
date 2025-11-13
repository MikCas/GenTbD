"""Keypoint R-CNN-based human pose estimator for COCO keypoints."""

import torch
import numpy as np
from typing import List
from torchvision.models.detection import keypointrcnn_resnet50_fpn
import cv2
from ..detector import Detector
from ..detection import Detection
from ...core.properties import BoundingBox, Keypoints


class KeypointDetector(Detector):
    """Keypoint R-CNN-based detector for human pose estimation.

    Detects people and estimates 17 body keypoints per person using
    Keypoint R-CNN pre-trained on COCO dataset.
    """

    MODELS = {
        'resnet50': keypointrcnn_resnet50_fpn,
    }

    def __init__(self, model='resnet50', device='cpu', conf_threshold=0.5,
                 keypoint_threshold=0.5):
        """Initialize keypoint detector with specified model.

        Args:
            model: Model name - 'resnet50' (only option currently)
            device: 'cpu', 'mps' or 'cuda'
            conf_threshold: Minimum confidence for person detections
            keypoint_threshold: Minimum visibility for individual keypoints

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

        # Store keypoint-specific threshold
        self.keypoint_threshold = keypoint_threshold

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
        """Run Keypoint R-CNN inference.

        Args:
            input_tensor: Preprocessed image tensor (3, H, W)

        Returns:
            Keypoint R-CNN output dict with 'boxes', 'labels', 'scores',
            'keypoints', and 'keypoints_scores'
        """
        # Add batch dimension and move to device
        input_batch = input_tensor.unsqueeze(0).to(self.device)
        # Run model (returns list of dicts, one per image)
        outputs = self.model(input_batch)
        return outputs[0]

    def postprocess(self, output: dict, image_shape: tuple) -> List[Detection]:
        """Filter detections and convert to Detection objects with keypoints.

        Args:
            output: Keypoint R-CNN output with 'boxes', 'labels', 'scores',
                   'keypoints', and 'keypoints_scores'
            image_shape: (height, width) of original image

        Returns:
            List of Detection objects with 'bbox', 'class_id', 'confidence',
            and 'keypoints' properties
        """
        boxes = output['boxes'].cpu()
        labels = output['labels'].cpu()
        scores = output['scores'].cpu()
        keypoints = output['keypoints'].cpu()  # [N, 17, 3]
        keypoints_scores = output['keypoints_scores'].cpu()  # [N, 17]

        # Filter by confidence threshold
        mask = scores >= self.conf_threshold
        boxes = boxes[mask]
        labels = labels[mask]
        scores = scores[mask]
        keypoints = keypoints[mask]
        keypoints_scores = keypoints_scores[mask]

        # Note: Keypoint R-CNN only detects people (class_id=1)
        # No need for additional class filtering

        # Create Detection objects
        detections = []
        for box, label, score, kpts, kpts_scores in zip(
            boxes, labels, scores, keypoints, keypoints_scores
        ):
            x1, y1, x2, y2 = box.tolist()

            # Convert keypoints to numpy for Keypoints class
            kpts_array = kpts.numpy()  # [17, 3] with [x, y, visibility]
            kpts_scores_array = kpts_scores.numpy()  # [17]

            # Create Keypoints object
            keypoints_obj = Keypoints(kpts_array, kpts_scores_array)

            detection = Detection({
                'bbox': BoundingBox(x1, y1, x2, y2),
                'class_id': int(label),  # Always 1 (person)
                'confidence': float(score),
                'keypoints': keypoints_obj
            })
            detections.append(detection)

        return detections
