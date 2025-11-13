"""
Feature extractor base class for appearance-based re-identification.

This module provides the FeatureExtractor abstract base class for extracting
appearance embeddings from person crops. Unlike Detector which processes full
frames, FeatureExtractor works on pre-cropped regions to generate embeddings.
"""

from abc import ABC, abstractmethod
from typing import List
import numpy as np
from .detection import Detection
from ..core.properties import Embedding


class FeatureExtractor(ABC):
    """
    Abstract base class for appearance feature extraction.

    FeatureExtractor differs from Detector in that it:
    - Operates on cropped regions (not full frames)
    - Produces embeddings (not detections)
    - Typically enhances existing detections rather than creating new ones

    Typical usage patterns:
    1. **Enhancement mode**: Add embeddings to existing detections
       >>> extractor = ReIDExtractor(model='osnet_x1_0')
       >>> detections = object_detector.detect(frame)
       >>> detections = extractor.enhance(frame, detections)

    2. **Direct extraction**: Extract embeddings from crops
       >>> crop = frame[y1:y2, x1:x2]
       >>> embedding = extractor.extract(crop)

    3. **Batch extraction**: Extract embeddings from multiple crops
       >>> crops = [frame[y1:y2, x1:x2] for bbox in bboxes]
       >>> embeddings = extractor.extract_batch(crops)
    """

    def __init__(self, model_name: str, device: str = 'cpu', embedding_dim: int = 512):
        """
        Initialize feature extractor.

        Args:
            model_name: Name of the feature extraction model
            device: Device to run on ('cpu', 'cuda', 'mps')
            embedding_dim: Dimensionality of output embeddings
        """
        self.model_name = model_name
        self.device = device
        self.embedding_dim = embedding_dim

    @abstractmethod
    def extract(self, image_crop: np.ndarray) -> Embedding:
        """
        Extract embedding from a single image crop.

        Args:
            image_crop: BGR image crop (H, W, 3)

        Returns:
            Embedding object with L2-normalized feature vector

        Example:
            >>> crop = frame[100:300, 50:150]
            >>> embedding = extractor.extract(crop)
            >>> print(f"Embedding dimension: {embedding.dim}")
        """
        pass

    @abstractmethod
    def extract_batch(self, image_crops: List[np.ndarray]) -> List[Embedding]:
        """
        Extract embeddings from multiple image crops (batch processing).

        More efficient than calling extract() multiple times.

        Args:
            image_crops: List of BGR image crops

        Returns:
            List of Embedding objects

        Example:
            >>> crops = [frame[y1:y2, x1:x2] for bbox in bboxes]
            >>> embeddings = extractor.extract_batch(crops)
            >>> for emb in embeddings:
            >>>     print(f"Dimension: {emb.dim}")
        """
        pass

    def enhance(self, frame: np.ndarray, detections: List[Detection]) -> List[Detection]:
        """
        Add embeddings to existing detections (primary usage mode).

        This is the main method for integrating feature extraction into
        detection pipelines:
        1. Object detector finds people → detections with bboxes
        2. FeatureExtractor adds embeddings → detections with bboxes + embeddings
        3. Tracker uses both for association

        Args:
            frame: Full frame BGR image (H, W, 3)
            detections: List of Detection objects with 'bbox' property

        Returns:
            Same detections list with 'embedding' property added

        Example:
            >>> detections = object_detector.detect(frame)
            >>> detections = extractor.enhance(frame, detections)
            >>> for det in detections:
            >>>     if 'embedding' in det:
            >>>         print(f"Added embedding: {det['embedding'].dim}D")
        """
        if not detections:
            return detections

        # Extract crops from frame using bounding boxes
        crops = []
        valid_indices = []

        for i, det in enumerate(detections):
            if 'bbox' not in det:
                continue

            bbox = det['bbox']
            x1, y1, x2, y2 = map(int, bbox.xyxy)

            # Ensure coordinates are within frame bounds
            h, w = frame.shape[:2]
            x1 = max(0, min(x1, w - 1))
            y1 = max(0, min(y1, h - 1))
            x2 = max(x1 + 1, min(x2, w))
            y2 = max(y1 + 1, min(y2, h))

            # Extract crop
            crop = frame[y1:y2, x1:x2]

            # Skip if crop is too small (not enough pixels for meaningful embedding)
            if crop.shape[0] < 10 or crop.shape[1] < 5:
                continue

            crops.append(crop)
            valid_indices.append(i)

        # Extract embeddings in batch
        if crops:
            embeddings = self.extract_batch(crops)

            # Add embeddings to detections
            for idx, embedding in zip(valid_indices, embeddings):
                detections[idx]['embedding'] = embedding

        return detections
