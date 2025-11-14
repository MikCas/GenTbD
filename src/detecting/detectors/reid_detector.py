"""ReID-based appearance feature extractor for person re-identification.

This module provides the ReIDExtractor class that extracts appearance embeddings
from person crops using deep learning models (OSNet, ResNet50, etc.) via torchreid.
"""

import torch
import numpy as np
from typing import List

from ..feature_extractor import FeatureExtractor
from ...core.properties import Embedding

# Lazy import to avoid circular dependency issues
_TorchReidFeatureExtractor = None


def _get_torchreid_extractor():
    """Lazy import of torchreid FeatureExtractor."""
    global _TorchReidFeatureExtractor
    if _TorchReidFeatureExtractor is None:
        import torchreid
        _TorchReidFeatureExtractor = torchreid.utils.FeatureExtractor
    return _TorchReidFeatureExtractor


class ReIDExtractor(FeatureExtractor):
    """
    ReID-based appearance feature extractor.

    Extracts appearance embeddings from person crops for re-identification
    across frames in multi-object tracking.

    Supported Models:
        - osnet_x1_0: OSNet with 1.0x width (512-dim, recommended)
        - osnet_x0_75: OSNet with 0.75x width (512-dim, faster)
        - osnet_x0_5: OSNet with 0.5x width (512-dim, very fast)
        - osnet_x0_25: OSNet with 0.25x width (512-dim, lightweight)
        - osnet_ain_x1_0: OSNet with Instance Normalization
        - resnet50: ResNet50 for ReID (2048-dim or 512-dim depending on variant)

    Example:
        >>> # Enhancement mode (typical usage)
        >>> obj_detector = ObjectDetector(model='mobilenet', classes=[1])  # person only
        >>> reid_extractor = ReIDExtractor(model='osnet_x1_0')
        >>>
        >>> detections = obj_detector.detect(frame)
        >>> detections = reid_extractor.enhance(frame, detections)
        >>>
        >>> # Now detections have 'embedding' property
        >>> for det in detections:
        >>>     if 'embedding' in det:
        >>>         print(f"Embedding dim: {det['embedding'].dim}")
        >>>
        >>> # Direct extraction from crops
        >>> crop = frame[y1:y2, x1:x2]
        >>> embedding = reid_extractor.extract(crop)
    """

    MODELS = {
        'osnet_x1_0': 'osnet_x1_0',
        'osnet_x0_75': 'osnet_x0_75',
        'osnet_x0_5': 'osnet_x0_5',
        'osnet_x0_25': 'osnet_x0_25',
        'osnet_ain_x1_0': 'osnet_ain_x1_0',
        'resnet50': 'resnet50',
    }

    def __init__(self, model='osnet_x1_0', device='cpu', embedding_dim=512):
        """
        Initialize ReID extractor.

        Args:
            model: Model name from MODELS dict
            device: 'cpu', 'mps', or 'cuda'
            embedding_dim: Embedding dimension (512 for OSNet, can vary)

        Raises:
            ValueError: If model name is not recognized
        """
        if model not in self.MODELS:
            valid_models = ', '.join(self.MODELS.keys())
            raise ValueError(f"Unknown model '{model}'. Valid options: {valid_models}")

        # Initialize base class
        super().__init__(model_name=model, device=device, embedding_dim=embedding_dim)

        # Initialize torchreid FeatureExtractor (lazy import)
        TorchReidExtractor = _get_torchreid_extractor()
        self._torchreid_extractor = TorchReidExtractor(
            model_name=self.MODELS[model],
            model_path=None,  # Use pretrained weights
            device=device
        )

    def extract(self, image_crop: np.ndarray) -> Embedding:
        """
        Extract embedding from a single image crop.

        Args:
            image_crop: BGR image crop (H, W, 3)

        Returns:
            Embedding object with L2-normalized vector

        Example:
            >>> crop = frame[y1:y2, x1:x2]
            >>> embedding = reid_extractor.extract(crop)
            >>> print(f"Embedding dimension: {embedding.dim}")
        """
        # Extract embedding using torchreid
        emb_tensor = self._torchreid_extractor([image_crop])[0]

        # Convert to numpy
        if isinstance(emb_tensor, torch.Tensor):
            emb_vector = emb_tensor.cpu().numpy()
        else:
            emb_vector = emb_tensor

        return Embedding(emb_vector, model=self.model_name)

    def extract_batch(self, image_crops: List[np.ndarray]) -> List[Embedding]:
        """
        Extract embeddings from multiple image crops (batch processing).

        More efficient than calling extract() multiple times.

        Args:
            image_crops: List of BGR image crops

        Returns:
            List of Embedding objects

        Example:
            >>> crops = [frame[y1:y2, x1:x2] for y1, x1, y2, x2 in bboxes]
            >>> embeddings = reid_extractor.extract_batch(crops)
        """
        if not image_crops:
            return []

        # Extract embeddings in batch using torchreid
        emb_tensors = self._torchreid_extractor(image_crops)

        # Convert to numpy if needed
        if isinstance(emb_tensors, torch.Tensor):
            emb_vectors = emb_tensors.cpu().numpy()
        else:
            emb_vectors = emb_tensors

        # Convert to Embedding objects
        embeddings = [
            Embedding(emb_vector, model=self.model_name)
            for emb_vector in emb_vectors
        ]

        return embeddings


# Backwards compatibility: Keep ReIDDetector as alias
ReIDDetector = ReIDExtractor
