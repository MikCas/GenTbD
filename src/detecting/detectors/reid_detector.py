"""ReID-based appearance feature extractor for person re-identification.

This module provides the ReIDDetector class that extracts appearance embeddings
from person crops using deep learning models (OSNet, ResNet50, etc.) via torchreid.
"""

import torch
import numpy as np
import cv2
from typing import List, Optional

from ..detector import Detector
from ..detection import Detection
from ...core.properties import BoundingBox, Embedding

# Lazy import to avoid circular dependency issues
_FeatureExtractor = None

def _get_feature_extractor():
    """Lazy import of torchreid FeatureExtractor."""
    global _FeatureExtractor
    if _FeatureExtractor is None:
        import torchreid
        _FeatureExtractor = torchreid.utils.FeatureExtractor
    return _FeatureExtractor


class ReIDDetector(Detector):
    """
    ReID-based appearance feature extractor.

    Extracts appearance embeddings from person crops for re-identification
    across frames in multi-object tracking. Can operate in two modes:

    1. **Enhancement Mode** (Primary): Add embeddings to existing detections
       ```python
       reid_detector = ReIDDetector(model='osnet_x1_0')
       enhanced = reid_detector.enhance(frame, object_detections)
       ```

    2. **Standalone Mode**: Detect and extract embeddings in one step
       ```python
       reid_detector = ReIDDetector(model='osnet_x1_0')
       detections = reid_detector.detect(frame)  # Returns detections with embeddings
       ```

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
        >>> reid_detector = ReIDDetector(model='osnet_x1_0')
        >>>
        >>> detections = obj_detector.detect(frame)
        >>> detections = reid_detector.enhance(frame, detections)
        >>>
        >>> # Now detections have 'embedding' property
        >>> for det in detections:
        >>>     if 'embedding' in det:
        >>>         print(f"Embedding dim: {det['embedding'].dim}")
    """

    MODELS = {
        'osnet_x1_0': 'osnet_x1_0',
        'osnet_x0_75': 'osnet_x0_75',
        'osnet_x0_5': 'osnet_x0_5',
        'osnet_x0_25': 'osnet_x0_25',
        'osnet_ain_x1_0': 'osnet_ain_x1_0',
        'resnet50': 'resnet50',
    }

    def __init__(self, model='osnet_x1_0', device='cpu', conf_threshold=0.5,
                 embedding_dim=512):
        """
        Initialize ReID detector.

        Args:
            model: Model name from MODELS dict
            device: 'cpu', 'mps', or 'cuda'
            conf_threshold: Minimum confidence (not used in ReID, kept for compatibility)
            embedding_dim: Embedding dimension (512 for OSNet, can vary)

        Raises:
            ValueError: If model name is not recognized
        """
        if model not in self.MODELS:
            valid_models = ', '.join(self.MODELS.keys())
            raise ValueError(f"Unknown model '{model}'. Valid options: {valid_models}")

        # Initialize torchreid FeatureExtractor (lazy import)
        FeatureExtractor = _get_feature_extractor()
        self.extractor = FeatureExtractor(
            model_name=self.MODELS[model],
            model_path=None,  # Use pretrained weights
            device=device
        )

        # Initialize base Detector class
        # Note: We pass the extractor's model, but preprocess/inference/postprocess
        # are overridden to use the extractor directly
        super().__init__(self.extractor.model, device, conf_threshold)

        self.model_name = model
        self.embedding_dim = embedding_dim

    def preprocess(self, image: np.ndarray) -> torch.Tensor:
        """
        Preprocess image crop for ReID model.

        This method is part of the Detector interface but is handled
        internally by torchreid's FeatureExtractor.

        Args:
            image: BGR image crop (H, W, 3)

        Returns:
            Preprocessed tensor (not actually used, handled by extractor)
        """
        # Torchreid's FeatureExtractor handles preprocessing internally
        # This method is here for interface compatibility
        return torch.zeros(1)  # Placeholder

    def inference(self, input_tensor: torch.Tensor) -> dict:
        """
        Extract embeddings from preprocessed tensor.

        This method is part of the Detector interface but is handled
        internally by torchreid's FeatureExtractor.

        Args:
            input_tensor: Preprocessed tensor

        Returns:
            Dict with embeddings (not actually used, handled by extractor)
        """
        # Torchreid's FeatureExtractor handles inference internally
        # This method is here for interface compatibility
        return {'embeddings': []}

    def postprocess(self, output: dict, image_shape: tuple) -> List[Detection]:
        """
        Convert embeddings to Detection objects.

        This method is part of the Detector interface but is not used
        in the typical enhance() workflow.

        Args:
            output: Dict containing embeddings
            image_shape: Original image shape

        Returns:
            List of Detection objects with embeddings
        """
        # This method is here for interface compatibility
        # Real usage is through enhance() method
        return []

    def detect(self, image: np.ndarray) -> List[Detection]:
        """
        Detect and extract embeddings (standalone mode).

        Note: This requires pre-existing detections or an internal person detector.
        For typical usage, use enhance() instead with existing detections.

        Args:
            image: Full frame BGR image

        Returns:
            List of Detection objects (empty in this implementation)
        """
        # Standalone mode not fully implemented
        # Primary usage is enhance() mode
        return []

    def enhance(self, frame: np.ndarray, detections: List[Detection]) -> List[Detection]:
        """
        Add ReID embeddings to existing detections (primary usage mode).

        This is the main method for using ReIDDetector in tracking pipelines:
        1. Object detector finds people → detections with bboxes
        2. ReIDDetector adds embeddings → detections with bboxes + embeddings
        3. Tracker uses both for association

        Args:
            frame: Full frame BGR image (H, W, 3)
            detections: List of Detection objects with 'bbox' property

        Returns:
            Same detections list with 'embedding' property added

        Example:
            >>> detections = object_detector.detect(frame)
            >>> detections = reid_detector.enhance(frame, detections)
            >>> for det in detections:
            >>>     if 'embedding' in det:
            >>>         dist = det['embedding'].cosine_distance(other_emb)
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

            # Skip if crop is too small
            if crop.shape[0] < 10 or crop.shape[1] < 5:
                continue

            crops.append(crop)
            valid_indices.append(i)

        # Extract embeddings in batch
        if crops:
            # Torchreid's FeatureExtractor handles BGR/RGB conversion and normalization
            embeddings = self.extractor(crops)  # Returns torch tensor (N, embedding_dim)

            # Convert to numpy if needed
            if isinstance(embeddings, torch.Tensor):
                embeddings = embeddings.cpu().numpy()

            # Add embeddings to detections
            for idx, emb_vector in zip(valid_indices, embeddings):
                embedding = Embedding(emb_vector, model=self.model_name)
                detections[idx]['embedding'] = embedding

        return detections

    def embed(self, image_crop: np.ndarray) -> Embedding:
        """
        Extract embedding from a single image crop.

        Convenience method for extracting embeddings from individual crops.

        Args:
            image_crop: BGR image crop (H, W, 3)

        Returns:
            Embedding object with L2-normalized vector

        Example:
            >>> crop = frame[y1:y2, x1:x2]
            >>> embedding = reid_detector.embed(crop)
            >>> print(f"Embedding dimension: {embedding.dim}")
        """
        # Extract embedding using torchreid
        emb_tensor = self.extractor([image_crop])[0]  # Returns torch tensor (embedding_dim,)

        # Convert to numpy
        if isinstance(emb_tensor, torch.Tensor):
            emb_vector = emb_tensor.cpu().numpy()
        else:
            emb_vector = emb_tensor

        return Embedding(emb_vector, model=self.model_name)

    def embed_batch(self, image_crops: List[np.ndarray]) -> List[Embedding]:
        """
        Extract embeddings from multiple image crops (batch processing).

        More efficient than calling embed() multiple times.

        Args:
            image_crops: List of BGR image crops

        Returns:
            List of Embedding objects

        Example:
            >>> crops = [frame[y1:y2, x1:x2] for y1, x1, y2, x2 in bboxes]
            >>> embeddings = reid_detector.embed_batch(crops)
        """
        if not image_crops:
            return []

        # Extract embeddings in batch
        emb_tensors = self.extractor(image_crops)  # Returns torch tensor (N, embedding_dim)

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
