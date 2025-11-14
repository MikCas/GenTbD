"""
ReID appearance embedding vector representation.

This module provides the Embedding class for storing and comparing
appearance feature vectors extracted by ReID models.
"""

import numpy as np
from typing import Optional


class Embedding:
    """
    Encapsulates a Re-Identification (ReID) appearance embedding vector.

    Embeddings are L2-normalized feature vectors that encode the visual
    appearance of detected objects (typically people). They enable identity
    matching across frames in multi-object tracking.

    Attributes:
        vector: L2-normalized embedding vector (typically 512-dim)
        model: Name of the ReID model that generated this embedding

    Example:
        >>> emb1 = Embedding(np.random.randn(512), model='osnet_x1_0')
        >>> emb2 = Embedding(np.random.randn(512), model='osnet_x1_0')
        >>> distance = emb1.cosine_distance(emb2)
        >>> is_same_person = distance < 0.3
    """

    def __init__(self, vector: np.ndarray, model: str = 'unknown'):
        """
        Initialize embedding with L2 normalization.

        Args:
            vector: Raw embedding vector (will be L2-normalized)
            model: Name of the model that generated this embedding

        Raises:
            ValueError: If vector is not 1D numpy array
        """
        if not isinstance(vector, np.ndarray):
            raise ValueError("Embedding vector must be a numpy array")

        if len(vector.shape) != 1:
            raise ValueError(f"Embedding must be 1D, got shape {vector.shape}")

        # L2 normalize to unit length
        norm = np.linalg.norm(vector)
        if norm > 0:
            self._vector = vector / norm
        else:
            self._vector = vector.copy()

        self._model = model

    @property
    def vector(self) -> np.ndarray:
        """
        Get a copy of the embedding vector.

        Returns:
            L2-normalized embedding vector
        """
        return self._vector.copy()

    @property
    def dim(self) -> int:
        """
        Get embedding dimensionality.

        Returns:
            Number of dimensions (e.g., 512)
        """
        return len(self._vector)

    @property
    def model(self) -> str:
        """
        Get the model name that generated this embedding.

        Returns:
            Model name string
        """
        return self._model

    def cosine_distance(self, other: 'Embedding') -> float:
        """
        Compute cosine distance to another embedding.

        Cosine distance = 1 - cosine_similarity
        Range: [0, 2] where 0 = identical, 2 = opposite

        Args:
            other: Another Embedding object

        Returns:
            Cosine distance in range [0, 2]

        Raises:
            ValueError: If embeddings have different dimensions
        """
        if self.dim != other.dim:
            raise ValueError(
                f"Embedding dimensions must match: {self.dim} vs {other.dim}"
            )

        # Since vectors are L2-normalized, dot product = cosine similarity
        cosine_sim = np.dot(self._vector, other._vector)
        return 1.0 - cosine_sim

    def euclidean_distance(self, other: 'Embedding') -> float:
        """
        Compute Euclidean distance to another embedding.

        Args:
            other: Another Embedding object

        Returns:
            Euclidean (L2) distance

        Raises:
            ValueError: If embeddings have different dimensions
        """
        if self.dim != other.dim:
            raise ValueError(
                f"Embedding dimensions must match: {self.dim} vs {other.dim}"
            )

        return float(np.linalg.norm(self._vector - other._vector))

    def similarity(self, other: 'Embedding') -> float:
        """
        Compute cosine similarity to another embedding.

        Range: [-1, 1] where 1 = identical, -1 = opposite, 0 = orthogonal

        Args:
            other: Another Embedding object

        Returns:
            Cosine similarity in range [-1, 1]

        Raises:
            ValueError: If embeddings have different dimensions
        """
        if self.dim != other.dim:
            raise ValueError(
                f"Embedding dimensions must match: {self.dim} vs {other.dim}"
            )

        return float(np.dot(self._vector, other._vector))

    def is_normalized(self, tolerance: float = 1e-6) -> bool:
        """
        Check if embedding is L2-normalized to unit length.

        Args:
            tolerance: Maximum deviation from unit length

        Returns:
            True if ||vector|| ≈ 1.0
        """
        norm = np.linalg.norm(self._vector)
        return abs(norm - 1.0) < tolerance

    def __repr__(self) -> str:
        """String representation for debugging."""
        return f"Embedding(dim={self.dim}, model='{self._model}', norm={np.linalg.norm(self._vector):.4f})"

    def __eq__(self, other) -> bool:
        """
        Check equality (embeddings are considered equal if vectors are very close).

        Args:
            other: Another Embedding object

        Returns:
            True if vectors are nearly identical
        """
        if not isinstance(other, Embedding):
            return False

        if self.dim != other.dim:
            return False

        return np.allclose(self._vector, other._vector, atol=1e-6)
