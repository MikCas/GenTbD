"""Unit tests for Embedding class."""

import pytest
import numpy as np
from src.core.properties import Embedding


class TestEmbedding:
    """Test Embedding functionality."""

    def test_initialization(self):
        """Test Embedding initializes correctly."""
        vector = np.random.randn(512)
        emb = Embedding(vector, model='osnet_x1_0')

        assert emb.dim == 512
        assert emb.model == 'osnet_x1_0'
        assert emb.is_normalized()

    def test_initialization_auto_normalizes(self):
        """Test Embedding auto-normalizes vector to unit length."""
        vector = np.array([3.0, 4.0])  # Length 5
        emb = Embedding(vector)

        # Should be normalized to [0.6, 0.8]
        assert np.allclose(emb.vector, [0.6, 0.8])
        assert emb.is_normalized()

    def test_initialization_zero_vector(self):
        """Test Embedding handles zero vector gracefully."""
        vector = np.zeros(512)
        emb = Embedding(vector)

        assert np.allclose(emb.vector, np.zeros(512))
        assert emb.dim == 512

    def test_initialization_invalid_not_array(self):
        """Test Embedding raises error for non-array input."""
        with pytest.raises(ValueError, match="must be a numpy array"):
            Embedding([1, 2, 3])

    def test_initialization_invalid_not_1d(self):
        """Test Embedding raises error for non-1D array."""
        with pytest.raises(ValueError, match="must be 1D"):
            Embedding(np.random.randn(10, 10))

    def test_vector_property_returns_copy(self):
        """Test vector property returns a copy, not reference."""
        vector = np.random.randn(512)
        emb = Embedding(vector)

        vec_copy = emb.vector
        vec_copy[0] = 999.0

        # Original should be unchanged
        assert emb.vector[0] != 999.0

    def test_dim_property(self):
        """Test dim property returns correct dimensionality."""
        emb_128 = Embedding(np.random.randn(128))
        emb_512 = Embedding(np.random.randn(512))
        emb_2048 = Embedding(np.random.randn(2048))

        assert emb_128.dim == 128
        assert emb_512.dim == 512
        assert emb_2048.dim == 2048

    def test_model_property(self):
        """Test model property returns correct model name."""
        emb1 = Embedding(np.random.randn(512), model='osnet_x1_0')
        emb2 = Embedding(np.random.randn(512), model='resnet50')
        emb3 = Embedding(np.random.randn(512))  # default

        assert emb1.model == 'osnet_x1_0'
        assert emb2.model == 'resnet50'
        assert emb3.model == 'unknown'

    def test_cosine_distance_identical(self):
        """Test cosine distance between identical embeddings is 0."""
        vector = np.random.randn(512)
        emb1 = Embedding(vector)
        emb2 = Embedding(vector)

        dist = emb1.cosine_distance(emb2)

        assert abs(dist) < 1e-6  # Should be ~0

    def test_cosine_distance_opposite(self):
        """Test cosine distance between opposite embeddings is 2."""
        vector = np.array([1.0, 0.0])
        emb1 = Embedding(vector)
        emb2 = Embedding(-vector)

        dist = emb1.cosine_distance(emb2)

        assert abs(dist - 2.0) < 1e-6  # Should be ~2

    def test_cosine_distance_orthogonal(self):
        """Test cosine distance between orthogonal embeddings is 1."""
        emb1 = Embedding(np.array([1.0, 0.0]))
        emb2 = Embedding(np.array([0.0, 1.0]))

        dist = emb1.cosine_distance(emb2)

        assert abs(dist - 1.0) < 1e-6  # Should be ~1

    def test_cosine_distance_dimension_mismatch(self):
        """Test cosine distance raises error for dimension mismatch."""
        emb1 = Embedding(np.random.randn(512))
        emb2 = Embedding(np.random.randn(256))

        with pytest.raises(ValueError, match="dimensions must match"):
            emb1.cosine_distance(emb2)

    def test_euclidean_distance_identical(self):
        """Test Euclidean distance between identical embeddings is 0."""
        vector = np.random.randn(512)
        emb1 = Embedding(vector)
        emb2 = Embedding(vector)

        dist = emb1.euclidean_distance(emb2)

        assert abs(dist) < 1e-6

    def test_euclidean_distance_opposite(self):
        """Test Euclidean distance between opposite unit vectors is 2."""
        vector = np.array([1.0, 0.0])
        emb1 = Embedding(vector)
        emb2 = Embedding(-vector)

        dist = emb1.euclidean_distance(emb2)

        # For unit vectors, ||a - (-a)|| = ||2a|| = 2
        assert abs(dist - 2.0) < 1e-6

    def test_euclidean_distance_dimension_mismatch(self):
        """Test Euclidean distance raises error for dimension mismatch."""
        emb1 = Embedding(np.random.randn(512))
        emb2 = Embedding(np.random.randn(256))

        with pytest.raises(ValueError, match="dimensions must match"):
            emb1.euclidean_distance(emb2)

    def test_similarity_identical(self):
        """Test similarity between identical embeddings is 1."""
        vector = np.random.randn(512)
        emb1 = Embedding(vector)
        emb2 = Embedding(vector)

        sim = emb1.similarity(emb2)

        assert abs(sim - 1.0) < 1e-6

    def test_similarity_opposite(self):
        """Test similarity between opposite embeddings is -1."""
        vector = np.array([1.0, 0.0])
        emb1 = Embedding(vector)
        emb2 = Embedding(-vector)

        sim = emb1.similarity(emb2)

        assert abs(sim - (-1.0)) < 1e-6

    def test_similarity_orthogonal(self):
        """Test similarity between orthogonal embeddings is 0."""
        emb1 = Embedding(np.array([1.0, 0.0]))
        emb2 = Embedding(np.array([0.0, 1.0]))

        sim = emb1.similarity(emb2)

        assert abs(sim) < 1e-6

    def test_similarity_dimension_mismatch(self):
        """Test similarity raises error for dimension mismatch."""
        emb1 = Embedding(np.random.randn(512))
        emb2 = Embedding(np.random.randn(256))

        with pytest.raises(ValueError, match="dimensions must match"):
            emb1.similarity(emb2)

    def test_is_normalized(self):
        """Test is_normalized correctly identifies normalized vectors."""
        vector = np.random.randn(512)
        emb = Embedding(vector)

        assert emb.is_normalized()

        # Manually check
        norm = np.linalg.norm(emb.vector)
        assert abs(norm - 1.0) < 1e-6

    def test_repr(self):
        """Test string representation contains key information."""
        emb = Embedding(np.random.randn(512), model='osnet_x1_0')

        repr_str = repr(emb)

        assert 'Embedding' in repr_str
        assert 'dim=512' in repr_str
        assert "model='osnet_x1_0'" in repr_str
        assert 'norm=' in repr_str

    def test_equality_same_vector(self):
        """Test equality for same vector."""
        vector = np.random.randn(512)
        emb1 = Embedding(vector)
        emb2 = Embedding(vector)

        assert emb1 == emb2

    def test_equality_different_vectors(self):
        """Test inequality for different vectors."""
        emb1 = Embedding(np.random.randn(512))
        emb2 = Embedding(np.random.randn(512))

        assert emb1 != emb2

    def test_equality_different_dimensions(self):
        """Test inequality for different dimensions."""
        emb1 = Embedding(np.random.randn(512))
        emb2 = Embedding(np.random.randn(256))

        assert emb1 != emb2

    def test_equality_non_embedding(self):
        """Test inequality when comparing to non-Embedding."""
        emb = Embedding(np.random.randn(512))

        assert emb != "not an embedding"
        assert emb != 42
        assert emb != None

    def test_typical_reid_workflow(self):
        """Test typical ReID workflow: extract → compare → threshold."""
        # Simulate two embeddings from same person
        base_vector = np.random.randn(512)
        noise = np.random.randn(512) * 0.1  # Small noise

        emb1 = Embedding(base_vector, model='osnet_x1_0')
        emb2 = Embedding(base_vector + noise, model='osnet_x1_0')

        # Compute distance
        dist = emb1.cosine_distance(emb2)

        # Should be close (same person)
        assert dist < 0.5  # Typical threshold is 0.2-0.4

        # Simulate embedding from different person
        emb3 = Embedding(np.random.randn(512), model='osnet_x1_0')
        dist_different = emb1.cosine_distance(emb3)

        # Should be larger
        assert dist_different > dist

    def test_different_embedding_dimensions(self):
        """Test Embedding works with various common dimensions."""
        for dim in [128, 256, 512, 1024, 2048]:
            emb = Embedding(np.random.randn(dim))
            assert emb.dim == dim
            assert emb.is_normalized()
