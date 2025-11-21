"""Unit tests for ReIDExtractor class."""

import pytest
import numpy as np
from src.detecting.reid import ReIDExtractor
from src.detecting.detection import Detection
from src.core.properties import BoundingBox, Embedding


class TestReIDExtractor:
    """Test ReIDExtractor functionality."""

    def test_init_osnet_x1_0(self):
        """Test ReIDExtractor initializes with osnet_x1_0 model."""
        extractor = ReIDExtractor(model='osnet_x1_0', device='cpu')

        assert extractor.model_name == 'osnet_x1_0'
        assert extractor.device == 'cpu'
        assert extractor.embedding_dim == 512

    def test_init_invalid_model(self):
        """Test ReIDExtractor raises error for invalid model."""
        with pytest.raises(ValueError, match="Unknown model"):
            ReIDExtractor(model='invalid_model')

    def test_enhance_adds_embeddings(self):
        """Test enhance() adds embeddings to detections."""
        extractor = ReIDExtractor(model='osnet_x1_0', device='cpu')

        # Create test frame
        frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

        # Create test detections with bounding boxes
        detections = [
            Detection({
                'bbox': BoundingBox(100, 100, 200, 300),
                'class_id': 1,
                'confidence': 0.9
            }),
            Detection({
                'bbox': BoundingBox(300, 150, 400, 350),
                'class_id': 1,
                'confidence': 0.85
            })
        ]

        # Extract embeddings from detections
        enhanced = extractor.enhance(frame, detections)

        # Check embeddings were added
        assert len(enhanced) == 2
        assert 'embedding' in enhanced[0]
        assert 'embedding' in enhanced[1]
        assert isinstance(enhanced[0]['embedding'], Embedding)
        assert isinstance(enhanced[1]['embedding'], Embedding)

    def test_enhance_embedding_dimension(self):
        """Test enhanced embeddings have correct dimension."""
        extractor = ReIDExtractor(model='osnet_x1_0', device='cpu')

        frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        detections = [
            Detection({
                'bbox': BoundingBox(100, 100, 200, 300),
                'class_id': 1,
                'confidence': 0.9
            })
        ]

        enhanced = extractor.enhance(frame, detections)

        # OSNet produces 512-dim embeddings
        assert enhanced[0]['embedding'].dim == 512

    def test_enhance_empty_detections(self):
        """Test enhance() handles empty detections list."""
        extractor = ReIDExtractor(model='osnet_x1_0', device='cpu')

        frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        detections = []

        enhanced = extractor.enhance(frame, detections)

        assert len(enhanced) == 0

    def test_enhance_skips_detections_without_bbox(self):
        """Test enhance() skips detections without bounding boxes."""
        extractor = ReIDExtractor(model='osnet_x1_0', device='cpu')

        frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        detections = [
            Detection({'class_id': 1, 'confidence': 0.9})  # No bbox
        ]

        enhanced = extractor.enhance(frame, detections)

        # Should not add embedding
        assert 'embedding' not in enhanced[0]

    def test_extract_single_crop(self):
        """Test extract() extracts embedding from single crop."""
        extractor = ReIDExtractor(model='osnet_x1_0', device='cpu')

        # Create person crop (256x128 is standard for ReID)
        crop = np.random.randint(0, 255, (256, 128, 3), dtype=np.uint8)

        embedding = extractor.extract(crop)

        assert isinstance(embedding, Embedding)
        assert embedding.dim == 512
        assert embedding.is_normalized()

    def test_extract_batch(self):
        """Test extract_batch() processes multiple crops."""
        extractor = ReIDExtractor(model='osnet_x1_0', device='cpu')

        # Create multiple crops
        crops = [
            np.random.randint(0, 255, (256, 128, 3), dtype=np.uint8),
            np.random.randint(0, 255, (256, 128, 3), dtype=np.uint8),
            np.random.randint(0, 255, (256, 128, 3), dtype=np.uint8)
        ]

        embeddings = extractor.extract_batch(crops)

        assert len(embeddings) == 3
        assert all(isinstance(emb, Embedding) for emb in embeddings)
        assert all(emb.dim == 512 for emb in embeddings)

    def test_extract_batch_empty(self):
        """Test extract_batch() handles empty list."""
        extractor = ReIDExtractor(model='osnet_x1_0', device='cpu')

        embeddings = extractor.extract_batch([])

        assert embeddings == []

    def test_enhance_clips_bbox_to_frame(self):
        """Test enhance() clips bounding boxes to frame bounds."""
        extractor = ReIDExtractor(model='osnet_x1_0', device='cpu')

        frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

        # Bbox extends beyond frame
        detections = [
            Detection({
                'bbox': BoundingBox(600, 400, 700, 500),  # Beyond frame width
                'class_id': 1,
                'confidence': 0.9
            })
        ]

        # Should handle gracefully (clip or skip)
        enhanced = extractor.enhance(frame, detections)

        # Either adds embedding with clipped crop or skips
        assert len(enhanced) == 1

    def test_embedding_model_name(self):
        """Test embeddings store model name."""
        extractor = ReIDExtractor(model='osnet_x1_0', device='cpu')

        frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        detections = [
            Detection({
                'bbox': BoundingBox(100, 100, 200, 300),
                'class_id': 1,
                'confidence': 0.9
            })
        ]

        enhanced = extractor.enhance(frame, detections)

        assert enhanced[0]['embedding'].model == 'osnet_x1_0'

    def test_different_osnet_models(self):
        """Test different OSNet model variants."""
        models = ['osnet_x1_0', 'osnet_x0_75', 'osnet_x0_5']

        for model_name in models:
            extractor = ReIDExtractor(model=model_name, device='cpu')
            assert extractor.model_name == model_name
