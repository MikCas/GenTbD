"""Unit tests for ReIDDetector class."""

import pytest
import numpy as np
from src.detecting.detectors.reid_detector import ReIDDetector
from src.detecting.detection import Detection
from src.core.properties import BoundingBox, Embedding


class TestReIDDetector:
    """Test ReIDDetector functionality."""

    def test_init_osnet_x1_0(self):
        """Test ReIDDetector initializes with osnet_x1_0 model."""
        detector = ReIDDetector(model='osnet_x1_0', device='cpu')

        assert detector.model_name == 'osnet_x1_0'
        assert detector.device == 'cpu'
        assert detector.embedding_dim == 512

    def test_init_invalid_model(self):
        """Test ReIDDetector raises error for invalid model."""
        with pytest.raises(ValueError, match="Unknown model"):
            ReIDDetector(model='invalid_model')

    def test_enhance_adds_embeddings(self):
        """Test enhance() adds embeddings to detections."""
        detector = ReIDDetector(model='osnet_x1_0', device='cpu')

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

        # Enhance with embeddings
        enhanced = detector.enhance(frame, detections)

        # Check embeddings were added
        assert len(enhanced) == 2
        assert 'embedding' in enhanced[0]
        assert 'embedding' in enhanced[1]
        assert isinstance(enhanced[0]['embedding'], Embedding)
        assert isinstance(enhanced[1]['embedding'], Embedding)

    def test_enhance_embedding_dimension(self):
        """Test enhanced embeddings have correct dimension."""
        detector = ReIDDetector(model='osnet_x1_0', device='cpu')

        frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        detections = [
            Detection({
                'bbox': BoundingBox(100, 100, 200, 300),
                'class_id': 1,
                'confidence': 0.9
            })
        ]

        enhanced = detector.enhance(frame, detections)

        # OSNet produces 512-dim embeddings
        assert enhanced[0]['embedding'].dim == 512

    def test_enhance_empty_detections(self):
        """Test enhance() handles empty detections list."""
        detector = ReIDDetector(model='osnet_x1_0', device='cpu')

        frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        detections = []

        enhanced = detector.enhance(frame, detections)

        assert len(enhanced) == 0

    def test_enhance_skips_detections_without_bbox(self):
        """Test enhance() skips detections without bounding boxes."""
        detector = ReIDDetector(model='osnet_x1_0', device='cpu')

        frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        detections = [
            Detection({'class_id': 1, 'confidence': 0.9})  # No bbox
        ]

        enhanced = detector.enhance(frame, detections)

        # Should not add embedding
        assert 'embedding' not in enhanced[0]

    def test_embed_single_crop(self):
        """Test embed() extracts embedding from single crop."""
        detector = ReIDDetector(model='osnet_x1_0', device='cpu')

        # Create person crop (256x128 is standard for ReID)
        crop = np.random.randint(0, 255, (256, 128, 3), dtype=np.uint8)

        embedding = detector.embed(crop)

        assert isinstance(embedding, Embedding)
        assert embedding.dim == 512
        assert embedding.is_normalized()

    def test_embed_batch(self):
        """Test embed_batch() processes multiple crops."""
        detector = ReIDDetector(model='osnet_x1_0', device='cpu')

        # Create multiple crops
        crops = [
            np.random.randint(0, 255, (256, 128, 3), dtype=np.uint8),
            np.random.randint(0, 255, (256, 128, 3), dtype=np.uint8),
            np.random.randint(0, 255, (256, 128, 3), dtype=np.uint8)
        ]

        embeddings = detector.embed_batch(crops)

        assert len(embeddings) == 3
        assert all(isinstance(emb, Embedding) for emb in embeddings)
        assert all(emb.dim == 512 for emb in embeddings)

    def test_embed_batch_empty(self):
        """Test embed_batch() handles empty list."""
        detector = ReIDDetector(model='osnet_x1_0', device='cpu')

        embeddings = detector.embed_batch([])

        assert embeddings == []

    def test_enhance_clips_bbox_to_frame(self):
        """Test enhance() clips bounding boxes to frame bounds."""
        detector = ReIDDetector(model='osnet_x1_0', device='cpu')

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
        enhanced = detector.enhance(frame, detections)

        # Either adds embedding with clipped crop or skips
        assert len(enhanced) == 1

    def test_embedding_model_name(self):
        """Test embeddings store model name."""
        detector = ReIDDetector(model='osnet_x1_0', device='cpu')

        frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        detections = [
            Detection({
                'bbox': BoundingBox(100, 100, 200, 300),
                'class_id': 1,
                'confidence': 0.9
            })
        ]

        enhanced = detector.enhance(frame, detections)

        assert enhanced[0]['embedding'].model == 'osnet_x1_0'

    def test_different_osnet_models(self):
        """Test different OSNet model variants."""
        models = ['osnet_x1_0', 'osnet_x0_75', 'osnet_x0_5']

        for model_name in models:
            detector = ReIDDetector(model=model_name, device='cpu')
            assert detector.model_name == model_name
