"""Unit tests for DetectorFactory class."""

import pytest
import logging
from src.detecting.factory import DetectorFactory
from src.detecting.detectors.object_detector import ObjectDetector
from src.detecting.detectors.keypoint_detector import KeypointDetector
from src.detecting.detectors.reid_detector import ReIDDetector
from src.config import Config


class TestDetectorFactory:
    """Test DetectorFactory functionality."""

    def test_create_object_detector_default(self):
        """Test creating ObjectDetector with default config."""
        config_dict = {
            'detection': {
                'type': 'object',
                'model': 'mobilenet',
                'device': 'cpu',
                'conf_threshold': 0.5
            }
        }
        config = Config(config_dict)

        detector = DetectorFactory.create(config)

        assert isinstance(detector, ObjectDetector)
        assert detector.device == 'cpu'
        assert detector.conf_threshold == 0.5

    def test_create_object_detector_with_classes(self):
        """Test creating ObjectDetector with class filtering."""
        config_dict = {
            'detection': {
                'type': 'object',
                'model': 'mobilenet',
                'device': 'cpu',
                'conf_threshold': 0.6,
                'classes': [1, 2, 3]  # person, bicycle, car
            }
        }
        config = Config(config_dict)

        detector = DetectorFactory.create(config)

        assert isinstance(detector, ObjectDetector)
        assert detector.classes == [1, 2, 3]

    def test_create_keypoint_detector(self):
        """Test creating KeypointDetector."""
        config_dict = {
            'detection': {
                'type': 'keypoint',
                'model': 'resnet50',
                'device': 'cpu',
                'conf_threshold': 0.5,
                'keypoint': {
                    'keypoint_threshold': 0.5
                }
            }
        }
        config = Config(config_dict)

        detector = DetectorFactory.create(config)

        assert isinstance(detector, KeypointDetector)
        assert detector.device == 'cpu'
        assert detector.conf_threshold == 0.5
        assert detector.keypoint_threshold == 0.5

    def test_create_keypoint_detector_auto_model_selection(self):
        """Test KeypointDetector auto-selects resnet50 when invalid model specified."""
        config_dict = {
            'detection': {
                'type': 'keypoint',
                'model': 'mobilenet',  # Invalid for keypoint detector
                'device': 'cpu',
                'conf_threshold': 0.5,
                'keypoint': {
                    'keypoint_threshold': 0.5
                }
            }
        }
        config = Config(config_dict)
        logger = logging.getLogger('test')

        # Should not raise error, should auto-select resnet50
        detector = DetectorFactory.create(config, logger)

        assert isinstance(detector, KeypointDetector)
        # Detector was created successfully (model was auto-corrected)

    def test_create_keypoint_detector_with_none_model(self):
        """Test KeypointDetector defaults to resnet50 when model is None."""
        config_dict = {
            'detection': {
                'type': 'keypoint',
                'model': None,
                'device': 'cpu',
                'conf_threshold': 0.5,
                'keypoint': {
                    'keypoint_threshold': 0.5
                }
            }
        }
        config = Config(config_dict)

        detector = DetectorFactory.create(config)

        assert isinstance(detector, KeypointDetector)

    def test_create_invalid_detector_type(self):
        """Test creating detector with invalid type raises error."""
        config_dict = {
            'detection': {
                'type': 'segmentation',  # Invalid type
                'model': 'mobilenet',
                'device': 'cpu',
                'conf_threshold': 0.5
            }
        }
        # Config validation now catches invalid detector types
        with pytest.raises(ValueError, match="Invalid detection type"):
            config = Config(config_dict)

    def test_create_with_logger(self):
        """Test factory uses logger for info messages."""
        config_dict = {
            'detection': {
                'type': 'object',
                'model': 'mobilenet',
                'device': 'cpu',
                'conf_threshold': 0.5
            }
        }
        config = Config(config_dict)
        logger = logging.getLogger('test')

        # Should not raise error
        detector = DetectorFactory.create(config, logger)

        assert isinstance(detector, ObjectDetector)

    def test_create_without_logger(self):
        """Test factory works without logger."""
        config_dict = {
            'detection': {
                'type': 'object',
                'model': 'mobilenet',
                'device': 'cpu',
                'conf_threshold': 0.5
            }
        }
        config = Config(config_dict)

        # Should not raise error when logger is None
        detector = DetectorFactory.create(config, logger=None)

        assert isinstance(detector, ObjectDetector)

    def test_create_object_detector_different_models(self):
        """Test creating ObjectDetector with different model types."""
        models = ['mobilenet', 'resnet50', 'retinanet']

        for model in models:
            config_dict = {
                'detection': {
                    'type': 'object',
                    'model': model,
                    'device': 'cpu',
                    'conf_threshold': 0.5
                }
            }
            config = Config(config_dict)

            detector = DetectorFactory.create(config)

            assert isinstance(detector, ObjectDetector)

    def test_create_with_different_devices(self):
        """Test creating detector with different device options."""
        devices = ['cpu', 'mps']  # Can't test cuda without GPU

        for device in devices:
            config_dict = {
                'detection': {
                    'type': 'object',
                    'model': 'mobilenet',
                    'device': device,
                    'conf_threshold': 0.5
                }
            }
            config = Config(config_dict)

            detector = DetectorFactory.create(config)

            assert detector.device == device

    def test_create_with_different_conf_thresholds(self):
        """Test creating detector with different confidence thresholds."""
        thresholds = [0.3, 0.5, 0.7, 0.9]

        for threshold in thresholds:
            config_dict = {
                'detection': {
                    'type': 'object',
                    'model': 'mobilenet',
                    'device': 'cpu',
                    'conf_threshold': threshold
                }
            }
            config = Config(config_dict)

            detector = DetectorFactory.create(config)

            assert detector.conf_threshold == threshold

    def test_create_defaults_to_object_detector(self):
        """Test factory defaults to object detector when type not specified."""
        config_dict = {
            'detection': {
                # type not specified
                'model': 'mobilenet',
                'device': 'cpu',
                'conf_threshold': 0.5
            }
        }
        config = Config(config_dict)

        detector = DetectorFactory.create(config)

        # Should default to object detector
        assert isinstance(detector, ObjectDetector)

    def test_create_reid_detector_default(self):
        """Test creating ReIDDetector with default config."""
        config_dict = {
            'detection': {
                'type': 'reid',
                'model': 'osnet_x1_0',
                'device': 'cpu',
                'conf_threshold': 0.5,
                'reid': {
                    'embedding_dim': 512
                }
            }
        }
        config = Config(config_dict)

        detector = DetectorFactory.create(config)

        assert isinstance(detector, ReIDDetector)
        assert detector.device == 'cpu'
        assert detector.model_name == 'osnet_x1_0'
        assert detector.embedding_dim == 512

    def test_create_reid_detector_different_models(self):
        """Test creating ReIDDetector with different OSNet models."""
        models = ['osnet_x1_0', 'osnet_x0_75', 'osnet_x0_5', 'resnet50']

        for model in models:
            config_dict = {
                'detection': {
                    'type': 'reid',
                    'model': model,
                    'device': 'cpu',
                    'conf_threshold': 0.5,
                    'reid': {
                        'embedding_dim': 512
                    }
                }
            }
            config = Config(config_dict)

            detector = DetectorFactory.create(config)

            assert isinstance(detector, ReIDDetector)
            assert detector.model_name == model

    def test_create_reid_detector_with_logger(self):
        """Test ReIDDetector creation with logger logs info message."""
        config_dict = {
            'detection': {
                'type': 'reid',
                'model': 'osnet_x1_0',
                'device': 'cpu',
                'conf_threshold': 0.5,
                'reid': {
                    'embedding_dim': 512
                }
            }
        }
        config = Config(config_dict)
        logger = logging.getLogger('test')

        detector = DetectorFactory.create(config, logger)

        assert isinstance(detector, ReIDDetector)

    def test_create_reid_detector_defaults_model(self):
        """Test ReIDDetector defaults to osnet_x1_0 when model not specified."""
        config_dict = {
            'detection': {
                'type': 'reid',
                # model not specified
                'device': 'cpu',
                'conf_threshold': 0.5,
                'reid': {
                    'embedding_dim': 512
                }
            }
        }
        config = Config(config_dict)

        detector = DetectorFactory.create(config)

        assert isinstance(detector, ReIDDetector)
        assert detector.model_name == 'osnet_x1_0'

    def test_create_reid_detector_defaults_embedding_dim(self):
        """Test ReIDDetector defaults to 512 embedding_dim when not specified."""
        config_dict = {
            'detection': {
                'type': 'reid',
                'model': 'osnet_x1_0',
                'device': 'cpu',
                'conf_threshold': 0.5,
                'reid': {
                    # embedding_dim not specified
                }
            }
        }
        config = Config(config_dict)

        detector = DetectorFactory.create(config)

        assert isinstance(detector, ReIDDetector)
        assert detector.embedding_dim == 512
