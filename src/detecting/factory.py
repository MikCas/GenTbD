"""
Factory for creating detector instances from configuration.

This module provides the DetectorFactory class that centralizes detector
instantiation logic, making it easy to add new detector types and keeping
main.py clean.
"""

import logging
from typing import Optional
from ..config import Config
from .detector import Detector
from .detectors.object_detector import ObjectDetector
from .detectors.keypoint_detector import KeypointDetector


class DetectorFactory:
    """
    Factory for creating detector instances based on configuration.

    This factory centralizes all detector creation logic, including:
    - Type-specific instantiation
    - Model validation and auto-selection
    - Parameter extraction from config
    - Error handling and logging

    Example:
        >>> config = Config.from_yaml('config/default.yaml')
        >>> detector = DetectorFactory.create(config, logger)
        >>> detections = detector.detect(image)
    """

    @staticmethod
    def create(config: Config, logger: Optional[logging.Logger] = None) -> Detector:
        """
        Create a detector instance based on configuration.

        Args:
            config: Configuration object containing detection settings
            logger: Optional logger for warnings and info messages

        Returns:
            Detector instance (ObjectDetector, KeypointDetector, etc.)

        Raises:
            ValueError: If detector type is unknown or configuration is invalid

        Example:
            >>> config = Config.from_yaml('config/default.yaml')
            >>> detector = DetectorFactory.create(config)
        """
        detector_type = config.get('detection.type', 'object')
        device = config.get('detection.device', 'cpu')
        conf_threshold = config.get('detection.conf_threshold', 0.5)

        if detector_type == 'keypoint':
            return DetectorFactory._create_keypoint_detector(
                config, device, conf_threshold, logger
            )
        elif detector_type == 'object':
            return DetectorFactory._create_object_detector(
                config, device, conf_threshold, logger
            )
        else:
            raise ValueError(
                f"Unknown detector type: '{detector_type}'. "
                f"Valid options: 'object', 'keypoint'"
            )

    @staticmethod
    def _create_keypoint_detector(
        config: Config,
        device: str,
        conf_threshold: float,
        logger: Optional[logging.Logger]
    ) -> KeypointDetector:
        """
        Create KeypointDetector with model validation.

        KeypointDetector only supports 'resnet50' model. If config specifies
        a different model, it will be overridden with a warning.

        Args:
            config: Configuration object
            device: Device to run detector on ('cpu', 'cuda', 'mps')
            conf_threshold: Confidence threshold for detections
            logger: Optional logger for warnings

        Returns:
            KeypointDetector instance
        """
        model = config.get('detection.model')

        # Validate and auto-select model for keypoint detector
        if model not in ['resnet50', None]:
            if logger:
                logger.warning(
                    f"Model '{model}' not supported for keypoint detector, "
                    f"using 'resnet50'"
                )
            model = 'resnet50'
        elif model is None:
            model = 'resnet50'

        # Get keypoint-specific parameters
        keypoint_threshold = config.get('detection.keypoint.keypoint_threshold', 0.5)

        detector = KeypointDetector(
            model=model,
            device=device,
            conf_threshold=conf_threshold,
            keypoint_threshold=keypoint_threshold
        )

        if logger:
            logger.info(f"Keypoint detector ({model}) loaded on device: {device}")

        return detector

    @staticmethod
    def _create_object_detector(
        config: Config,
        device: str,
        conf_threshold: float,
        logger: Optional[logging.Logger]
    ) -> ObjectDetector:
        """
        Create ObjectDetector with optional class filtering.

        Args:
            config: Configuration object
            device: Device to run detector on ('cpu', 'cuda', 'mps')
            conf_threshold: Confidence threshold for detections
            logger: Optional logger for info messages

        Returns:
            ObjectDetector instance
        """
        model = config.get('detection.model', 'mobilenet')
        classes = config.get('detection.classes', None)

        detector = ObjectDetector(
            model=model,
            device=device,
            conf_threshold=conf_threshold,
            classes=classes
        )

        if logger:
            if classes:
                logger.info(
                    f"Object detector ({model}) loaded on device: {device}, "
                    f"filtering classes: {classes}"
                )
            else:
                logger.info(f"Object detector ({model}) loaded on device: {device}")

        return detector
