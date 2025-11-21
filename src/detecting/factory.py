"""
Factory for creating detector instances from configuration.

This module provides the DetectorFactory class that centralizes detector
instantiation logic, making it easy to add new detector types and keeping
main.py clean.
"""

import logging
from typing import Optional
from ..config import Config
from .base import Detector
from .keypoint.torchvision import TorchVisionKeypointDetector
from .model_registry import get_registry, ModelBackend


class DetectorFactory:
    """
    Factory for creating detector instances based on configuration.

    This factory centralizes all detector creation logic, including:
    - Type-specific instantiation
    - Model validation and auto-selection via ModelRegistry
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
        detector_type = config.get('detection.type', 'object_detection')
        device = config.get('detection.device', 'cpu')
        conf_threshold = config.get('detection.conf_threshold', 0.5)
        model_name = config.get('detection.model', 'fasterrcnn_mobilenet')

        # Route to appropriate factory method
        if detector_type == 'keypoint_detection':
            return DetectorFactory._create_keypoint_detector(
                config, device, conf_threshold, logger
            )
        elif detector_type == 'object_detection':
            return DetectorFactory._create_object_detector(
                model_name, config, device, conf_threshold, logger
            )
        else:
            raise ValueError(
                f"Unknown detector type: '{detector_type}'. "
                f"Valid options: 'object_detection', 'keypoint_detection'"
            )

    @staticmethod
    def _create_keypoint_detector(
        config: Config,
        device: str,
        conf_threshold: float,
        logger: Optional[logging.Logger]
    ) -> TorchVisionKeypointDetector:
        """
        Create TorchVisionKeypointDetector with model validation.

        TorchVisionKeypointDetector only supports 'keypointrcnn_resnet50'. If config specifies
        a different model, it will be overridden with a warning.
        """
        model = config.get('detection.model', 'keypointrcnn_resnet50')

        # Extract base model (only resnet50 supported)
        if model == 'keypointrcnn_resnet50':
            base_model = 'resnet50'
        else:
            if logger:
                logger.warning(
                    f"Model '{model}' not supported for keypoint detector, "
                    f"using 'keypointrcnn_resnet50'"
                )
            model = 'keypointrcnn_resnet50'
            base_model = 'resnet50'

        # Get keypoint-specific parameters
        keypoint_threshold = config.get('detection.keypoint.keypoint_threshold', 0.5)

        detector = TorchVisionKeypointDetector(
            model=base_model,
            device=device,
            conf_threshold=conf_threshold,
            keypoint_threshold=keypoint_threshold
        )

        if logger:
            logger.info(f"Keypoint detector ({model}) loaded on device: {device}")

        return detector

    @staticmethod
    def _create_object_detector(
        model_name: str,
        config: Config,
        device: str,
        conf_threshold: float,
        logger: Optional[logging.Logger]
    ) -> Detector:
        """
        Create Object Detector (FasterRCNN or YOLO) using ModelRegistry.
        """
        registry = get_registry()
        
        # Lookup model spec (handles aliases automatically)
        try:
            spec = registry.get(model_name)
        except ValueError as e:
            # Re-raise with helpful context if needed, or let it bubble up
            raise ValueError(f"Failed to create detector: {e}")

        # Check device support
        if not spec.check_device_support(device):
            if logger:
                logger.warning(f"Device '{device}' not supported for {spec.name}, falling back to CPU")
            device = 'cpu'

        classes = config.get('detection.classes', None)

        # Instantiate based on backend
        if spec.backend == ModelBackend.ULTRALYTICS:
            # YOLO models
            detector = spec.detector_class(
                model=spec.model_path, # e.g., 'v8n'
                device=device,
                conf_threshold=conf_threshold,
                classes=classes
            )
        elif spec.backend == ModelBackend.TORCHVISION:
            # TorchVision models need the base name (e.g., 'mobilenet')
            # We can extract it from the canonical name 'fasterrcnn_mobilenet' -> 'mobilenet'
            # Or rely on the detector class to handle it.
            # Currently ObjectDetector expects 'mobilenet', 'resnet50', etc.
            
            # Simple mapping strategy:
            base_model = spec.name.replace('fasterrcnn_', '').replace('retinanet_', '')
            
            detector = spec.detector_class(
                model=base_model,
                device=device,
                conf_threshold=conf_threshold,
                classes=classes
            )
        else:
            raise ValueError(f"Unsupported backend: {spec.backend}")

        if logger:
            # Log with verified metadata
            params_str = f"{spec.params_million:.1f}M params" if spec.params_million > 0 else "unknown params"
            logger.info(
                f"Loaded {spec.name} ({params_str}) on {device}" + 
                (f", filtering classes: {classes}" if classes else "")
            )

        return detector
