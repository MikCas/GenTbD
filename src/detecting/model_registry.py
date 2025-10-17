"""
Model Registry for easy detector creation.

This module provides a factory pattern for creating detectors from configuration.
Instead of hardcoding detector initialization, you specify a model name and get
the right detector automatically.

Why Registry Pattern?
- Decouple code from specific detector classes
- Switch models by changing config, not code
- Add new models without modifying existing code
- Centralize model-specific settings

Before:
    detector = YOLOv8PyTorch(
        model_name='yolov8n.pt',
        confidence_threshold=0.3,
        iou_threshold=0.5,
        classes=[0],
        logger=logger
    )

After:
    detector = ModelRegistry.create('yolov8n', logger=logger, **config)

Learning Point - Factory Pattern:
    A factory creates objects without exposing creation logic.
    You ask for "yolov8n" and get the right detector automatically.
"""

from typing import Optional, Dict, Any, List
import logging

from detecting.detectors.Detector import Detector
from detecting.detectors.Detector_PyTorch_YOLO import YOLOv8PyTorch
from detecting.detectors.Detector_ONNX_YOLO7 import YOLOv7ONNX


class ModelRegistry:
    """
    Factory for creating detectors from model names.

    This registry knows about all available models and how to create them.
    It handles the mapping: model_name -> detector class + parameters.

    Example:
        >>> detector = ModelRegistry.create('yolov8n', confidence_threshold=0.3)
        >>> detector = ModelRegistry.create('yolov7-tiny-onnx', classes=[0, 2])
        >>> models = ModelRegistry.list_models()  # See what's available
    """

    # Model definitions
    # Learning Point: This is metadata about each model
    # - name: Identifier used in config
    # - class: Which detector class to instantiate
    # - model_path/model_name: Passed to detector __init__
    # - backend: Which framework (pytorch, onnx)
    # - description: Human-readable info
    _MODELS = {
        # PyTorch YOLO models (Ultralytics)
        'yolov8n': {
            'class': YOLOv8PyTorch,
            'model_name': 'yolov8n.pt',
            'backend': 'pytorch',
            'description': 'YOLOv8 Nano (fastest, 3.2M params, auto-download)',
            'default_confidence': 0.25,
            'default_iou': 0.45,
        },
        'yolov8s': {
            'class': YOLOv8PyTorch,
            'model_name': 'yolov8s.pt',
            'backend': 'pytorch',
            'description': 'YOLOv8 Small (balanced, 11.2M params, auto-download)',
            'default_confidence': 0.25,
            'default_iou': 0.45,
        },
        'yolov8m': {
            'class': YOLOv8PyTorch,
            'model_name': 'yolov8m.pt',
            'backend': 'pytorch',
            'description': 'YOLOv8 Medium (accurate, 25.9M params, auto-download)',
            'default_confidence': 0.25,
            'default_iou': 0.45,
        },
        'yolov8l': {
            'class': YOLOv8PyTorch,
            'model_name': 'yolov8l.pt',
            'backend': 'pytorch',
            'description': 'YOLOv8 Large (very accurate, 43.7M params, auto-download)',
            'default_confidence': 0.25,
            'default_iou': 0.45,
        },
        'yolov8x': {
            'class': YOLOv8PyTorch,
            'model_name': 'yolov8x.pt',
            'backend': 'pytorch',
            'description': 'YOLOv8 Extra (best accuracy, 68.2M params, auto-download)',
            'default_confidence': 0.25,
            'default_iou': 0.45,
        },
        # ONNX YOLO models
        'yolov7-tiny-onnx': {
            'class': YOLOv7ONNX,
            'model_path': 'models/yolov7-tiny_640x640.onnx',
            'backend': 'onnx',
            'description': 'YOLOv7 Tiny ONNX (fast inference, requires model file)',
            'default_confidence': 0.25,
            'default_iou': 0.45,
        },
    }

    @classmethod
    def create(cls,
               model_name: str,
               confidence_threshold: Optional[float] = None,
               iou_threshold: Optional[float] = None,
               classes: Optional[List[int]] = None,
               device: Optional[str] = None,
               logger: Optional[logging.Logger] = None) -> Detector:
        """
        Create a detector from model name.

        Args:
            model_name: Model identifier (e.g., 'yolov8n', 'yolov7-tiny-onnx')
            confidence_threshold: Override default confidence threshold
            iou_threshold: Override default IoU threshold for NMS
            classes: List of class IDs to detect (None = all, [0] = person only)
            device: Device to use ('cpu', 'cuda', 'mps', None = auto)
            logger: Logger instance

        Returns:
            Detector instance ready to use

        Raises:
            ValueError: If model_name not found in registry

        Learning Point - Factory Method:
            This method:
            1. Looks up model metadata
            2. Uses default values if not overridden
            3. Creates the right detector class
            4. Returns ready-to-use instance

            User just needs to know the model name!

        Example:
            >>> # Use all defaults
            >>> detector = ModelRegistry.create('yolov8n')
            >>>
            >>> # Override confidence threshold
            >>> detector = ModelRegistry.create('yolov8n', confidence_threshold=0.5)
            >>>
            >>> # Detect only person class
            >>> detector = ModelRegistry.create('yolov8n', classes=[0])
        """
        if model_name not in cls._MODELS:
            available = ', '.join(cls._MODELS.keys())
            raise ValueError(
                f"Unknown model: '{model_name}'. "
                f"Available models: {available}"
            )

        # Get model metadata
        model_info = cls._MODELS[model_name]
        detector_class = model_info['class']

        # Use defaults from model info if not overridden
        confidence_threshold = confidence_threshold or model_info.get('default_confidence', 0.25)
        iou_threshold = iou_threshold or model_info.get('default_iou', 0.45)
        classes = classes if classes is not None else [0]  # Default to person

        # Build kwargs based on detector type
        # Learning Point: Different detectors need different parameters
        # PyTorch: model_name
        # ONNX: model_path
        kwargs = {
            'confidence_threshold': confidence_threshold,
            'iou_threshold': iou_threshold,
            'classes': classes,
            'logger': logger,
        }

        # Add backend-specific parameters
        if model_info['backend'] == 'pytorch':
            kwargs['model_name'] = model_info['model_name']
            kwargs['device'] = device
        elif model_info['backend'] == 'onnx':
            kwargs['model_path'] = model_info['model_path']

        # Create detector
        if logger:
            logger.info(f"Creating detector: {model_name} ({model_info['description']})")

        return detector_class(**kwargs)

    @classmethod
    def list_models(cls) -> Dict[str, str]:
        """
        List all available models.

        Returns:
            Dictionary mapping model_name -> description

        Example:
            >>> models = ModelRegistry.list_models()
            >>> for name, desc in models.items():
            >>>     print(f"{name}: {desc}")
        """
        return {
            name: info['description']
            for name, info in cls._MODELS.items()
        }

    @classmethod
    def get_model_info(cls, model_name: str) -> Dict[str, Any]:
        """
        Get detailed information about a model.

        Args:
            model_name: Model identifier

        Returns:
            Dictionary with model metadata

        Raises:
            ValueError: If model not found

        Example:
            >>> info = ModelRegistry.get_model_info('yolov8n')
            >>> print(f"Backend: {info['backend']}")
            >>> print(f"Default confidence: {info['default_confidence']}")
        """
        if model_name not in cls._MODELS:
            raise ValueError(f"Unknown model: '{model_name}'")
        return cls._MODELS[model_name].copy()

    @classmethod
    def register_model(cls,
                      name: str,
                      detector_class: type,
                      description: str,
                      backend: str,
                      default_confidence: float = 0.25,
                      default_iou: float = 0.45,
                      **kwargs) -> None:
        """
        Register a new model.

        This allows users to add custom models without modifying this file.

        Args:
            name: Model identifier
            detector_class: Detector class to instantiate
            description: Human-readable description
            backend: 'pytorch', 'onnx', etc.
            default_confidence: Default confidence threshold
            default_iou: Default IoU threshold
            **kwargs: Additional model-specific parameters

        Learning Point - Open/Closed Principle:
            "Open for extension, closed for modification"
            Users can add new models without editing this file!

        Example:
            >>> # Register a custom model
            >>> ModelRegistry.register_model(
            >>>     name='my-yolo',
            >>>     detector_class=MyYOLODetector,
            >>>     description='My custom YOLO model',
            >>>     backend='pytorch',
            >>>     model_path='models/my_yolo.pt'
            >>> )
            >>>
            >>> # Now use it like any other model
            >>> detector = ModelRegistry.create('my-yolo')
        """
        cls._MODELS[name] = {
            'class': detector_class,
            'description': description,
            'backend': backend,
            'default_confidence': default_confidence,
            'default_iou': default_iou,
            **kwargs
        }
