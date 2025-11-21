"""
Model Registry for managing detection models and their metadata.

This module provides a centralized registry for all supported detection models,
replacing ad-hoc string parsing with structured metadata. It supports:
- Programmatic verification of model parameters
- Runtime device support checks
- Lazy loading of metadata
- User-friendly aliases
"""

from dataclasses import dataclass, field
from typing import Dict, Optional, Callable, Type, List, Any
from enum import Enum
import torch

from .detector import Detector
from .detectors.object_detector import ObjectDetector
from .detectors.keypoint_detector import KeypointDetector
from .detectors.yolo_detector import YOLODetector


class ModelBackend(Enum):
    """Backend framework used by the model."""
    TORCHVISION = "torchvision"
    ULTRALYTICS = "ultralytics"
    TORCHREID = "torchreid"


@dataclass
class ModelSpec:
    """
    Model specification with programmatically verified metadata.
    
    Attributes:
        name: Canonical model name (e.g., 'fasterrcnn_resnet50')
        detector_class: The Detector subclass to instantiate
        backend: The framework backend
        model_fn: Optional factory function for TorchVision models
        model_path: Optional path/name for YOLO models
        _params_million: Cached parameter count (lazy loaded)
        _device_support: Cached device support checks
        user_benchmarks: User-provided performance data
    """
    name: str
    detector_class: Type[Detector]
    backend: ModelBackend
    
    # Framework-specific instantiation
    model_fn: Optional[Callable] = None      # TorchVision loader function
    model_path: Optional[str] = None         # YOLO .pt file path
    
    # Programmatic metadata (auto-populated on first access)
    _params_million: Optional[float] = None
    _device_support: Dict[str, bool] = field(default_factory=dict)
    
    # User benchmarks (optional, clearly marked as user-provided)
    user_benchmarks: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def params_million(self) -> float:
        """Get parameter count in millions (lazy-loaded)."""
        if self._params_million is None:
            self._params_million = self._calculate_params()
        return self._params_million
    
    def _calculate_params(self) -> float:
        """Calculate parameters programmatically based on backend."""
        try:
            if self.backend == ModelBackend.TORCHVISION:
                # Use TorchVision's official metadata if available
                weights_class = self._get_weights_class()
                if weights_class:
                    return weights_class.DEFAULT.meta['num_params'] / 1e6
                return 0.0
            
            elif self.backend == ModelBackend.ULTRALYTICS:
                # For YOLO, we'd need to load the model to count params exactly.
                # To avoid heavy loading just for metadata, we might return 0.0 
                # or implement a lightweight check if possible.
                # For now, return 0.0 to avoid IO/loading cost on simple registry access.
                # Actual implementation could load 'yolov8n.pt' if present.
                return 0.0
            
            return 0.0
        except Exception:
            return 0.0
    
    def _get_weights_class(self):
        """Get TorchVision WeightsEnum class for metadata."""
        try:
            from torchvision.models.detection import (
                FasterRCNN_ResNet50_FPN_Weights,
                FasterRCNN_MobileNet_V3_Large_FPN_Weights,
                RetinaNet_ResNet50_FPN_Weights,
                KeypointRCNN_ResNet50_FPN_Weights,
            )
            
            mapping = {
                'fasterrcnn_resnet50': FasterRCNN_ResNet50_FPN_Weights,
                'fasterrcnn_mobilenet': FasterRCNN_MobileNet_V3_Large_FPN_Weights,
                'retinanet_resnet50': RetinaNet_ResNet50_FPN_Weights,
                'keypointrcnn_resnet50': KeypointRCNN_ResNet50_FPN_Weights,
            }
            return mapping.get(self.name)
        except ImportError:
            return None
    
    def check_device_support(self, device: str) -> bool:
        """Check if device is available (runtime verification)."""
        if device not in self._device_support:
            if device == 'cpu':
                self._device_support['cpu'] = True
            elif device == 'cuda':
                self._device_support['cuda'] = torch.cuda.is_available()
            elif device == 'mps':
                self._device_support['mps'] = torch.backends.mps.is_available()
            else:
                self._device_support[device] = False
        
        return self._device_support[device]


class ModelRegistry:
    """Central registry for detection models."""
    
    def __init__(self):
        self._models: Dict[str, ModelSpec] = {}
        self._aliases: Dict[str, str] = {}
    
    def register(self, spec: ModelSpec):
        """Register a model specification."""
        self._models[spec.name] = spec
    
    def add_alias(self, alias: str, canonical: str):
        """Add a user-friendly alias for a model."""
        if canonical not in self._models:
            raise ValueError(f"Cannot alias '{alias}' to unknown model '{canonical}'")
        self._aliases[alias] = canonical
    
    def get(self, name: str) -> ModelSpec:
        """Get model specification by name or alias."""
        canonical = self._aliases.get(name, name)
        if canonical not in self._models:
            # Try to find close matches or list available
            available = sorted(list(self._models.keys()) + list(self._aliases.keys()))
            raise ValueError(f"Unknown model: '{name}'. Available: {', '.join(available)}")
        return self._models[canonical]
    
    def list_models(self) -> List[str]:
        """List all registered model names."""
        return sorted(list(self._models.keys()))


# Global registry instance
_registry = ModelRegistry()

def get_registry() -> ModelRegistry:
    """Get the global model registry instance."""
    return _registry


# --- Registration of Standard Models ---

def _register_standard_models():
    """Register all standard supported models."""
    registry = get_registry()
    
    # 1. TorchVision Object Detection
    try:
        from torchvision.models.detection import (
            fasterrcnn_resnet50_fpn,
            fasterrcnn_mobilenet_v3_large_fpn,
            retinanet_resnet50_fpn,
        )
        
        registry.register(ModelSpec(
            name='fasterrcnn_resnet50',
            detector_class=ObjectDetector,
            backend=ModelBackend.TORCHVISION,
            model_fn=fasterrcnn_resnet50_fpn
        ))
        
        registry.register(ModelSpec(
            name='fasterrcnn_mobilenet',
            detector_class=ObjectDetector,
            backend=ModelBackend.TORCHVISION,
            model_fn=fasterrcnn_mobilenet_v3_large_fpn
        ))
        
        registry.register(ModelSpec(
            name='retinanet_resnet50',
            detector_class=ObjectDetector,
            backend=ModelBackend.TORCHVISION,
            model_fn=retinanet_resnet50_fpn
        ))
        
        # Aliases
        registry.add_alias('resnet50', 'fasterrcnn_resnet50')
        registry.add_alias('mobilenet', 'fasterrcnn_mobilenet')
        registry.add_alias('retinanet', 'retinanet_resnet50')
        
    except ImportError:
        pass  # TorchVision might be missing or old version

    # 2. TorchVision Keypoint Detection
    try:
        from torchvision.models.detection import keypointrcnn_resnet50_fpn
        
        registry.register(ModelSpec(
            name='keypointrcnn_resnet50',
            detector_class=KeypointDetector,
            backend=ModelBackend.TORCHVISION,
            model_fn=keypointrcnn_resnet50_fpn
        ))
        
    except ImportError:
        pass

    # 3. YOLO Models
    yolo_variants = [
        'yolo_v8n', 'yolo_v8s', 'yolo_v8m', 'yolo_v8l', 'yolo_v8x',
        'yolo_v11n', 'yolo_v11s', 'yolo_v11m'
    ]
    
    for yolo_name in yolo_variants:
        # yolo_v8n -> v8n (for model_path/name expected by ultralytics)
        short_name = yolo_name.replace('yolo_', '')
        
        registry.register(ModelSpec(
            name=yolo_name,
            detector_class=YOLODetector,
            backend=ModelBackend.ULTRALYTICS,
            model_path=short_name
        ))
        
        # Add short alias (e.g., 'v8n' -> 'yolo_v8n')
        registry.add_alias(short_name, yolo_name)


# Initialize registry on module import
_register_standard_models()
