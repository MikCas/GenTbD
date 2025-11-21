"""
Tests for the Model Registry system.
"""

import pytest
import torch
from unittest.mock import MagicMock, patch

from src.detecting.model_registry import (
    ModelRegistry, 
    ModelSpec, 
    ModelBackend, 
    get_registry
)
from src.detecting.object.torchvision import TorchVisionObjectDetector
from src.detecting.object.yolo import YOLOObjectDetector

class TestModelRegistry:
    
    def test_registry_singleton(self):
        """Test that get_registry returns the same instance."""
        reg1 = get_registry()
        reg2 = get_registry()
        assert reg1 is reg2
        
    def test_standard_models_registered(self):
        """Test that standard models are registered by default."""
        registry = get_registry()
        
        # Check TorchVision models
        assert registry.get('fasterrcnn_resnet50') is not None
        assert registry.get('fasterrcnn_mobilenet') is not None
        
        # Check YOLO models
        assert registry.get('yolo_v8n') is not None
        
        # Check Aliases
        assert registry.get('mobilenet').name == 'fasterrcnn_mobilenet'
        assert registry.get('v8n').name == 'yolo_v8n'
        
    def test_get_unknown_model(self):
        """Test error when requesting unknown model."""
        registry = get_registry()
        with pytest.raises(ValueError) as excinfo:
            registry.get('non_existent_model')
        assert "Unknown model" in str(excinfo.value)
        
    def test_add_alias(self):
        """Test adding custom aliases."""
        registry = ModelRegistry()
        spec = ModelSpec(
            name='test_model',
            detector_class=TorchVisionObjectDetector,
            backend=ModelBackend.TORCHVISION
        )
        registry.register(spec)
        
        registry.add_alias('my_alias', 'test_model')
        assert registry.get('my_alias') is spec
        
        with pytest.raises(ValueError):
            registry.add_alias('bad_alias', 'missing_model')

class TestModelSpec:
    
    def test_device_support_cpu(self):
        """Test CPU support check."""
        spec = ModelSpec(
            name='test',
            detector_class=TorchVisionObjectDetector,
            backend=ModelBackend.TORCHVISION
        )
        assert spec.check_device_support('cpu') is True
        
    def test_device_support_cuda(self):
        """Test CUDA support check (mocked)."""
        spec = ModelSpec(
            name='test',
            detector_class=TorchVisionObjectDetector,
            backend=ModelBackend.TORCHVISION
        )
        
        with patch('torch.cuda.is_available', return_value=True):
            # Clear cache if needed or create new spec
            spec._device_support = {} 
            assert spec.check_device_support('cuda') is True
            
        with patch('torch.cuda.is_available', return_value=False):
            spec._device_support = {}
            assert spec.check_device_support('cuda') is False

    def test_lazy_params_calculation(self):
        """Test that params are calculated lazily."""
        spec = ModelSpec(
            name='fasterrcnn_resnet50', # Use a real name to trigger internal mapping
            detector_class=TorchVisionObjectDetector,
            backend=ModelBackend.TORCHVISION
        )
        
        # Should be None initially
        assert spec._params_million is None
        
        # Should calculate on access
        # Note: This relies on torchvision being installed. 
        # If not, it returns 0.0, which is also a valid float.
        params = spec.params_million
        assert isinstance(params, float)
        assert spec._params_million is not None
