"""
Abstract base class for model inference backends.

This module provides the interface that all backend implementations must follow.
Backends handle model loading and inference for different frameworks (PyTorch, ONNX, TensorFlow, etc.).

Key Concept: Separation of Concerns
- Backend: HOW to load and run a model (PyTorch, ONNX, etc.)
- Detector: WHAT to do with the results (create Detection objects)
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Union
import numpy as np
import logging


class Backend(ABC):
    """
    Abstract base class for model inference backends.

    A backend handles:
    1. Loading models from different sources (files, URLs, HuggingFace, etc.)
    2. Running inference (forward pass)
    3. Managing device (CPU, CUDA, MPS)
    4. Providing model metadata

    Why this abstraction?
    - Easy to swap between PyTorch, ONNX, TensorFlow, etc.
    - Detector code doesn't need to know about framework details
    - Can add new backends without changing detector code

    Example:
        >>> backend = PyTorchBackend()
        >>> backend.load_model('yolov8n.pt')
        >>> outputs = backend.inference(image)
    """

    def __init__(self, logger: Optional[logging.Logger] = None):
        """
        Initialize backend.

        Args:
            logger: Logger instance for logging messages
        """
        self._logger = logger
        self._model = None
        self._device = None

    @abstractmethod
    def load_model(self, model_spec: Union[str, Dict[str, Any]]) -> None:
        """
        Load a model.

        Args:
            model_spec: Either:
                - str: Path to model file (e.g., 'models/yolov8n.pt')
                - dict: Configuration dict for more complex loading
                  Example: {'type': 'ultralytics', 'model': 'yolov8n'}

        Raises:
            FileNotFoundError: If model file doesn't exist
            ValueError: If model specification is invalid
        """
        pass

    @abstractmethod
    def inference(self, image: Any) -> Any:
        """
        Run inference on an image.

        Args:
            image: Input image (format depends on backend)
                - PyTorch: torch.Tensor or numpy array
                - ONNX: numpy array

        Returns:
            Model outputs (format depends on backend and model)

        Note:
            This should handle:
            - Device placement (moving image to GPU if needed)
            - Inference mode (e.g., torch.no_grad() for PyTorch)
            - Batch dimension handling
        """
        pass

    @abstractmethod
    def get_device(self) -> str:
        """
        Get the device the model is running on.

        Returns:
            Device string ('cpu', 'cuda', 'mps', etc.)
        """
        pass

    def get_input_shape(self) -> Optional[tuple]:
        """
        Get expected input shape for the model.

        Returns:
            Tuple of (batch, channels, height, width) or None if not available
            Example: (1, 3, 640, 640)
        """
        return None

    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the loaded model.

        Returns:
            Dictionary with model metadata:
                - 'type': Backend type (e.g., 'pytorch', 'onnx')
                - 'device': Device model is on
                - 'input_shape': Expected input shape
                - Any other backend-specific info
        """
        return {
            'type': self.__class__.__name__,
            'device': self.get_device(),
            'input_shape': self.get_input_shape()
        }

    def log(self, level: int, message: str) -> None:
        """
        Log a message.

        Args:
            level: Logging level (e.g., logging.INFO)
            message: Message to log
        """
        if self._logger:
            self._logger.log(level, message)
