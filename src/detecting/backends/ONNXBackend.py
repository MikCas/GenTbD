"""
ONNX backend for model inference.

This module implements the Backend interface for ONNX Runtime models.
ONNX (Open Neural Network Exchange) is a framework-agnostic format for neural networks.

Why ONNX?
- Faster inference than PyTorch (optimized)
- Smaller file size
- Cross-platform (works everywhere)
- No PyTorch dependency needed for deployment

Trade-offs vs PyTorch:
+ Faster inference
+ Smaller models
- Less flexible (model is frozen)
- Harder to modify
- Manual conversion needed (PyTorch → ONNX)
"""

import onnxruntime as ort
import numpy as np
from typing import Union, Dict, Any, Optional, List
import logging
from pathlib import Path

from .Backend import Backend


class ONNXBackend(Backend):
    """
    ONNX Runtime implementation of the Backend interface.

    Execution Providers (in priority order):
    1. CoreML (Apple Silicon) - Best for Mac M1/M2/M3
    2. CUDA (NVIDIA GPU) - Best for NVIDIA GPUs
    3. OpenVINO (Intel CPU/GPU) - Best for Intel hardware
    4. DirectML (Windows GPU) - Best for Windows with any GPU
    5. CPU - Fallback, works everywhere

    Example:
        >>> backend = ONNXBackend()
        >>> backend.load_model('models/yolov7-tiny_640x640.onnx')
        >>> outputs = backend.inference(preprocessed_image)
    """

    def __init__(self,
                 providers: Optional[List[str]] = None,
                 logger: Optional[logging.Logger] = None):
        """
        Initialize ONNX backend.

        Args:
            providers: List of execution providers to try, in priority order.
                      If None, uses platform-appropriate defaults.
                      Examples: ['CUDAExecutionProvider', 'CPUExecutionProvider']
            logger: Logger instance

        Learning Point - Execution Providers:
            ONNX Runtime can use different "providers" for acceleration:
            - CoreMLExecutionProvider: Apple Silicon GPU (M1/M2/M3)
            - CUDAExecutionProvider: NVIDIA GPU
            - CPUExecutionProvider: Standard CPU (always available)

            If a provider isn't available, ONNX falls back to the next one.
        """
        super().__init__(logger)

        # Auto-detect providers if not specified
        if providers is None:
            providers = self._get_default_providers()

        self._providers = providers
        self._session = None
        self._input_names = None
        self._output_names = None
        self._input_shape = None

        self.log(logging.INFO, f"ONNX Backend initialized with providers: {providers}")

    def _get_default_providers(self) -> List[str]:
        """
        Get platform-appropriate default providers.

        Returns:
            List of provider names in priority order

        Learning Point:
            Different platforms have different optimal providers:
            - macOS: CoreML > CPU
            - Linux/Windows with NVIDIA: CUDA > CPU
            - Others: CPU only
        """
        import platform

        providers = []

        # Check available providers
        available = ort.get_available_providers()

        # Platform-specific defaults
        system = platform.system()

        if system == 'Darwin':  # macOS
            if 'CoreMLExecutionProvider' in available:
                providers.append('CoreMLExecutionProvider')
        else:
            if 'CUDAExecutionProvider' in available:
                providers.append('CUDAExecutionProvider')
            if 'OpenVINOExecutionProvider' in available:
                providers.append('OpenVINOExecutionProvider')
            if 'DmlExecutionProvider' in available:  # DirectML (Windows)
                providers.append('DmlExecutionProvider')

        # CPU is always available as fallback
        providers.append('CPUExecutionProvider')

        return providers

    def load_model(self, model_spec: Union[str, Dict[str, Any]]) -> None:
        """
        Load an ONNX model.

        Args:
            model_spec: Path to .onnx file (string)
                       Example: 'models/yolov7-tiny_640x640.onnx'

        Raises:
            FileNotFoundError: If model file doesn't exist
            RuntimeError: If model loading fails

        Learning Point:
            InferenceSession is the main ONNX Runtime class:
            - Loads the model
            - Optimizes it for the target hardware
            - Provides inference method
        """
        if isinstance(model_spec, dict):
            model_path = model_spec.get('path')
        else:
            model_path = model_spec

        if not Path(model_path).exists():
            raise FileNotFoundError(f"ONNX model not found: {model_path}")

        try:
            self.log(logging.INFO, f"Loading ONNX model from: {model_path}")

            # Create ONNX Runtime session
            # Learning Point: InferenceSession automatically:
            # - Optimizes the model graph
            # - Selects best execution provider
            # - Prepares for inference
            self._session = ort.InferenceSession(
                model_path,
                providers=self._providers
            )

            # Extract model metadata
            # This is useful for preprocessing
            model_inputs = self._session.get_inputs()
            self._input_names = [inp.name for inp in model_inputs]
            self._input_shape = model_inputs[0].shape  # e.g., [1, 3, 640, 640]

            model_outputs = self._session.get_outputs()
            self._output_names = [out.name for out in model_outputs]

            # Log what provider is actually being used
            used_providers = self._session.get_providers()
            self.log(logging.INFO, f"Model loaded with provider: {used_providers[0]}")
            self.log(logging.INFO, f"Input shape: {self._input_shape}")

        except Exception as e:
            self.log(logging.ERROR, f"Failed to load ONNX model: {e}")
            raise

    def inference(self, image_blob: np.ndarray) -> List[np.ndarray]:
        """
        Run inference on preprocessed image blob.

        Args:
            image_blob: Preprocessed image as numpy array
                       Expected shape: (1, 3, H, W)
                       Expected dtype: float32
                       Expected range: [0, 1]

        Returns:
            List of output arrays (model-dependent)

        Learning Point:
            ONNX Runtime inference is simpler than PyTorch:
            - No device management needed (handled internally)
            - No gradient tracking to disable
            - Just call session.run()

            But you must ensure:
            - Input is numpy array (not torch.Tensor)
            - Input shape matches model expectations
            - Input dtype is correct (usually float32)
        """
        if self._session is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        if not isinstance(image_blob, np.ndarray):
            raise TypeError(f"Expected numpy array, got {type(image_blob)}")

        # Run inference
        # Learning Point: session.run() takes:
        # - output_names: Which outputs to return (None = all)
        # - input_feed: Dict mapping input names to arrays
        outputs = self._session.run(
            self._output_names,
            {self._input_names[0]: image_blob}
        )

        return outputs

    def get_device(self) -> str:
        """
        Get the execution provider being used.

        Returns:
            Provider name (e.g., 'CoreMLExecutionProvider', 'CPUExecutionProvider')

        Note:
            ONNX Runtime handles device management internally.
            This returns the provider name, not a device like 'cuda' or 'cpu'.
        """
        if self._session:
            providers = self._session.get_providers()
            return providers[0] if providers else 'unknown'
        return 'not_loaded'

    def get_input_shape(self) -> Optional[tuple]:
        """
        Get expected input shape.

        Returns:
            Input shape tuple (e.g., (1, 3, 640, 640))
        """
        return self._input_shape

    def get_model_info(self) -> Dict[str, Any]:
        """
        Get model information.

        Returns:
            Dictionary with model details
        """
        info = super().get_model_info()
        info['type'] = 'onnx'
        info['framework'] = 'onnxruntime'

        if self._session:
            info['provider'] = self.get_device()
            info['input_names'] = self._input_names
            info['output_names'] = self._output_names
            info['available_providers'] = ort.get_available_providers()

        return info
