"""
PyTorch backend for model inference.

This module implements the Backend interface for PyTorch models.
Supports loading models from Ultralytics (YOLO) and other sources.

Learning Points:
1. torch.device - Managing where tensors/models live (CPU/GPU)
2. torch.no_grad() - Disabling gradient tracking for inference (faster, less memory)
3. Tensor conversion - numpy ↔ torch
4. Model loading - Different ways to load PyTorch models
"""

import torch
import numpy as np
from typing import Union, Dict, Any, Optional
import logging
from pathlib import Path

from .Backend import Backend


class PyTorchBackend(Backend):
    """
    PyTorch implementation of the Backend interface.

    Supports:
    - Ultralytics YOLO models (auto-download)
    - Local .pt files
    - torchvision models
    - HuggingFace models (future)

    Device Selection Priority:
    1. CUDA (NVIDIA GPU) - if available
    2. MPS (Apple Silicon GPU) - if available
    3. CPU - fallback

    Example:
        >>> backend = PyTorchBackend()
        >>> backend.load_model('yolov8n.pt')  # Auto-downloads if needed
        >>> outputs = backend.inference(image)
    """

    def __init__(self, device: Optional[str] = None, logger: Optional[logging.Logger] = None):
        """
        Initialize PyTorch backend.

        Args:
            device: Device to use ('cpu', 'cuda', 'mps', or None for auto-detect)
            logger: Logger instance

        Learning Point:
            device='cpu' - Always use CPU
            device='cuda' - Use NVIDIA GPU (errors if not available)
            device='mps' - Use Apple Silicon GPU (errors if not available)
            device=None - Auto-detect best device (recommended)
        """
        super().__init__(logger)

        # Auto-detect best device if not specified
        if device is None:
            device = self._detect_best_device()

        self._device = torch.device(device)
        self.log(logging.INFO, f"PyTorch Backend initialized on device: {self._device}")

    def _detect_best_device(self) -> str:
        """
        Detect the best available device.

        Priority: CUDA > MPS > CPU

        Returns:
            Device string ('cuda', 'mps', or 'cpu')

        Learning Point:
            - torch.cuda.is_available() - Check if NVIDIA GPU is available
            - torch.backends.mps.is_available() - Check if Apple Silicon GPU is available
            - Always have CPU as fallback
        """
        if torch.cuda.is_available():
            return 'cuda'
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            return 'mps'
        else:
            return 'cpu'

    def load_model(self, model_spec: Union[str, Dict[str, Any]]) -> None:
        """
        Load a PyTorch model.

        Args:
            model_spec: Either:
                - str: Path to .pt file or model name
                  Examples: 'models/yolov8n.pt', 'yolov8s.pt'
                - dict: {'type': 'ultralytics', 'model': 'yolov8n'}

        Learning Point:
            - Ultralytics YOLO class handles downloading automatically
            - model.to(device) moves model to GPU/CPU
            - model.eval() sets model to evaluation mode (disables dropout, etc.)
        """
        try:
            if isinstance(model_spec, str):
                # Load from file or model name
                self._load_from_path(model_spec)
            elif isinstance(model_spec, dict):
                # Load from dict specification
                model_type = model_spec.get('type', 'ultralytics')
                if model_type == 'ultralytics':
                    self._load_ultralytics(model_spec['model'])
                else:
                    raise ValueError(f"Unknown model type: {model_type}")
            else:
                raise ValueError(f"Invalid model_spec type: {type(model_spec)}")

            # Move model to device and set to evaluation mode
            # Learning Point: .eval() is important!
            # - Disables dropout layers
            # - Sets batch norm to use running stats (not batch stats)
            # - Required for consistent inference results
            self._model = self._model.to(self._device)
            self._model.eval()

            self.log(logging.INFO, f"Model loaded successfully on {self._device}")

        except Exception as e:
            self.log(logging.ERROR, f"Failed to load model: {e}")
            raise

    def _load_from_path(self, path: str) -> None:
        """
        Load model from file path.

        Args:
            path: Path to .pt file

        Learning Point:
            This uses Ultralytics YOLO for simplicity.
            For custom PyTorch models, you'd use:
                self._model = torch.load(path)
            or
                self._model = MyModel()
                self._model.load_state_dict(torch.load(path))
        """
        from ultralytics import YOLO

        # Check if file exists locally
        if Path(path).exists():
            self.log(logging.INFO, f"Loading model from: {path}")
            self._model = YOLO(path)
        else:
            # Try as model name (Ultralytics will download)
            self.log(logging.INFO, f"Model file not found locally. Attempting to download: {path}")
            self._model = YOLO(path)

    def _load_ultralytics(self, model_name: str) -> None:
        """
        Load Ultralytics YOLO model.

        Args:
            model_name: Model name (e.g., 'yolov8n', 'yolov8s', 'yolov8m')

        Learning Point:
            Ultralytics automatically downloads models on first use!
            Models are cached in: ~/.ultralytics/
        """
        from ultralytics import YOLO

        self.log(logging.INFO, f"Loading Ultralytics model: {model_name}")
        self._model = YOLO(model_name)

    def inference(self, image: Union[np.ndarray, torch.Tensor]) -> Any:
        """
        Run inference on an image.

        Args:
            image: Input image as numpy array or torch tensor
                   Expected format: (H, W, C) for numpy, (C, H, W) for torch

        Returns:
            Model outputs (depends on model type)

        Learning Point - torch.no_grad():
            Why we use this context manager:
            1. Disables gradient calculation (we're not training!)
            2. Reduces memory usage significantly
            3. Makes inference faster

            Without no_grad():
                - PyTorch tracks all operations for backprop
                - Stores intermediate values
                - Uses ~2x more memory

            With no_grad():
                - No tracking overhead
                - Much faster
                - Less memory

        Example:
            >>> with torch.no_grad():
            >>>     outputs = model(image)  # Fast, low memory
        """
        if self._model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        # Learning Point: torch.no_grad() is ESSENTIAL for inference
        with torch.no_grad():
            # Convert numpy to torch if needed
            # Learning Point: Ultralytics handles this internally,
            # but for custom models you'd do:
            # if isinstance(image, np.ndarray):
            #     image = torch.from_numpy(image).to(self._device)

            # Run inference
            # Note: Ultralytics YOLO returns Results object, not raw tensors
            outputs = self._model(image, device=self._device)

            return outputs

    def get_device(self) -> str:
        """
        Get current device.

        Returns:
            Device string ('cpu', 'cuda', 'mps')
        """
        return str(self._device)

    def get_input_shape(self) -> Optional[tuple]:
        """
        Get expected input shape.

        Returns:
            Input shape tuple or None

        Note:
            For Ultralytics models, input size is typically (640, 640)
            but can be dynamic
        """
        # Ultralytics models are flexible with input size
        # but typically use 640x640
        return (1, 3, 640, 640)

    def get_model_info(self) -> Dict[str, Any]:
        """
        Get model information.

        Returns:
            Dictionary with model details
        """
        info = super().get_model_info()
        info['type'] = 'pytorch'
        info['framework'] = 'ultralytics'

        # Add PyTorch-specific info
        if self._model is not None:
            # Check if it's an Ultralytics model
            if hasattr(self._model, 'names'):
                info['classes'] = self._model.names  # Class names
            if hasattr(self._model, 'model'):
                # Count parameters
                total_params = sum(p.numel() for p in self._model.model.parameters())
                info['num_parameters'] = f"{total_params / 1e6:.1f}M"

        return info

    def to_numpy(self, tensor: torch.Tensor) -> np.ndarray:
        """
        Convert PyTorch tensor to numpy array.

        Args:
            tensor: PyTorch tensor

        Returns:
            Numpy array

        Learning Point:
            - tensor.cpu() moves tensor to CPU first (required before .numpy())
            - .numpy() converts to numpy array (zero-copy if on CPU)
            - .detach() removes from computation graph (if it was part of one)
        """
        return tensor.cpu().detach().numpy()

    def to_tensor(self, array: np.ndarray) -> torch.Tensor:
        """
        Convert numpy array to PyTorch tensor.

        Args:
            array: Numpy array

        Returns:
            PyTorch tensor on the backend's device

        Learning Point:
            - torch.from_numpy() creates tensor (shares memory with numpy!)
            - .to(device) moves to GPU if needed (creates copy)
        """
        tensor = torch.from_numpy(array)
        return tensor.to(self._device)
