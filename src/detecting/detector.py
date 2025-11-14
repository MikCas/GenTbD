from abc import ABC, abstractmethod
from typing import List, Any
import torch
import numpy as np
import cv2
from .detection import Detection

class Detector(ABC):
    """
    Base PyTorch detector with standard pipeline.

    The detector follows a three-stage pipeline:
    1. preprocess: Convert image to tensor
    2. inference: Run model forward pass
    3. postprocess: Convert outputs to Detection objects

    Subclasses must implement these three methods. Default implementations
    are provided for common PyTorch/torchvision preprocessing and inference
    patterns, which can be used by calling super() or overridden entirely.
    """

    def __init__(self, model, device='cpu', conf_threshold=0.5):
        self.model = model.to(device)
        self.model.eval()
        self.device = device
        self.conf_threshold = conf_threshold

    @abstractmethod
    def preprocess(self, image: np.ndarray) -> torch.Tensor:
        """
        Prepare image for model inference.

        Args:
            image: Input image in BGR format (H, W, 3)

        Returns:
            Preprocessed tensor ready for model
        """
        pass

    @abstractmethod
    def inference(self, input_tensor: torch.Tensor) -> Any:
        """
        Run model forward pass.

        Args:
            input_tensor: Preprocessed tensor

        Returns:
            Raw model output
        """
        pass

    @abstractmethod
    def postprocess(self, output: Any, image_shape: tuple) -> List[Detection]:
        """
        Convert model outputs to Detection objects.

        Args:
            output: Raw model output
            image_shape: Original image shape (H, W)

        Returns:
            List of Detection objects
        """
        pass

    def detect(self, image: np.ndarray) -> List[Detection]:
        """
        Run full detection pipeline.

        Args:
            image: Input image in BGR format (H, W, 3)

        Returns:
            List of Detection objects
        """
        input_tensor = self.preprocess(image)

        with torch.no_grad():
            output = self.inference(input_tensor)

        detections = self.postprocess(output, image.shape[:2])
        return detections

    # Default implementations for common patterns
    # Subclasses can use these by calling super() or override entirely

    def _default_preprocess_pytorch(self, image: np.ndarray) -> torch.Tensor:
        """
        Default preprocessing for PyTorch/torchvision models.

        Converts BGR → RGB, normalizes to [0, 1], and converts to CHW format.

        Optimizations applied:
        - Use numpy view for BGR→RGB (faster than cv2.cvtColor)
        - Ensure contiguous memory for efficient GPU transfer
        - Single normalization operation

        Args:
            image: Input image in BGR format (H, W, 3)

        Returns:
            Preprocessed tensor (C, H, W)

        Example:
            >>> def preprocess(self, image):
            >>>     return self._default_preprocess_pytorch(image)
        """
        # OPTIMIZATION: Use numpy view for BGR→RGB (faster than cv2.cvtColor)
        # This creates a view instead of a copy, reducing allocations
        # The [:, :, ::-1] reverses the channel dimension (BGR → RGB)
        image_rgb = image[:, :, ::-1]

        # Convert to contiguous array (required for efficient GPU transfer)
        # Then convert to tensor and normalize in one operation
        tensor = torch.from_numpy(np.ascontiguousarray(image_rgb)).float() / 255.0

        # HWC to CHW
        tensor = tensor.permute(2, 0, 1)

        return tensor.contiguous()  # Ensure contiguous for GPU transfer

    def _default_inference_pytorch(self, input_tensor: torch.Tensor) -> Any:
        """
        Default inference for PyTorch models expecting (N, C, H, W) input.

        Adds batch dimension, moves to device, runs model, returns first output.

        Args:
            input_tensor: Preprocessed tensor (C, H, W)

        Returns:
            First model output (typically a dict or list)

        Example:
            >>> def inference(self, input_tensor):
            >>>     return self._default_inference_pytorch(input_tensor)
        """
        # Add batch dimension
        input_batch = input_tensor.unsqueeze(0).to(self.device)

        # Run model
        outputs = self.model(input_batch)

        # Return first output (most models return list/dict)
        return outputs[0]