from abc import ABC, abstractmethod
from typing import List, Any
import torch
import numpy as np
from .detection import Detection

class Detector(ABC):
    """Base PyTorch detector with standard pipeline."""
    
    def __init__(self, model, device='cpu', conf_threshold=0.5):
        self.model = model.to(device)
        self.model.eval()
        self.device = device
        self.conf_threshold = conf_threshold
    
    @abstractmethod
    def preprocess(self, image: np.ndarray) -> torch.Tensor:
        """Prepare image for model."""
        pass
    
    @abstractmethod
    def inference(self, input_tensor: torch.Tensor) -> Any:
        """Run model forward pass."""
        pass
    
    @abstractmethod
    def postprocess(self, output: Any, image_shape: tuple) -> List[Detection]:
        """Filter, NMS, convert to Detection objects."""
        pass
    
    def detect(self, image: np.ndarray) -> List[Detection]:
        input_tensor = self.preprocess(image)

        with torch.no_grad():
            output = self.inference(input_tensor)
            
        detections = self.postprocess(output, image.shape[:2])
        return detections