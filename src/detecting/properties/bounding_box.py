import torch
from torchvision.ops import box_iou
import cv2

class BoundingBox:

    @property 
    def xyxy(self):
        return tuple(self._tensor[0].tolist())
    
    def __init__(self, x1, y1, x2, y2):
        self._tensor = torch.tensor([[x1, y1, x2, y2]], dtype=torch.float32)

    def iou(self, other):
        return float(box_iou(self._tensor, other._tensor)[0, 0])
    
    def draw(self, image, color=(0, 0, 255)):
        x1, y1, x2, y2 = map(int, self.xyxy)
        cv2.rectangle(image, (x1, y1), (x2,y2), color, 2)

    def __repr__(self):
        x1, y1, x2, y2 = self.xyxy
        return f"BoundingBox({x1:.1f}, {y1:.1f}, {x2:.1f}, {y2:.1f})"