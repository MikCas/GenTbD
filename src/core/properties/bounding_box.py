import torch
from torchvision.ops import box_iou
import cv2

class BoundingBox:
    """Immutable bounding box with IoU calculation and rendering.

    Stores coordinates in (x1, y1, x2, y2) format where (x1, y1) is the top-left
    corner and (x2, y2) is the bottom-right corner.

    Args:
        x1: Left x coordinate
        y1: Top y coordinate
        x2: Right x coordinate
        y2: Bottom y coordinate

    Usage:
        bbox = BoundingBox(10, 20, 100, 200)
        x1, y1, x2, y2 = bbox.xyxy
        iou_score = bbox.iou(other_bbox)
        bbox.draw(image, color=(0, 255, 0))
    """

    def __init__(self, x1, y1, x2, y2):
        self._tensor = torch.tensor([[x1, y1, x2, y2]], dtype=torch.float32)

    @property
    def xyxy(self):
        return tuple(self._tensor[0].tolist())

    def iou(self, other):
        return float(box_iou(self._tensor, other._tensor)[0, 0])

    def scale(self, factor):
        """Return new BoundingBox scaled by factor.

        Args:
            factor: Scale factor (e.g., 2.0 = double size, 0.5 = half size)

        Returns:
            New BoundingBox instance with scaled coordinates
        """
        x1, y1, x2, y2 = self.xyxy
        return BoundingBox(x1 * factor, y1 * factor, x2 * factor, y2 * factor)

    def draw(self, image, color=(0, 0, 255)):
        x1, y1, x2, y2 = map(int, self.xyxy)
        cv2.rectangle(image, (x1, y1), (x2,y2), color, 2)

    def __repr__(self):
        x1, y1, x2, y2 = self.xyxy
        return f"BoundingBox({x1:.1f}, {y1:.1f}, {x2:.1f}, {y2:.1f})"