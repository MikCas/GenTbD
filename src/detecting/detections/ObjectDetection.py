from .Detection import Detection
from properties import BoundingBox

from typing import Optional
import cv2

class ObjectDetection(Detection):
    """
    Detection subclass representing detected object in an image or video frame obtained from an object detector

    Attributes:
        IPP (Identity-Preserving Properties):
            - bounding_box (BoundingBox): The bounding box of the detected object.
        NIPP (Non-Identity-Preserving Properties):
            - class_id (int): The ID of the detected class.
            - confidence_score (float): The confidence score of the detection.
    """

    ##### PROPERTIES #####
    __slots__ = ('_class_id', '_bounding_box', '_confidence_score')
    @property
    def class_id(self) -> int:
        return self._class_id
    @property
    def bounding_box(self) -> BoundingBox:
        return self._bounding_box
    @property
    def confidence_score(self) -> float:
        return self._confidence_score

    ##### SETUP #####
    def __init__(self, 
                 class_id: int = 0, 
                 bounding_box: BoundingBox = BoundingBox(), 
                 confidence_score: float = 0) -> None:
        """
        Args:
            class_id (int): The ID of the detected class.
            bounding_box (BoundingBox): The bounding box of the detected object.
            confidence_score (float): The confidence score of the detection.
        """
        self._class_id = class_id
        self._bounding_box = bounding_box
        self._confidence_score = confidence_score

    ##### FUNCTIONS #####
    def calculate_similarity(self, other: 'Detection') -> float:
        """
        Calculate the similarity between two bounding boxes using the similarity methods of the IPPs

        Args:
            other (Detection): The detection to compare with.
        Returns:
            float: The similarity score between the two bounding boxes.
        """
        similarity_iou = self._bounding_box.similarity(other.bounding_box)
        return similarity_iou

    ##### DISPLAY #####
    def draw(self, 
                image: cv2.Mat, 
                label: str = "", 
                colour:Optional[tuple] = None, 
                seed:Optional[int] = None) -> None:
    
        """
        Display the detection on the given frame.
        Args:
            image (cv2.Mat): The image to draw on.
            label (str, optional): The label text. Defaults to "".
            colour (tuple, optional): The colour of the bounding box and label. Defaults to None.
            seed (int, optional): Seed for random colour generation. Defaults to None.
        """

        # Generate random colour if not provided
        if colour is None:
            colour = self.generate_random_colour(seed)

        # Extract bounding box coordinates
        x_min, y_min, x_max, y_max = map(int, self._bounding_box.xyxy())

        # Add confidence score to the label
        label_with_score = f"{label} - {self._confidence_score:.2f}"
        
        # Draw the bounding box and label
        self.draw_bounding_box(image, x_min, y_min, x_max, y_max, colour)
        self.draw_label(image, label_with_score, x_min, y_min, colour)

    def __str__(self, format='corners') -> str:
        """
        Args:
            format (str): Format for the bounding box representation. Defaults to 'corners'.
        """
        return (f"DET(BB={self._bounding_box.__str__(format=format)}, "
                f"CONF={self._confidence_score:.2f})")
    