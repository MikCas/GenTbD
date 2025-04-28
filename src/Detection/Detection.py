from Properties.BoundingBox import BoundingBox
from typing import Optional
import cv2
import random

class Detection:
    """
    A class representing a detected object in an image or video frame.

    Attributes:
        Consists of Identity-Preserving Properties (IPP), which are properties that remain constant across frames for the same identity:
            - bounding_box (BoundingBox): The bounding box of the detected object.
        and Non-Identity-Preserving Properties (NIPP), which are properties that do not depend on time:
            - class_id (int): The ID of the detected class.
            - confidence_score (float): The confidence score of the detection.
    """

    ### ATTRIBUTES
    __slots__ = ('_class_id', '_bounding_box', '_confidence_score')
    @property
    def bounding_box(self): return self._bounding_box
    @property
    def class_id(self): return self._class_id
    @property
    def confidence_score(self): return self._confidence_score
    
    ### SETUP 
    def __init__(self, class_id: int = 0, bounding_box: BoundingBox = BoundingBox(), confidence_score: float = 0) -> None:
        self._bounding_box = bounding_box
        self._class_id = class_id
        self._confidence_score = confidence_score
    
    ### FUNCTIONS
    def calculate_similarity(self, other_bounding_box: BoundingBox) -> float:
        """
        Calculate the similarity between two bounding boxes using the similarity methods of the IPPs
        Note: you can combine similarity metrics any way you want

        Args:
            other_bounding_box (BoundingBox): The bounding box to compare with.
        Returns:
            float: The similarity score between the two bounding boxes.
        """
        similarity_iou = self._bounding_box.similarity(other_bounding_box)
        return similarity_iou

    ### REPRESENTATION
    @staticmethod
    def generate_random_colour(seed: Optional[int] = 0) -> tuple:
        """
        Generate a random colour given a seed 
        Args:
            seed (int, optional): Seed fro random number generation
        Returns:
            tuple: A tuple representing the RGB colour.
        """
        if seed is not None:
            random.seed(seed)
        return (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
    
    @staticmethod
    def draw_bounding_box(image: cv2.Mat, 
                          x_min: int, 
                          y_min: int, 
                          x_max: int, 
                          y_max: int, 
                          colour: tuple) -> None:
        """
        Draw the bounding box on the image.

        Args:
            image (cv2.Mat): The image to draw on.
            x_min (int): Minimum x-coordinate.
            y_min (int): Minimum y-coordinate.
            x_max (int): Maximum x-coordinate.
            y_max (int): Maximum y-coordinate.
            colour (tuple): The colour of the bounding box.
        """
        cv2.rectangle(image, (x_min, y_min), (x_max, y_max), colour, 2)

    @staticmethod
    def draw_label(image: cv2.Mat, 
                   label: str, 
                   x_min: int, 
                   y_min: int, 
                   colour: tuple) -> None:
        """
        Draw the label above the bounding box.

        Args:
            image (cv2.Mat): The image to draw on.
            label (str): The label text.
            x_min (int): Minimum x-coordinate of the bounding box.
            y_min (int): Minimum y-coordinate of the bounding box.
            colour (tuple): The colour of the label text.
        """
        cv2.putText(image, label, (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, colour, 2)
        
    def display(self, 
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

    def __repr__(self, format='corners'):
        return (f"DET({self._bounding_box.__repr__(format=format)}, "
                f"SCORE={self._confidence_score:.2f})")