# from properties.BoundingBox import BoundingBox
from properties import BoundingBox
from abc import ABC, abstractmethod
from typing import Optional
import cv2
import random

class Detection(ABC):
    """
    Abstract class for a detection.
    Provides a template for derived classes to implement specific detection types, bsaed on different
    detection algorithms or models
    
    Attributes:
        Consists of Identity-Preserving Properties (IPP), which are properties that remain constant across frames for the same identity:
        and Non-Identity-Preserving Properties (NIPP), which are properties that do not depend on time:
    """
    
    ##### FUNCTIONS #####
    @abstractmethod
    def calculate_similarity(self, other: 'Detection') -> float:
        """
        Calculate similarity between this detection and another detection.
        Args:
            other_bounding_box (BoundingBox): The bounding box to compare with.
        """
        pass

    ##### DISPLAY #####
    @staticmethod
    def generate_random_colour(seed: Optional[int] = 0) -> tuple:
        """
        Generate a random colour given a seed 
        Args:
            seed (int, optional): Seed for random number generation
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

    ##### ABSTRACT METHODS #####
    @abstractmethod
    def draw(self, 
             image: cv2.Mat, 
             label: str = "", 
             colour: Optional[tuple] = None, 
             seed: Optional[int] = None) -> None:
        """
        Display the detection on the given frame.
        Args:
            image (cv2.Mat): The image to draw on.
            label (str, optional): The label text. Defaults to "".
            colour (tuple, optional): The colour of the bounding box and label. Defaults to None.
            seed (int, optional): Seed for random colour generation. Defaults to None.
        """
        pass

    @abstractmethod
    def __str__(self, format: str = 'corners') -> str:
        pass