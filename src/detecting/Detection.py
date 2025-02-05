# TODO: ADD MORE PROPERTIES - APPEARANCE, KEYPOINTS 
# HOW TO ADD MORE PROPERTIES:
# 1. IF THE PROPERTY HAS A COMPLEX TYPE STRUCTURE AND IS IPP, THEN CREATE CLASS FOR IT IN PROPERTIES (EX. BOUNDING_BOX)
# 2. UPDATE THE DETECTION CLASS __SLOTS__, CONSTRUCTOR, GETTER
# 3. ADD PROPERTY SPECIFIC SIMILARITY METRIC IN CALCULATE SIMILARITY
# 4. ADD PROPERTY SPECIFIC DISPLAY FUNCTION

# IMPORT IDENTITY-PRESERVING PROPERTIES
from properties.BoundingBox import BoundingBox

import cv2
import random

class Detection:

    __slots__ = ('_class_id', '_bounding_box', '_confidence_score')
    
    def __init__(self, class_id=0, bounding_box=BoundingBox(), confidence_score=0):
        # IPP
        self._bounding_box = bounding_box

        # NIPP
        self._class_id = class_id
        self._confidence_score = confidence_score
    
    # PROPERTIES
    @property
    def bounding_box(self): return self._bounding_box
    @property
    def class_id(self): return self._class_id
    @property
    def confidence_score(self): return self._confidence_score
    
    # TODO: CREATE A SIMILARITY FUNCTION FOR EACH PROPERTY?, THEN IN THE COST FUNCTION WE CAN ADD THEM IN DIFFERENT WAYS
    # SIMILARITY
    def calculate_bounding_box_similarity(self, other_bounding_box):
        similarity_iou = self._bounding_box.iou(other_bounding_box)
        return similarity_iou

    # OUTPUT
    def random_colour(self, seed):
        random.seed(seed)  
        return (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
    
    def display(self, frame, label="", colour=None, seed=0):
        # IF COLOUR IS NOT PROVIDED, GENERATE A RANDOM COLOUR
        if colour is None:
            colour = self.random_colour(seed)

        x_min, y_min, x_max, y_max = map(int, self._bounding_box.xyxy())                      # BOUNDING BOX CORNERS
        label += f" - {self._confidence_score:.2f}"                                           # ADD CONFIDENCE SCORE TO LABEL
        cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), colour, 2)                       # GENERATE BOUNDING BOX
        cv2.putText(frame, label, (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, colour, 2)

    def __repr__(self, format='corners'):
        return (f"DET({self._bounding_box.__repr__(format=format)}, "
                f"SCORE={self._confidence_score:.2f})")