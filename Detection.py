# IMPORT IDENTITY-PRESERVING PROPERTIES
from properties.BoundingBox import BoundingBox

import cv2
import random

# TODO: ADD APPEARANCE
# TODO: ADD KEYPOINTS

class Detection:

    __slots__ = ('_class_id', '_bounding_box', '_confidence_score')

    # CLASS CONSTANTS
    _LINE_THICKNESS = 2 # Default line thickness for bounding box display
    
    # TODO: UPDATE CONSTRUCTOR WITH MORE PROPERTIES
    #def __init__(self, classId=0, boundingBox=BoundingBox(), keypoints=KeyPoints(), appearance=Appearance(), confidenceScore=0):
    def __init__(self, class_id=0, bounding_box=BoundingBox(), confidence_score=0):

        # NIPP
        self._class_id = class_id
        self._confidence_score = confidence_score
    
        # IPP
        self._bounding_box = bounding_box
    
    # PROPERTIES
    @property
    def class_id(self):
        return self._class_id
    
    @property
    def bounding_box(self):
        return self._bounding_box

    @property
    def confidence_score(self):
        return self._confidence_score
    
    # SIMILARITY
    # TODO: UPDATE THIS METHOD BY ADDING MORE PROPERTIES
    def calculate_similarity(self, other_bounding_box):
        similarity_iou = self._bounding_box.iou(other_bounding_box)
        return similarity_iou
    
    # OUTPUT
    # RANDOM COLOUR GENERATOR BASED ON ID TO UNIQUELY IDENTIFY TRACKS
    def random_colour(self, seed):
        random.seed(seed)  
        return (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
    
    # DISPLAY DETECTION 
    def display(self, frame, label="", colour=None, seed=0):
        
        # IF COLOUR IS NOT PROVIDED, GENERATE A RANDOM COLOUR
        if colour is None:
            colour = self.random_colour(seed)

        x_min, y_min, x_max, y_max = map(int, self._bounding_box.xyxy())  # BOUNDING BOX CORNERS
        label += f" - {self._confidence_score:.2f}" # ADD CONFIDENCE SCORE TO LABEL
        cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), colour, self._LINE_THICKNESS)
        cv2.putText(frame, label, (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, colour, self._LINE_THICKNESS)
    
    def __repr__(self, format='corners'):
        return (f"DET({self._bounding_box.__repr__(format=format)}, "
                f"SCORE={self._confidence_score:.2f})")
    # TODO: REMOVE THIS
    # def __repr__(self):
    #     return (f"Detection(ClassID={self._class_id}, "
    #             f"BoundingBox={self._bounding_box.__repr__(format='center')}, "
    #             f"ConfidenceScore={self._confidence_score:.2f})")
    