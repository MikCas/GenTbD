import cv2
import random

# IMPORT IDENTITY-PRESERVING PROPERTIES
from properties.BoundingBox import BoundingBox
# TODO: ADD APPEARANCE
# TODO: ADD KEYPOINTS

class Detection:

    # TODO: UPDATE CONSTRUCTOR WITH MORE PROPERTIES
    #def __init__(self, classId=0, boundingBox=BoundingBox(), keypoints=KeyPoints(), appearance=Appearance(), confidenceScore=0):
    def __init__(self, classId=0, boundingBox=BoundingBox(), confidenceScore=0):

        #NIPP
        self._confidenceScore = confidenceScore
        self._classId = classId

        #IPP
        self._boundingBox = boundingBox
    
    def getConfidenceScore(self):
        return self._confidenceScore
    
    def getClassId(self):
        return self._classId
    
    def getBoundingBox(self):
        return self._boundingBox
    
    # TODO: UPDATE THIS METHOD BY ADDING MORE PROPERTIES
    def calculateSimilarity(self, otherBoundingBox):
        similarityIou = self._boundingBox.iou(otherBoundingBox)
        return similarityIou
    
    # Generate a color based on ID
    def generateColorFromId(self, id):
        random.seed(id)  
        color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
        return color
    
    def display(self, frame, id=0):
        color = self.generateColorFromId(id)
        lineThickness = 2

        # Get the bounding box in (xmin, ymin, xmax, ymax) format
        xmin, ymin, xmax, ymax = self._boundingBox.xyxy()
        xmin, ymin, xmax, ymax = int(xmin), int(ymin), int(xmax), int(ymax)
        
        # Debugging: Print the bounding box values
        # print(f"BBOX([{xmin}, {ymin}, {xmax}, {ymax}]),  SCORE({self._confidenceScore})")

        label = f"{id} - {self._confidenceScore:.2f}"
        cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), color, lineThickness)
        cv2.putText(frame, label, (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, lineThickness)
    
    def __repr__(self):
        # Using the BoundingBox __repr__ directly for bbox representation
        return f"Detection(ClassID={self._classId}, BoundingBox={repr(self._boundingBox)}, ConfidenceScore={self._confidenceScore:.2f})"

    def pretty_print(self):
        # More readable format for easier inspection
        return f"Class ID: {self._classId}\n{self._boundingBox}\nConfidence Score: {self._confidenceScore:.2f}"
