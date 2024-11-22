import cv2

from properties.BoundingBox import BoundingBox
import random

class Detection:

    def __init__(self, classId=0, boundingBox=BoundingBox(), confidenceScore=0):
        self._classId = classId
        self._boundingBox = boundingBox
        self._confidenceScore = confidenceScore

    def getClassId(self):
        return self._classId
    
    def getBoundingBox(self):
        return self._boundingBox
    
    def getConfidenceScore(self):
        return self._confidenceScore
    
    # UPDATE THIS METHOD BY ADDING OTHER FEATURES TO THE COST SUCH AS KEYPOINTS AND APPEARANCE
    def calculateCost(self, otherBoundingBox):
        iou = self._boundingBox.iou(otherBoundingBox)
        return 1 - iou
    
    def generateColorFromId(self, id):
        """
        Generate a unique color based on the id.
        """
        random.seed(id)  # Use the ID as the seed for reproducibility
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
        # return f"Detection(classId={self._classId}, boundingBox={self._boundingBox}, confidenceScore={self._confidenceScore})"
        return f"[{self._classId}, {self._boundingBox}, {self._confidenceScore}]"
    
