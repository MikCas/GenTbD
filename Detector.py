from ultralytics import YOLO
from pprint import pprint

from properties.BoundingBox import BoundingBox
from Detection import Detection

class Detector:

    _HUMANCLASS = 0
    
    def __init__(self, modelName, confidenceThreshold=0.1,):
        self._modelPath = 'data/' + modelName + '.pt'
        self._model = YOLO(self._modelPath)
        self._imgsz = (0, 0)

        # PARAMETERS
        self._confidenceThreshold = confidenceThreshold
        
    def setImgSize(self, imgsz):
        self._imgsz = imgsz
        return
    
    # Perform inference on frame and output detections
    def inference(self, frame): 

        print("---DETECTING")

        results = self._model.predict(frame, imgsz=self._imgsz, classes=[Detector._HUMANCLASS], conf=self._confidenceThreshold)

        detections = []

        # Loop through the results and extract bounding box info
        for result in results:

            boxes = result.boxes
            cls = boxes.cls.cpu().numpy()    # Class labels
            xyxy = boxes.xyxy.cpu().numpy()  # Bounding box coordinates
            conf = boxes.conf.cpu().numpy()  # Confidence scores

            # Combine xyxy, confidence, and class into one array per detection
            for i in range(len(xyxy)):
                bbox = BoundingBox.fromCorners(xyxy[i][0], xyxy[i][1], xyxy[i][2], xyxy[i][3])
                detection = Detection(cls[i], bbox, conf[i])
                detections.append(detection)

        return detections