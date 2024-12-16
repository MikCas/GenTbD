from ultralytics import YOLO
from pprint import pprint

from properties.BoundingBox import BoundingBox
from Detection import Detection

class Detector:

    HUMANCLASS = 0
    
    def __init__(self, modelName, confidenceThreshold=0.1,):
        self._modelPath = 'data/' + modelName + '.pt'
        self._model = YOLO(self._modelPath)

        # Detection properties
        self._confidenceThreshold = confidenceThreshold
        self._imgsz = (0, 0)

    def setImgSize(self, imgsz):
        self._imgsz = imgsz
        return
    
    def inference(self, frame): 

        results = self._model.predict(frame, imgsz=self._imgsz, classes=[Detector.HUMANCLASS], conf=self._confidenceThreshold)

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