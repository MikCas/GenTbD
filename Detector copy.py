from ultralytics import YOLO

# WRAPPER CLASS WHICH DEPENDS ON THE DETECTOR USED (MAYBE CREATE AS AN ABSTRACT CLASS)
# UPDATE BASED ON THE PROPERTIES/PARAMETERS OF A DETECTOR
# 1. MODEL NAME, MODEL PATH, IMG SIZE, CONFIDENCE THRESHOLD
# 2. ANY OTHER PARAMETERS
# 3. INFERENCE FUNCTION

from properties.BoundingBox import BoundingBox
from Detection import Detection

class Detector:

    # CLASS CONSTANTS
    _HUMAN_CLASS = 0
    
    def __init__(self, model_name, confidence_threshold=0.1, logger=None):

        self._logger = logger if logger else None

        self._model_path = f"data/{model_name}.pt"
        self._model = YOLO(self._model_path)
        self._confidence_threshold = confidence_threshold
        self._imgsz = (0, 0)

        if self.logger: self.logger.info(f"########DETECTOR INITIALISED - MODEL {self._model_path}")
    
    # GETTERS/SETTERS
    @property
    def logger(self): return self._logger
    @property
    def confidence_threshold(self): return self._confidence_threshold
    @property
    def imgsz(self): return self._imgsz
    @imgsz.setter
    def imgsz(self, imgsz):
        self._imgsz = imgsz
        return
    
    # DETECTION
    def inference(self, frame):
        if self.logger: self.logger.info("########DETECTING")

        # INFERENCE
        try:
            results = self._model.predict(frame, imgsz=self.imgsz, classes=[Detector._HUMAN_CLASS], conf=self.confidence_threshold)
        except Exception as e:
            if self.logger: self.logger.error(f"########ERROR DURING INFERENCE: {e}")
            return []
        
        # GET DETECTIONS FROM RESULTS
        results = results[0]
        boxes = results.boxes
        if len(boxes) == 0:
            return []
            
        # GET DETECTION DATA FROM RESULTS
        cls_array = boxes.cls.cpu().numpy()
        xyxy_array = boxes.xyxy.cpu().numpy() 
        conf_array = boxes.conf.cpu().numpy()
        
        # CREATE DETECTIONS FROM DATA
        detections = [
            Detection(
                cls_array[i],
                BoundingBox.from_corners(*xyxy_array[i]),
                conf_array[i]
            )
            for i in range(len(boxes))
        ]

        if self.logger: self.logger.info(f"########DETECTIONS: {len(detections)}")
        return detections
    
    def display_detections(self, detections, frame):
        for detection in detections:
            detection.display(frame, colour=(255, 255, 255))
            if self.logger: self.logger.info(f"############DETECTED: {detection}")