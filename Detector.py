import logging
from ultralytics import YOLO

from properties.BoundingBox import BoundingBox
from Detection import Detection

class Detector:

    # CLASS CONSTANTS
    _HUMAN_CLASS = 0
    _CONFIDENCE_THRESHOLD = 0.1
    
    def __init__(self, model_name, logger=None):

        # INITIALISE LOGGER, OTHERWISE USE DEFAULT LOGGER
        self._logger = logger if logger else logging.getLogger(__name__)

        self._model_path = f"data/{model_name}.pt"
        self._model = YOLO(self._model_path)
        self._imgsz = (0, 0)

        self._logger.info(f"----DETECTOR INITIALISED - MODEL {self._model_path}")
        
    def set_img_size(self, imgsz):
        self._imgsz = imgsz
        return
    
    # PERFORM DETECTION ON FRAME
    def inference(self, frame): 
        self._logger.info("----DETECTING")

        try:
            results = self._model.predict(frame, imgsz=self._imgsz, classes=[Detector._HUMAN_CLASS], conf=Detector._CONFIDENCE_THRESHOLD)
        except Exception as e:
            self._logger.error(f"----ERROR DURING INFERENCE: {e}")
            return []
        
        detections = []
        for result in results:
            boxes = result.boxes
            cls, xyxy, conf = boxes.cls.cpu().numpy(), boxes.xyxy.cpu().numpy(), boxes.conf.cpu().numpy()
            detections.extend(self.process_detections(cls, xyxy, conf))

        return detections
    
    # CREATES DETECTION OBJECTS FROM INFERENCE RESULTS
    def process_detections(self, cls, xyxy, conf):
        detections = []
        for i in range(len(xyxy)):
            bbox = BoundingBox.from_corners(*xyxy[i])
            detection = Detection(cls[i], bbox, conf[i])
            detections.append(detection)
        return detections