import onnxruntime as ort 
import numpy as np
import cv2
import time

from properties.BoundingBox import BoundingBox
from Detection import Detection

def nms(boxes, scores, iou_threshold):
        # Sort by score
        sorted_indices = np.argsort(scores)[::-1]

        keep_boxes = []
        while sorted_indices.size > 0:
            # Pick the last box
            box_id = sorted_indices[0]
            keep_boxes.append(box_id)

            # Compute IoU of the picked box with the rest
            ious = compute_iou(boxes[box_id, :], boxes[sorted_indices[1:], :])

            # Remove boxes with IoU over the threshold
            keep_indices = np.where(ious < iou_threshold)[0]

            # print(keep_indices.shape, sorted_indices.shape)
            sorted_indices = sorted_indices[keep_indices + 1]

        return keep_boxes
    
def compute_iou(box, boxes):
    # Compute xmin, ymin, xmax, ymax for both boxes
    xmin = np.maximum(box[0], boxes[:, 0])
    ymin = np.maximum(box[1], boxes[:, 1])
    xmax = np.minimum(box[2], boxes[:, 2])
    ymax = np.minimum(box[3], boxes[:, 3])

    # Compute intersection area
    intersection_area = np.maximum(0, xmax - xmin) * np.maximum(0, ymax - ymin)

    # Compute union area
    box_area = (box[2] - box[0]) * (box[3] - box[1])
    boxes_area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    union_area = box_area + boxes_area - intersection_area

    # Compute IoU
    iou = intersection_area / union_area

    return iou

class Detector:
    
    # CONSTRUCTOR
    def __init__(self, model_name="yolov7-tiny_640x640.onnx", confidence_threshold=0.1, iou_threshold=0.5, classes=[0], logger=None):

        self._logger = logger
        self._model_path = f"models/{model_name}"
        self._confidence_threshold = confidence_threshold
        self._iou_threshold = iou_threshold
        self._classes = classes

        # INIT ONNX RUNTIME SESSION
        self._session = ort.InferenceSession(self._model_path, providers=['CoreMLExecutionProvider', 'CPUExecutionProvider'])

        # INPUT DETAILS
        model_inputs = self._session.get_inputs()
        self._input_names = [input.name for input in model_inputs]     # NAMES

        self._input_shape = model_inputs[0].shape                      # SHAPE (1, 3, 640, 640)
        self._input_height = self._input_shape[2]                      ### HEIGHT
        self._input_width = self._input_shape[3]                       ### WIDTH

        # OUTPUT DETAILS
        model_outputs = self._session.get_outputs()
        self._output_names = [output.name for output in model_outputs] # NAMES

        if self.logger: self.logger.info(f"########DETECTOR INITIALISED - MODEL {self._model_path}")
    
    # GETTERS/SETTERS
    @property
    def logger(self): return self._logger

    # DETECT OBJECTS
    def detect(self, image):
        blob = self.preprocess(image)                           # PREPROCESS
        outputs = self.inference(blob)                          # INFERENCE
        boxes, class_ids, scores = self.postprocess(outputs)    # POSTPROCESS

        detections = []
        for box, class_id, confidence_score in zip(boxes, class_ids, scores):
            detections.append(
                Detection(
                    class_id,
                    BoundingBox.from_corners(*box),
                    confidence_score
            ))
        return detections

    ##### PREPROCESS #####
    def preprocess(self, image):

        input_height, input_width = self._input_shape[2:] # INPUT SHAPE
        image_height, image_width = image.shape[:2]       # IMAGE SHAPE
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)    # CONVERT TO RGB

        # RESIZE IMAGE
        scale = min(input_height / image_height, input_width / image_width)
        resized_image = cv2.resize(image, (0,0), fx=scale, fy=scale)
        resized_height, resized_width = resized_image.shape[:2]

        # PAD IMAGE
        pad_top = (input_height - resized_height) // 2
        pad_bottom = input_height - resized_height - pad_top
        pad_left = (input_width - resized_width) // 2
        pad_right = input_width - resized_width - pad_left
        padded_image = cv2.copyMakeBorder(resized_image, 
                                    pad_top, pad_bottom,
                                    pad_left, pad_right,
                                    cv2.BORDER_CONSTANT,
                                    value=(114, 114, 114))

        # CREATE BLOB
        blob = padded_image.transpose(2,0,1)[np.newaxis,...].astype(np.float32)
        blob /= 255.0 # NORMALISE

        # PARAMETERS FOR POSTPROCESSING
        self._image_shape = (image_height, image_width)
        self._scale = scale
        self._padding = (pad_top, pad_bottom, pad_left, pad_right)

        return blob
    
    ##### INFERENCE #####
    def inference(self, blob):
        start = time.perf_counter()
        outputs = self._session.run(self._output_names, {self._input_names[0]: blob}) # OUTPUT FORMAT: [[batch, num_anchors, (cx, cy, w, h, conf, class_id)]]
        if self.logger:
            inference_time = (time.perf_counter() - start) * 1000
            self.logger.info(f"########Inference time: {inference_time:.2f} ms")
        return outputs

    ##### POSTPROCESS #####
    def postprocess(self, outputs):
        predictions = outputs[0][0] # [num_anchors, 5 + num_classes]
        
        # FIRST FILTER - BASED ON EXISTENCE OF OBJECT
        mask = predictions[:, 4] > self._confidence_threshold
        predictions = predictions[mask]
        if predictions.shape[0] == 0: return [], [], []

        # MULTIPLY CLASS CONFIDENCE WITH BOUNDING BOX CONFIDENCE
        confidence_scores = predictions[:, 4]
        predictions[:, 5:] *= confidence_scores[:, np.newaxis]

        # SECOND FILTER - BASED ON CLASS CONFIDENCE
        scores = np.max(predictions[:, 5:], axis=1)
        mask = scores > self._confidence_threshold
        predictions = predictions[mask]
        scores = scores[mask]
        if len(scores) == 0: return [], [], []

        # THURD FILTER - ONLY HUMAN CLASSES
        class_ids = np.argmax(predictions[:, 5:], axis=1).astype(np.int32)
        mask = np.isin(class_ids, self._classes)
        predictions = predictions[mask]
        scores = scores[mask]
        class_ids = class_ids[mask]
        
        # EXTRACT BOXES
        boxes = self.extract_boxes(predictions)
        indices = nms(boxes, scores, self._iou_threshold)
        return boxes[indices], class_ids[indices], scores[indices]
        
    def extract_boxes(self, boxes_xywh):
        image_height, image_width = self._image_shape
        scale = self._scale
        pad_top, pad_bottom, pad_left, pad_right = self._padding
        
        # REMOVE PADDING AND RESCALE TO ORIGINAL IMAGE SIZE
        x_center = (boxes_xywh[:, 0] - pad_left) / scale
        y_center = (boxes_xywh[:, 1] - pad_top) / scale
        width = boxes_xywh[:, 2] / scale
        height = boxes_xywh[:, 3] / scale

        # CALCULATE CORNERS
        x1 = x_center - width / 2
        y1 = y_center - height / 2
        x2 = x_center + width / 2
        y2 = y_center + height / 2

        # CLIP COORDINATES TO IMAGE BOUNDARIES
        x1 = np.clip(x1, 0, image_width)
        y1 = np.clip(y1, 0, image_height)
        x2 = np.clip(x2, 0, image_width)
        y2 = np.clip(y2, 0, image_height)

        return np.stack([x1, y1, x2, y2], axis=1)
    
    def display_detections(self, detections, frame):
        for detection in detections:
            detection.display(frame, colour=(255, 255, 255))
            if self.logger: self.logger.info(f"############DETECTED: {detection}")

  