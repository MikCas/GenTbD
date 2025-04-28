# from abc import ABC, abstractmethod
# from Detection.Detectors.AbstractDetector import AbstractDetector

# from Properties.BoundingBox import BoundingBox
# from Detection.Detection import Detection

# import cv2
# import onnxruntime as ort 
# import numpy as np
# import time

# def nms(boxes, scores, iou_threshold):
#     # Sort by score
#     sorted_indices = np.argsort(scores)[::-1]

#     keep_boxes = []
#     while sorted_indices.size > 0:
#         # Pick the last box
#         box_id = sorted_indices[0]
#         keep_boxes.append(box_id)

#         # Compute IoU of the picked box with the rest
#         ious = compute_iou(boxes[box_id, :], boxes[sorted_indices[1:], :])

#         # Remove boxes with IoU over the threshold
#         keep_indices = np.where(ious < iou_threshold)[0]

#         # print(keep_indices.shape, sorted_indices.shape)
#         sorted_indices = sorted_indices[keep_indices + 1]

#     return keep_boxes
    
# def compute_iou(box, boxes):
#     # Compute xmin, ymin, xmax, ymax for both boxes
#     xmin = np.maximum(box[0], boxes[:, 0])
#     ymin = np.maximum(box[1], boxes[:, 1])
#     xmax = np.minimum(box[2], boxes[:, 2])
#     ymax = np.minimum(box[3], boxes[:, 3])

#     # Compute intersection area
#     intersection_area = np.maximum(0, xmax - xmin) * np.maximum(0, ymax - ymin)

#     # Compute union area
#     box_area = (box[2] - box[0]) * (box[3] - box[1])
#     boxes_area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
#     union_area = box_area + boxes_area - intersection_area

#     # Compute IoU
#     iou = intersection_area / union_area

#     return iou

# class YOLOv7ONNX(AbstractDetector):
#     """
#     YOLOv7 model in ONNX format.
    
#     Attributes:
#         iou_threshold (float): IoU threshold for non-maximum suppression
#         classes (list): List of class IDs to detect
#         image_shape (tuple): Original image dimensions (height, width) [preprocess()
#         scale (float): Scale factor for resizing [preprocess()]
#         padding (tuple): Padding values (top, bottom, left, right) [preprocess()]
#     """

#     ### ATTRIBUTES
#     @property
#     def logger(self): return self._logger

#     ### SETUP
#     def __init__(self, 
#                 *args,
#                 iou_threshold: float = 0.5, 
#                 classes: list = [0], 
#                 **kwargs):
#         self._iou_threshold = iou_threshold
#         self._classes = classes
#         super().__init__(*args, **kwargs)
    
#     ### FUNCTIONS
#     # def extract_boxes(self, boxes_xywh: np.ndarray) -> np.ndarray:
#     #     """
#     #     Convert bounding boxes from center format (cx, cy, w, h) to corner format (x1, y1, x2, y2),
#     #     remove padding, rescale to the original image size, and clip to image boundaries.

#     #     Args:
#     #         boxes_xywh (np.ndarray): Bounding boxes in center format (cx, cy, w, h).

#     #     Returns:
#     #         np.ndarray: Bounding boxes in corner format (x1, y1, x2, y2), clipped to image boundaries.
#     #     """
#     #     # Retrieve image dimensions, scale, and padding
#     #     image_height, image_width = self._image_shape
#     #     scale = self._scale
#     #     pad_top, pad_bottom, pad_left, pad_right = self._padding

#     #     # Step 1: Remove padding and rescale to the original image size
#     #     x_center = (boxes_xywh[:, 0] - pad_left) / scale
#     #     y_center = (boxes_xywh[:, 1] - pad_top) / scale
#     #     width = boxes_xywh[:, 2] / scale
#     #     height = boxes_xywh[:, 3] / scale

#     #     # Step 2: Convert center format (cx, cy, w, h) to corner format (x1, y1, x2, y2)
#     #     x1 = x_center - width / 2
#     #     y1 = y_center - height / 2
#     #     x2 = x_center + width / 2
#     #     y2 = y_center + height / 2

#     #     # Step 3: Clip coordinates to image boundaries
#     #     x1 = np.clip(x1, 0, image_width)
#     #     y1 = np.clip(y1, 0, image_height)
#     #     x2 = np.clip(x2, 0, image_width)
#     #     y2 = np.clip(y2, 0, image_height)

#     #     # Step 4: Stack the coordinates into a single array
#     #     return np.stack([x1, y1, x2, y2], axis=1)
#     # def preprocess(self, image: cv2.Mat) -> np.ndarray:
#     #     """
#     #     Preprocess the input image for the model.
        
#     #     Steps:
#     #     1. Convert the image to RGB format.
#     #     2. Resize the image while maintaining aspect ratio.
#     #     3. Pad the resized image to match the model's input dimensions.
#     #     4. Normalize the image and convert it to a blob.

#     #     Args:
#     #         image (cv2.Mat): Input image.
#     #     Returns:
#     #         np.ndarray: Preprocessed image blob.
#     #     """

#     #     # input dimensions of the model
#     #     input_height, input_width = self._input_shape[2:] # Model input shape (H, W)
#     #     image_height, image_width = image.shape[:2]       # Original image dimensions

#     #     # Step 1: Convert to RGB format
#     #     image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  

#     #     # Step 2: Resize the image while maintaining aspect ratio
#     #     scale = min(input_height / image_height, input_width / image_width)
#     #     resized_image = cv2.resize(image_rgb, (0,0), fx=scale, fy=scale)
#     #     resized_height, resized_width = resized_image.shape[:2]

#     #     # Step 3: Pad the resized image to match the model's input dimensions
#     #     pad_top = (input_height - resized_height) // 2
#     #     pad_bottom = input_height - resized_height - pad_top
#     #     pad_left = (input_width - resized_width) // 2
#     #     pad_right = input_width - resized_width - pad_left
#     #     padded_image = cv2.copyMakeBorder(
#     #         resized_image, 
#     #         pad_top, pad_bottom, pad_left, pad_right,
#     #         cv2.BORDER_CONSTANT,
#     #         value=(114, 114, 114) # Padding color (gray)
#     #     )

#     #     # Step 4: Normalize the image and convert it to a blob
#     #     blob = padded_image.transpose(2,0,1)[np.newaxis,...].astype(np.float32) / 255.0

#     #     # Save parameters for postprocessing
#     #     self._image_shape = (image_height, image_width)            # Original image dimensions
#     #     self._scale = scale                                        # Scale factor for resizing         
#     #     self._padding = (pad_top, pad_bottom, pad_left, pad_right) # Padding values

#     #     return blob
#     # def inference(self, blob: np.ndarray) -> list:
#     #     """
#     #     Perform inference on the preprocessed input blob using the ONNX model.
#     #     Args:
#     #         blob (np.ndarray): Preprocessed input image blob.

#     #     Returns:
#     #         list: Model outputs in the format Ex. [[batch, num_anchors, (cx, cy, w, h, conf, class_id)]].
#     #     """
#     #     # Start timing for inference
#     #     start_time = time.perf_counter()

#     #     # Perform inference using the ONNX runtime session
#     #     outputs = self._session.run(self._output_names, {self._input_names[0]: blob})

#     #     # Log inference time if logger is available
#     #     if self.logger:
#     #         inference_time_ms = (time.perf_counter() - start_time) * 1000
#     #         self.logger.debug(f"Inference time: {inference_time_ms:.2f} ms")

#     #     return outputs
#     # def postprocess(self, outputs: list) -> list:
#     #     """
#     #     Postprocess the model outputs to extract bounding boxes, class IDs, and confidence scores.

#     #     Steps:
#     #     1. Filter predictions based on object confidence.
#     #     2. Multiply class confidence with bounding box confidence.
#     #     3. Filter predictions based on class confidence.
#     #     4. Filter predictions to include only specified classes.
#     #     5. Extract bounding boxes and apply non-maximum suppression (NMS).

#     #     Args:
#     #         outputs (list): Model outputs in the format [batch, num_anchors, (cx, cy, w, h, conf, class_id)].

#     #     Returns:
#     #         list: A tuple containing filtered bounding boxes, class IDs, and confidence scores.
#     #     """
#     #     # Extract predictions from model outputs
#     #     predictions = outputs[0][0]  # Shape: [num_anchors, 5 + num_classes]

#     #     # Step 1: Filter predictions based on object confidence
#     #     object_confidence_mask = predictions[:, 4] > self._confidence_threshold
#     #     predictions = predictions[object_confidence_mask]
#     #     if predictions.shape[0] == 0:
#     #         return [], [], []

#     #     # Step 2: Multiply class confidence with bounding box confidence
#     #     confidence_scores = predictions[:, 4]
#     #     predictions[:, 5:] *= confidence_scores[:, np.newaxis]

#     #     # Step 3: Filter predictions based on class confidence
#     #     scores = np.max(predictions[:, 5:], axis=1)
#     #     class_confidence_mask = scores > self._confidence_threshold
#     #     predictions = predictions[class_confidence_mask]
#     #     scores = scores[class_confidence_mask]
#     #     if len(scores) == 0:
#     #         return [], [], []

#     #     # Step 4: Filter predictions to include only specified classes
#     #     class_ids = np.argmax(predictions[:, 5:], axis=1).astype(np.int32)
#     #     class_filter_mask = np.isin(class_ids, self._classes)
#     #     predictions = predictions[class_filter_mask]
#     #     scores = scores[class_filter_mask]
#     #     class_ids = class_ids[class_filter_mask]

#     #     # Step 5: Extract bounding boxes and apply non-maximum suppression (NMS)
#     #     boxes = self.extract_boxes(predictions)
#     #     selected_indices = nms(boxes, scores, self._iou_threshold)

#     #     print(f"Selected indices: {selected_indices}")

#     #     # Return filtered boxes, class IDs, and scores
#     #     return boxes[selected_indices], class_ids[selected_indices], scores[selected_indices]



from Properties.BoundingBox import BoundingBox
from Detection.Detection import Detection

import cv2
import onnxruntime as ort 
import numpy as np
import time

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
        self._model_path = model_name
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

        # THIRD FILTER - ONLY HUMAN CLASSES
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

  