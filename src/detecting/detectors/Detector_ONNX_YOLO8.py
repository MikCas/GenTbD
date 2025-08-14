from properties import BoundingBox
# from detecting import ObjectDetection as Detection
from ..detections.ObjectDetection import ObjectDetection as Detection
from .Detector import Detector

from abc import ABC
import cv2
import onnxruntime as ort
import numpy as np
import time
import logging


class YOLOv8ONNX(Detector):
    """
    YOLOv8 model in ONNX format.

    Attributes:
        iou_threshold (float): IoU threshold for non-maximum suppression
        classes (list): List of class IDs to detect
        image_shape (tuple): Original image dimensions (height, width) [preprocess()
        scale (float): Scale factor for resizing [preprocess()]
        padding (tuple): Padding values (top, bottom, left, right) [preprocess()]

        session (ort.InferenceSession): ONNX inference session.
        input_names (list): List of input names for the model.
        input_shape (tuple): Shape of the model input.
        input_height (int): Height of the model input.
        input_width (int): Width of the model input.
        output_names (list): List of output names for the model.
    """

    ##### SETUP #####
    def create_onnx_model(self, model_path: str) -> None:
        """
        Set up the ONNX model.
        Args:
            model_path (str): Path to the ONNX model file.
        Returns:
            ort.InferenceSession: Inference session for the ONNX model.
        """
        try:
            # Create the ONNX inference session
            self._session = ort.InferenceSession(model_path, providers=['CoreMLExecutionProvider',
                                                                        'CPUExecutionProvider'])  # Apple Silicon
            # session = ort.InferenceSession(model_path, providers=['CUDAExecutionProvider', 'CPUExecutionProvider']) # NVIDIA GPU
            # session = ort.InferenceSession(model_path, providers=['OpenVINOExecutionProvider', 'CPUExecutionProvider']) # Intel CPU
            # session = ort.InferenceSession(model_path, providers=['DmlExecutionProvider', 'CPUExecutionProvider']) # Windows GPU

            # Model input details
            model_inputs = self._session.get_inputs()
            self._input_names = [input.name for input in model_inputs]  # Names
            # TODO This needs to be fixed. Some models break this.
            self._input_shape = model_inputs[0].shape  # Shape Ex. (1, 3, 640, 640)
            self._input_height = self._input_shape[2]  # Height
            self._input_width = self._input_shape[3]  # Width

            # Model output details
            model_outputs = self._session.get_outputs()
            self._output_names = [output.name for output in model_outputs]  # Names

        except Exception as e:
            self._logger.error(f"COULD NOT LOAD ONNX MODEL: {e}")
            raise

    def __init__(self, model_path, confidence_threshold=0.1, iou_threshold=0.5, classes=[0], logger=None):
        super().__init__(model_path, confidence_threshold, logger)
        self._iou_threshold = iou_threshold
        self._classes = classes
        self.create_onnx_model(model_path)
        self.log(logging.INFO, f"|| DETECTOR INITIALISED\n"
                               f"\t\t\t\t    - MODEL: {model_path}\n"
                               f"\t\t\t\t    - CONFIDENCE: {confidence_threshold}\n"
                               f"\t\t\t\t    - IOU: {iou_threshold}")

    ##### DETECTION #####
    # TODO: CHANGE THE EXTRACT_BOXES METHOD TO USE THE BOUNDINGBOX CLASS
    def extract_boxes(self, boxes_xywh: np.ndarray) -> np.ndarray:
        """
        Convert bounding boxes from center format (cx, cy, w, h) to corner format (x1, y1, x2, y2),
        remove padding, rescale to the original image size, and clip to image boundaries.

        Args:
            boxes_xywh (np.ndarray): Bounding boxes in center format (cx, cy, w, h).

        Returns:
            np.ndarray: Bounding boxes in corner format (x1, y1, x2, y2), clipped to image boundaries.
        """
        # Retrieve image dimensions, scale, and padding
        image_height, image_width = self._image_shape
        scale = self._scale
        pad_top, pad_bottom, pad_left, pad_right = self._padding

        # Remove padding and rescale to the original image size
        x_center = (boxes_xywh[:, 0] - pad_left) / scale
        y_center = (boxes_xywh[:, 1] - pad_top) / scale
        width = boxes_xywh[:, 2] / scale
        height = boxes_xywh[:, 3] / scale

        # Convert center format (cx, cy, w, h) to corner format (x1, y1, x2, y2)
        x1 = x_center - width / 2
        y1 = y_center - height / 2
        x2 = x_center + width / 2
        y2 = y_center + height / 2

        # x2 = boxes_xywh[:, 0] / scale
        # y2 = boxes_xywh[:, 1] / scale
        # x1 = boxes_xywh[:, 2] / scale
        # y1 = boxes_xywh[:, 3] / scale

        # Clip coordinates to image boundaries
        x1 = np.clip(x1, 0, image_width)
        y1 = np.clip(y1, 0, image_height)
        x2 = np.clip(x2, 0, image_width)
        y2 = np.clip(y2, 0, image_height)

        # Stack the coordinates into a single array
        return np.stack([x1, y1, x2, y2], axis=1)

    def create_detections(self, data):
        """
        Create Detection objects from model output data.

        Args:
            data (tuple): A tuple containing:
                - np.ndarray: Bounding boxes in corner format (x1, y1, x2, y2).
                - np.ndarray: Class IDs corresponding to the bounding boxes.
                - np.ndarray: Confidence scores for the bounding boxes.

        Returns:
            list: A list of Detection objects.
        """
        boxes, class_ids, scores = data
        return [
            Detection(
                class_id,
                BoundingBox.from_corners(*box),
                confidence_score
            )
            for box, class_id, confidence_score in zip(boxes, class_ids, scores)
        ]

    def preprocess(self, image: cv2.Mat) -> np.ndarray:
        """
        Preprocess the input image for the model.

        Steps:
        1. Convert the image to RGB format.
        2. Resize the image while maintaining aspect ratio.
        3. Pad the resized image to match the model's input dimensions.
        4. Normalize the image and convert it to a blob.

        Args:
            image (cv2.Mat): Input image.
        Returns:
            np.ndarray: Preprocessed image blob.
        """
        self.log(logging.INFO, "\t|| PREPROCESSING")

        # input dimensions of the model
        input_height, input_width = self._input_shape[2:]  # Model input shape (H, W)
        image_height, image_width = image.shape[:2]  # Original image dimensions

        # Step 1: Convert to RGB format
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Step 2: Resize the image while maintaining aspect ratio
        scale = min(input_height / image_height, input_width / image_width)
        # resized_image = cv2.resize(image_rgb, (0, 0), fx=scale, fy=scale) #Mikahil's
        resized_image = cv2.resize(image_rgb, (input_width, input_height))
        resized_height, resized_width = resized_image.shape[:2]
        #
        # Step 3: Pad the resized image to match the model's input dimensions
        pad_top = (input_height - resized_height) // 2
        pad_bottom = input_height - resized_height - pad_top
        pad_left = (input_width - resized_width) // 2
        pad_right = input_width - resized_width - pad_left
        padded_image = cv2.copyMakeBorder(
            resized_image,
            pad_top, pad_bottom, pad_left, pad_right,
            cv2.BORDER_CONSTANT,
            value=(114, 114, 114)  # Padding color (gray)
        )

        # # Step 4: Normalize the image and convert it to a blob
        # blob = padded_image.transpose(2, 0, 1)[np.newaxis, ...].astype(np.float32) / 255.0


        blob = padded_image.transpose(2, 0, 1)[np.newaxis, ...].astype(np.float32) / 255.0

        debug_img = (blob[0].transpose(1, 2, 0) * 255).astype(np.uint8)
        cv2.imshow("Input to model", debug_img)
        cv2.waitKey(0)

        # Save parameters for postprocessing
        self._image_shape = (image_height, image_width)  # Original image dimensions
        self._scale = scale  # Scale factor for resizing
        self._padding = (pad_top, pad_bottom, pad_left, pad_right)  # Padding values

        return blob

    def inference(self, blob: np.ndarray) -> list:
        """
        Perform inference on the preprocessed input blob using the ONNX model.

        Args:
            blob (np.ndarray): Preprocessed input image blob.

        Returns:
            list: Model outputs in the format [[batch, num_anchors, (cx, cy, w, h, conf, class_id)]].
        """
        self.log(logging.INFO, "\t|| INFERENCE")
        # Start timing for inference
        start_time = time.perf_counter()

        print("Blob shape:", blob.shape)
        print("Blob dtype:", blob.dtype)
        print("Min/Max:", np.min(blob), np.max(blob))
        print("Input name:", self._session.get_inputs()[0].name)
        print("Model expects shape:", self._session.get_inputs()[0].shape)

        # Perform inference using the ONNX runtime session
        outputs = self._session.run(self._output_names, {self._input_names[0]: blob})

        # DEBUG: Inspect raw outputs
        print(f"Number of outputs: {len(outputs)}")
        for i, output in enumerate(outputs):
            print(f"Output {i} shape: {output.shape}")
            print(f"Output {i} sample values: {output[0, :5, :10] if len(output.shape) > 2 else output[:10]}")

        # Log inference time if a logger is available
        if self._logger:
            inference_time_ms = (time.perf_counter() - start_time) * 1000
            self.log(logging.INFO, f"\t// INFERENCE TIME: {inference_time_ms:.2f} ms")

        return outputs

    def postprocess(self, outputs: list) -> tuple:
        """
        Postprocess the model outputs to extract bounding boxes, class IDs, and confidence scores.

        Steps:
        1. Filter predictions based on object confidence.
        2. Multiply class confidence with bounding box confidence.
        3. Filter predictions based on class confidence.
        4. Filter predictions to include only specified classes.
        5. Extract bounding boxes and apply non-maximum suppression (NMS).

        Args:
            outputs (list): Model outputs from YOLOv8 ONNX model.

        Returns:
            tuple: A tuple containing:
                - np.ndarray: Filtered bounding boxes in corner format (x1, y1, x2, y2).
                - np.ndarray: Class IDs corresponding to the filtered bounding boxes.
                - np.ndarray: Confidence scores for the filtered bounding boxes.
        """

        self.log(logging.INFO, "\t|| POSTPROCESS")

        # Handle different YOLOv8 output formats
        predictions = outputs[0]  # Shape: [1, num_predictions, features]
        
        # DEBUG: Initial predictions info
        print(f"\nDEBUG Postprocess:")
        print(f"Raw predictions shape: {predictions.shape}")
        print(f"Number of features per prediction: {predictions.shape[2]}")
        print(f"Classes we're looking for: {self._classes}")

        predictions = np.transpose(predictions, (0, 2, 1))
        predictions = np.squeeze(predictions)
        # Step 1: Filter predictions based on object confidence
        object_confidence_mask = predictions[:, 4] > self._confidence_threshold
        predictions = predictions[object_confidence_mask]
        if predictions.shape[0] == 0:
            return np.array([]), np.array([]), np.array([])

        # Step 2: Multiply class confidence with bounding box confidence
        confidence_scores = predictions[:, 4]
        predictions[:, 5:] *= confidence_scores[:, np.newaxis]

        predictions[:, 4:] = 1 / (1 + np.exp(-predictions[:, 4:]))  # sigmoid

        # Step 3: Filter predictions based on class confidence
        scores = np.max(predictions[:,  5:], axis=1)
        class_confidence_mask = scores > self._confidence_threshold
        predictions = predictions[class_confidence_mask]
        scores = scores[class_confidence_mask]
        if len(scores) == 0:
            return np.array([]), np.array([]), np.array([])

        # Step 4: Filter predictions to include only specified classes
        class_ids = np.argmax(predictions[:, 5:], axis=1).astype(np.int32)
        class_filter_mask = np.isin(class_ids, self._classes)
        predictions = predictions[class_filter_mask]
        scores = scores[class_filter_mask]
        class_ids = class_ids[class_filter_mask]

        print("Scale:", self._scale)
        print("Padding:", self._padding)

        # Step 5: Extract bounding boxes and apply non-maximum suppression (NMS)
        boxes = self.extract_boxes(predictions)
        selected_indices = Detector.nms(boxes, scores, self._iou_threshold)

        # Return filtered boxes, class IDs, and scores
        return boxes[selected_indices], class_ids[selected_indices], scores[selected_indices]

