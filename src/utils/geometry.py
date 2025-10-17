"""
Geometry utilities for bounding box operations.

This module provides utility functions for geometric operations on bounding boxes,
including Intersection over Union (IoU) calculations and Non-Maximum Suppression (NMS).
"""

import numpy as np
from typing import List


def compute_iou_array(box: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    """
    Compute the Intersection over Union (IoU) between a given box and a set of boxes.

    This is optimized for NMS operations where one box needs to be compared against
    many boxes efficiently.

    Args:
        box (np.ndarray): A single bounding box in corner format (x1, y1, x2, y2).
        boxes (np.ndarray): An array of bounding boxes in corner format (x1, y1, x2, y2).
                           Shape: (N, 4) where N is the number of boxes.

    Returns:
        np.ndarray: An array of IoU values between the given box and each box in the input array.
                    Shape: (N,)

    Example:
        >>> box = np.array([10, 10, 50, 50])
        >>> boxes = np.array([[15, 15, 45, 45], [60, 60, 100, 100]])
        >>> ious = compute_iou_array(box, boxes)
        >>> print(ious)  # [0.56, 0.0]  (approximate values)
    """
    # Calculate the coordinates of the intersection rectangle
    xmin = np.maximum(box[0], boxes[:, 0])
    ymin = np.maximum(box[1], boxes[:, 1])
    xmax = np.minimum(box[2], boxes[:, 2])
    ymax = np.minimum(box[3], boxes[:, 3])

    # Compute the area of the intersection rectangle
    intersection_width = np.maximum(0, xmax - xmin)
    intersection_height = np.maximum(0, ymax - ymin)
    intersection_area = intersection_width * intersection_height

    # Compute the area of the union
    box_area = (box[2] - box[0]) * (box[3] - box[1])
    boxes_area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    union_area = box_area + boxes_area - intersection_area

    # Compute the IoU
    iou = intersection_area / np.maximum(union_area, 1e-6)  # Avoid division by zero

    return iou


def nms(boxes: np.ndarray, scores: np.ndarray, iou_threshold: float) -> List[int]:
    """
    Perform Non-Maximum Suppression (NMS) to filter overlapping bounding boxes.

    NMS keeps only the boxes with highest confidence scores and suppresses boxes
    that significantly overlap (IoU > threshold) with higher-scoring boxes.

    Args:
        boxes (np.ndarray): Array of bounding boxes in corner format (x1, y1, x2, y2).
                           Shape: (N, 4) where N is the number of boxes.
        scores (np.ndarray): Array of confidence scores corresponding to the bounding boxes.
                            Shape: (N,)
        iou_threshold (float): IoU threshold for suppressing overlapping boxes.
                              Boxes with IoU > threshold will be suppressed.
                              Typical values: 0.3-0.7

    Returns:
        List[int]: Indices of the bounding boxes to keep after applying NMS.

    Example:
        >>> boxes = np.array([[10, 10, 50, 50], [15, 15, 45, 45], [100, 100, 150, 150]])
        >>> scores = np.array([0.9, 0.8, 0.95])
        >>> keep_indices = nms(boxes, scores, iou_threshold=0.5)
        >>> print(keep_indices)  # [2, 0] (keeps box 2 and box 0, suppresses box 1)
    """
    # Sort indices of boxes by scores in descending order
    sorted_indices = np.argsort(scores)[::-1]

    keep_boxes = []
    while sorted_indices.size > 0:
        # Select the box with the highest score
        current_box_index = sorted_indices[0]
        keep_boxes.append(current_box_index)

        # Compute IoU of the selected box with the remaining boxes
        ious = compute_iou_array(boxes[current_box_index], boxes[sorted_indices[1:]])

        # Filter out boxes with IoU above the threshold
        remaining_indices = np.where(ious < iou_threshold)[0]

        # Update the sorted indices to exclude suppressed boxes
        sorted_indices = sorted_indices[remaining_indices + 1]

    return keep_boxes
