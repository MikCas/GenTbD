from Tracking.Partition import Partition 
import numpy as np
from scipy.optimize import linear_sum_assignment
from typing import List, Any

# PLACE THESE FUNCTIONS AS STATIC METHODS IN THE TRACKER CLASS

# DEFINE AS AN ABSTRACT STATIC METHOD FOR A GIVEN 
# THE ACTUAL MATCH SHOULD BE A SELF PARAMETER OF THE TRACKER CLASS SO CAN REDEFINE IN THEIR
def match_condition(cost: float, match_threshold: float) -> bool: 
    """
    Check if the cost of a match is below the match threshold. 
    A cost that is less than the match threshold implies that the matched objects are similar enough to be considered a good match.

    Args:
        cost (float): The cost of the match.
        match_threshold (float): The threshold for determining a match.
    Returns:
        bool: True if match is valid (passes the condition), otherwise False. 
    """
    return cost <= match_threshold

def calculate_cost_matrix(xs: List[Any], ys: List[Any]) -> np.ndarray:
    """
    Calculate the cost matrix for matching elements in two sets.

    Args:
        xs (List[Any]): The first set of elements.
        ys (List[Any]): The second set of elements.

    Returns:
        np.ndarray: A 2D cost matrix where each entry represents the cost of matching an element from `xs` to `ys`.
    """
    num_xs = len(xs)
    num_ys = len(ys)

    # Initialize the cost matrix with ones
    cost_matrix = np.ones((num_xs, num_ys))

    # Populate the cost matrix with calculated costs
    for i, x in enumerate(xs):
        for j, y in enumerate(ys):
            cost_matrix[i, j] = x.calculate_cost(y)

    return cost_matrix

def linear_assignment(
    frame_count: int,
    xs: List[Any],
    ys: List[Any],
    match_threshold: float
) -> Partition:
    """
    Perform linear assignment to match elements in two sets based on a cost matrix.

    Args:
        frame_count (int): The current frame count.
        xs (List[Any]): The first set of elements to match.
        ys (List[Any]): The second set of elements to match.
        match_threshold (float): The maximum allowable cost for a match.

    Returns:
        Partition: A Partition object containing matched pairs and unmatched elements.
    """
    # Handle empty input sets
    if len(xs) == 0 or len(ys) == 0:
        return Partition(unmatched_x=xs, unmatched_y=ys)

    # Calculate the cost matrix
    cost_matrix = calculate_cost_matrix(xs, ys)

    # Perform linear sum assignment
    matched_xs_indexes, matched_ys_indexes = linear_sum_assignment(cost_matrix)

    # Initialize partitions
    matches = []
    unmatched_xs = [xs[i] for i in range(len(xs)) if i not in matched_xs_indexes]
    unmatched_ys = [ys[i] for i in range(len(ys)) if i not in matched_ys_indexes]

    # Process matches
    for x_index, y_index in zip(matched_xs_indexes, matched_ys_indexes):
        x = xs[x_index]
        y = ys[y_index]

        # Check if the match satisfies the condition
        if match_condition(cost_matrix[x_index][y_index], match_threshold):
            x.update_matched(frame_count, y)
            matches.append(x)
        else:
            unmatched_xs.append(x)
            unmatched_ys.append(y)

    # Return the partition
    return Partition(matched=matches, unmatched_x=unmatched_xs, unmatched_y=unmatched_ys)