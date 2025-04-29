from Tracking.Partition import Partition 

import numpy as np
from scipy.optimize import linear_sum_assignment

def match_condition(cost, match_threshold): return cost <= match_threshold

def calculate_cost_matrix(xs, ys):
    num_xs = len(xs)
    num_ys = len(ys)

    cost_matrix = np.ones((num_xs, num_ys))

    for i, x in enumerate(xs):
        for j, y in enumerate(ys):
            cost_matrix[i, j] = x.calculate_cost(y)
    
    return cost_matrix

def linear_assignment(frameCount, xs, ys, match_threshold):

    # EMPTY PARTITION
    if len(xs) == 0 or len(ys) == 0: return Partition.create_unmatched(xs, ys)

    # CALCULATE COST MATRIX
    cost_matrix = calculate_cost_matrix(xs, ys)

    # LINEAR ASSIGNMENT
    matched_xs_indexes, matched_ys_indexes = linear_sum_assignment(cost_matrix)

    # PARTITION
    matches = []
    unmatched_xs = [xs[i] for i in range(len(xs)) if i not in matched_xs_indexes]
    unmatched_ys = [ys[i] for i in range(len(ys)) if i not in matched_ys_indexes]

    # PROCESS MATCHES
    for x_index, y_index in zip(matched_xs_indexes, matched_ys_indexes):
        x = xs[x_index]
        y = ys[y_index]

        if match_condition(cost_matrix[x_index][y_index], match_threshold):
            x.update_matched(frameCount, y)
            matches.append(x)
            
        else:
            unmatched_xs.append(x)
            unmatched_ys.append(y)  

    return Partition(matches, unmatched_xs, unmatched_ys)