"""
Contains functions that did not fit any other module.
"""
from functools import lru_cache

import numpy as np


@lru_cache(maxsize=1000)
def get_rotation_matrix(angle_rad: float) -> np.array:
    """Construct and memoize a 2D rotation matrix for a given angle (in radians)."""
    cos = np.cos(angle_rad)
    sin = np.sin(angle_rad)
    return np.array([[cos, sin], [-sin, cos]])


def rotate_into_map_coord(vec: np.array, angle_rad: float) -> np.array:
    """Rotate a vector (or vectors) by a given angle (in radians)."""
    rot_mat = get_rotation_matrix(angle_rad)
    return vec.dot(rot_mat)


def closest_point_idx(point_or_points: np.array, other_points: np.array) -> int:
    """Find the indices of the points in other_points that are closest to the points in point_or_points."""
    return_one = (len(point_or_points.shape) == 1)
    dists = np.linalg.norm(other_points[:, np.newaxis] - point_or_points[np.newaxis], axis=2)
    argmins = np.argmin(dists, axis=0)
    if return_one:
        return argmins[0]
    else:
        return argmins


def cyclic_slice(points: np.array, start_idx: int, num_points_ahead: int) -> np.array:
    end_idx = start_idx + num_points_ahead
    num_points_missing = end_idx - len(points)
    if num_points_missing <= 0:
        return points[start_idx:end_idx]
    else:
        return np.r_[points[start_idx:], points[:num_points_missing]]


def determine_direction_of_bound(bound: np.array, start_position: np.array, end_position: np.array):
    closest_bound_idx_start = closest_point_idx(start_position, bound)
    closest_bound_idx_end = closest_point_idx(end_position, bound)

    idx_diff = closest_bound_idx_end - closest_bound_idx_start
    halfway = len(bound) // 2
    its_cyclic = (abs(idx_diff) > halfway)
    if its_cyclic:
        if closest_bound_idx_start > halfway:
            return 1
        else:
            return -1

    if idx_diff > 0:
        return 1
    else:
        return -1


def reward_for_getting_far(positions: np.array, centerline_ahead: np.array) -> float:
    last_position = positions[-1]
    dists = np.linalg.norm(centerline_ahead - last_position, axis=1)
    closest_centerline_idx = np.argmin(dists, axis=0)
    reward = np.linalg.norm(np.diff(centerline_ahead, axis=0), axis=1)[:closest_centerline_idx].sum()
    link_to_last_position = last_position - centerline_ahead[closest_centerline_idx]
    link_to_last_position /= (np.linalg.norm(link_to_last_position) + 1e-10)
    
    try:
        last_centerline_vector = centerline_ahead[closest_centerline_idx + 1] - centerline_ahead[closest_centerline_idx + 1]
        reward -= link_to_last_position.dot(last_centerline_vector)
    except IndexError:
        print('There was an instance where "closest_centerline_idx" was the last point')

    return float(reward)


def penalty_from_bounds(positions: np.array, left_bound: np.array, right_bound: np.array, sigma: float = 0.4) -> float:
    penalty = 0
    for bound in [left_bound, right_bound]:
        exps = np.exp(-np.linalg.norm(positions[np.newaxis] - bound[:, np.newaxis], axis=2) / sigma**2)
        penalty += exps.sum()

    return penalty
