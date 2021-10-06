import numpy as np    


def rotate_into_map_coord(vec, angle):
    # TODO: this can be done faster
    cos = np.cos(angle)
    sin = np.sin(angle)
    rot_mat = np.array([[cos, sin], [-sin, cos]])
    return vec.dot(rot_mat)


def closest_point_idx(point, other_points):
    dists = np.linalg.norm(other_points - point, axis=1)
    return np.argmin(dists)


def cyclic_slice(points, start_idx, num_points_ahead):
    end_idx = start_idx + num_points_ahead
    num_points_missing = end_idx - len(points)
    if num_points_missing <= 0:
        return points[start_idx:end_idx]
    else:
        return np.r_[points[start_idx:], points[:num_points_missing]]


def determine_direction_of_bound(bound, start_position, end_position):
    closest_bound_idx_start = closest_point_idx(bound, start_position)
    closest_bound_idx_end = closest_point_idx(bound, end_position)

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