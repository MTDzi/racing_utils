from typing import Tuple

import numpy as np
import torch
from scipy.interpolate import interp1d

from utils import rotate_into_map_coord, closest_point_idx

from .bezier import get_bezier_parameters
from .utils import cyclic_slice
from .torch_related import straighten_up_arc as straighten_up_arc_in_torch
from .torch_related import bezier_curve as bezier_curve_in_torch
from .torch_related import bezier_curve_batch as bezier_curve_batch_in_torch
from .torch_related import rotate_into_map_coord as rotate_into_map_coord_in_torch
from .torch_related import modify_waypoints, modify_waypoints_in_batch


class RandomProbePlanner:
    def __init__(
            self,
            csv_waypoints_path,
            direction,
            modification_steps,
            overall_steps,
            bound_decimation,
            step_length,
            bezier_degree,
            lowest_acceptable_dist,
            num_samples_for_grad,
            eta_for_grad,
            eta_for_update,
            device,
    ):
        self.modification_steps = modification_steps
        self.overall_steps = overall_steps
        self.bound_decimation = bound_decimation

        self.centerline = self._prep_points(str(csv_waypoints_path), direction, step_length)
        self.left_bound = self._prep_points(str(csv_waypoints_path).replace('SOCHI', 'interior'), direction, step_length)
        self.right_bound = self._prep_points(str(csv_waypoints_path).replace('SOCHI', 'exterior'), direction, step_length)
        self.waypoints = self.centerline.copy()

        self.lowest_acceptable_dist = lowest_acceptable_dist
        self.num_samples_for_grad = num_samples_for_grad
        self.eta_for_grad = eta_for_grad
        self.eta_for_update = eta_for_update

        self.device = device

        self.set_bezier_degree(bezier_degree)


    def _prep_points(self, csv_waypoints_path, direction, step_length):
        points = self.read_csv_as_array(csv_waypoints_path)[::direction]

        # We need the waypoints to be evenly spaced out
        points = np.r_[points, points[-1][np.newaxis]]  # Close the loop
        diffs = np.linalg.norm(np.diff(points, axis=0), axis=1)
        progress = np.r_[0, np.cumsum(diffs)]
        overall_length = progress[-1]
        progress /= overall_length
        num_points = int(overall_length / step_length)

        x_coord_fn = interp1d(progress, points[:, 0], assume_sorted=True)
        y_coord_fn = interp1d(progress, points[:, 1], assume_sorted=True)

        x_coords = x_coord_fn(np.linspace(0, 1, num_points))
        y_coords = y_coord_fn(np.linspace(0, 1, num_points))

        return np.c_[x_coords, y_coords]

    
    @staticmethod
    def read_csv_as_array(csv_path):
        with open(csv_path, 'r') as f:
            data = f.read().split('\n')[:-1]
            data = [[float(__) for __ in _.split(',')] for _ in data]
        return np.array(data)[:, :2]


    def set_bezier_degree(self, bezier_degree: int):
        self.bezier_degree = bezier_degree

        # Initialize the zero_bezier_change with control points corresponding to a straight line with
        #  all Y-s equal to zero (which means: don't modify the centerline)
        zero_modifications = np.c_[np.linspace(0, 1, self.modification_steps), np.zeros(self.modification_steps)]
        self.zero_bezier_change = np.r_[get_bezier_parameters(zero_modifications[:, 0], zero_modifications[:, 1], degree=self.bezier_degree)]
        self.zero_bezier_change = torch.tensor(self.zero_bezier_change, device=self.device)

        torch.manual_seed(0)
        self.num_modifiable_bezier_points = self.bezier_degree - 1  # Meaning: the terminal points will NOT be modified
        self.bezier_samples = torch.rand(self.num_samples_for_grad, self.num_modifiable_bezier_points)
        self.bezier_samples[:self.num_samples_for_grad // 2] *= -1
        self.bezier_samples = torch.column_stack([
            torch.zeros(self.num_samples_for_grad, 1),
            self.bezier_samples,
            torch.zeros(self.num_samples_for_grad, 1),
        ])
        self.bezier_samples /= torch.unsqueeze(torch.linalg.norm(self.bezier_samples, axis=1), 1)
        self.bezier_samples = torch.stack([torch.zeros_like(self.bezier_samples), self.bezier_samples], axis=2)
        self.bezier_samples = self.bezier_samples.to(self.device)


    @staticmethod
    def score_fn(waypoints, obstacles):
        distances = torch.linalg.norm(waypoints[:, None] - obstacles[None, :, None], axis=3)
        return -distances.min(axis=1).values.min(axis=1).values


    def extract_waypoints_and_bounds(
            self,
            yaw: float,
            position: np.array,
    ) -> Tuple[torch.tensor, torch.tensor, torch.tensor]:
        """Extract centerline and bound slices used for prediction and reward calculations."""
        points_slices = []
        for points in [self.left_bound, self.right_bound, self.waypoints, self.waypoints]:
            closest_idx = closest_point_idx(position, points)
            points_slice = cyclic_slice(points, closest_idx, self.overall_steps)
            points_slice = rotate_into_map_coord(points_slice - position, -yaw)
            points_slice = torch.tensor(points_slice, device=self.device, dtype=torch.float)
            points_slices.append(points_slice)

        self.closest_waypoint_idx = closest_idx
        left_bound, right_bound, waypoints, centerline = points_slices
        left_bound = left_bound[::self.bound_decimation]
        right_bound = right_bound[::self.bound_decimation]

        return waypoints, centerline, left_bound, right_bound, yaw


    def _modify_waypoints(self, bezier_points: np.array, waypoints: np.array, translation: np.array, yaw: float) -> np.array:
        bcurve = bezier_curve_in_torch(bezier_points, len(waypoints), self.device)
        mod_waypoints = torch.column_stack([
            waypoints[:, 0],
            waypoints[:, 1] + bcurve[:, 1]
        ])
        mod_waypoints = rotate_into_map_coord_in_torch(mod_waypoints, yaw, self.device) + translation

        return mod_waypoints


    def _modify_waypoints_in_batch(self, bezier_points: np.array, waypoints: np.array, translation: np.array, yaw: float) -> np.array:
        bcurve = bezier_curve_batch_in_torch(bezier_points, len(waypoints), self.device)
        mod_waypoints = torch.stack([
            waypoints[:, 0].repeat(self.num_samples_for_grad, 1),
            (waypoints[:, 1] + bcurve[..., 1]),
        ], axis=2)
        mod_waypoints = rotate_into_map_coord_in_torch(mod_waypoints, yaw, self.device) + translation

        return mod_waypoints


    def plan(self, position, yaw, ranges_as_vec):
        overall_waypoints, centerline, left_bound, right_bound, yaw = self.extract_waypoints_and_bounds(yaw, position)
        modifiable_waypoints = overall_waypoints[:self.modification_steps]
        
        base_score = max(
            self.score_fn(modifiable_waypoints[None], ranges_as_vec),
            self.score_fn(modifiable_waypoints[None], left_bound),
            self.score_fn(modifiable_waypoints[None], right_bound)
        )

        best_score = float(base_score)

        if base_score > -self.lowest_acceptable_dist:
            modifiable_waypoints, translation, local_yaw = straighten_up_arc_in_torch(modifiable_waypoints)
            bezier_points_modified = self.zero_bezier_change + self.eta_for_grad * self.bezier_samples
            
            waypoints_modified = modify_waypoints_in_batch(bezier_points_modified, modifiable_waypoints, translation, local_yaw)

            scores_modified = torch.max(torch.stack([
                self.score_fn(waypoints_modified, ranges_as_vec),
                self.score_fn(waypoints_modified, left_bound),
                self.score_fn(waypoints_modified, right_bound),    
            ]), axis=0).values

            if scores_modified.min() < base_score:
                offset = 0
                which_best = offset + torch.argmin(scores_modified)
                best_score = float(scores_modified[which_best])
                bezier_points = self.zero_bezier_change + self.eta_for_update * self.bezier_samples[which_best]
                modifiable_waypoints = modify_waypoints(bezier_points, modifiable_waypoints, translation, local_yaw)
            else:
                modifiable_waypoints = rotate_into_map_coord_in_torch(modifiable_waypoints, local_yaw) + translation

        overall_waypoints[:self.modification_steps] = modifiable_waypoints

        return overall_waypoints, centerline, left_bound, right_bound, best_score


    def update_waypoints(self, waypoints, position, yaw):
        # Now we can modify the original waypoints
        waypoints_back_in_map_coordinates = rotate_into_map_coord(waypoints, yaw) + position
        if self.closest_waypoint_idx + self.overall_steps >= self.waypoints.shape[0]:
            self.waypoints = np.roll(self.waypoints, -self.closest_waypoint_idx, axis=0)
            self.centerline = np.roll(self.centerline, -self.closest_waypoint_idx, axis=0)
            self.closest_waypoint_idx = 0

        self.waypoints[self.closest_waypoint_idx:(self.closest_waypoint_idx + self.overall_steps)] = waypoints_back_in_map_coordinates
        
        return waypoints
