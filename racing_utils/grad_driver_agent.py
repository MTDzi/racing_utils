from typing import Dict, Optional, Sequence, Tuple, Dict, Any
import matplotlib.pyplot as plt

import numpy as np

import torch

from .torch_related import TensorStandardScaler, calc_progress_and_penalty_while_driving
from .utils import rotate_into_map_coord, closest_point_idx, cyclic_slice, straighten_up_arc
from .base_agent import BaseAgent
from .bezier import get_bezier_parameters, bezier_curve


NOT_MODIFIED_IDX = 0
POINT_DIM = 2
IDX = 0


class GradientDriverAgent(BaseAgent):
    """Controls the car using a trained model + modifying part of its input so as to maximize reward.

    Runs inference for the model and modifies the controller parameters (that are part of the input of the model)
    according to an approximated gradient of the (progress - penalty) objective function.

    Attributes:
        points: A list containing the arrays: centerline, left_bound, and right_bound.
        num_steps: How many elements from arrays in the points list are used.

        omniward_model: Model that predicts the future trajectory (2D positions of the car) and
            future actuator actions needed to realize that trajectory.
        features_scalers: A mapping from feature names to features scalers that are required to make predictions
            with the model.
        targets_scalers: A mapping from target names to target scalers that are required to unpack predictions
            made with the model (so as to get proper actuator values).

        eta: Step size used in the space of controller parameters (after scaling) used for approximating the gradient
            of the objective function; the bigger the step, the less reliable it is; the smaller the step, the more
            noisy the approximation. After approximating the gradient, controller parameter values are modified via a crude
            gradient descent algorithm in which the step size is also eta (although almost certainly a different value would be
            more suitable, yet for simplicity sake the step size in both cases is the same, and it's this attribute eta).
        num_steps_for_grad: Number of steps in the positive direction of each of the controller parameters when approximating
            the gradient. Consequently, the size of the batch is (2 * num_steps_for_grad + 1) ** num_contr_params.
        num_contr_params: Number of controller parameters emulated by the model, i.e. used as input by the model and
            modifying its behavior based on their values.
        step_directions: A matrix whose rows correnspond to vectors by which the controller parameter values ought to be
            modified when forming a batch for estimating.

        penalty_scale_coeff: The objective function is actually not (progress - penalty) but rather:
            progress + penalty_scale_coeff * penalty.
        penalty_sigma: The sigma used in the Gaussian-distribution-like components when calculating the penalty (see 
            the calc_progress_and_penalty function).
        contr_params_limits: When doing gradient descent on the controller parameters, the new values will be clipped
            by the values specified in this tensor.
    """

    SCAN_LEN = 1080
    ANGLE_MIN = -2.34999990463
    ANGLE_INCR = 0.00435185199603
    IDX = 0

    def __init__(
            self,

            # Map-related
            centerline: np.array,
            num_steps_centerline: int,

            left_bound: np.array,
            right_bound: np.array,
            num_steps_ahead_bound: int,

            # Controller-related
            init_contr_params: np.array,

            # Model-related
            omniward_model: torch.nn.Module,
            features_scalers: Dict[str, TensorStandardScaler],
            targets_scalers: Dict[str, TensorStandardScaler],

            # Bezier-related
            bezier_degree: int,
            num_alternatives: int = 11,

            # Gradient-related
            eta_for_grad: float = 0.1,
            eta_for_update: Optional[float] = None,
            num_steps_for_grad: int = 3,
            penalty_sigma: float = 0.2,
            penalty_scale_coeff: float = -0.9,
            contr_params_limits: torch.Tensor = torch.tensor([
                (3.0, 5.0),
                (8.0, 13.0),
                (8.0, 12.0),
            ]),
            only_closest_for_penalty: bool = False,

            device: str = 'cpu',
            debug: bool = False,
    ):
        self.points = [centerline, left_bound, right_bound]
        self.nums_steps = [num_steps_centerline, num_steps_ahead_bound, num_steps_ahead_bound]

        self.omniward_model = omniward_model
        self.features_scalers = features_scalers
        self.targets_scalers = targets_scalers

        self.eta_for_grad = eta_for_grad
        self.eta_for_update = eta_for_update or eta_for_grad
        self.num_steps_for_grad = num_steps_for_grad

        self.curr_delta = 0.0
        self.curr_speed = 0.0
        self.curr_contr_params = torch.tensor(init_contr_params, device=device, dtype=torch.float)

        self.device = device

        self.num_contr_params = len(init_contr_params)
        self.contr_params_step_directions = torch.cat(
            [-step * torch.eye(self.num_contr_params) for step in reversed(range(1, num_steps_for_grad + 1))]
            + [ step * torch.eye(self.num_contr_params) for step in range(1, num_steps_for_grad + 1)],
        axis=0).to(self.device)
        self.contr_params_step_shape = self.contr_params_step_directions.shape[0]

        self.set_bezier_degree(bezier_degree, num_alternatives)

        assert penalty_scale_coeff < 0, (
            'The penalty_scale_coeff passed was positive which makes no sense given that the objective function is:\n'
            '\t progress + penalty_scale_coeff * penalty\n'
        )
        self.penalty_scale_coeff = penalty_scale_coeff
        self.penalty_sigma = penalty_sigma
        self.contr_params_limits = contr_params_limits.to(self.device)
        self.only_closest_for_penalty = only_closest_for_penalty

        self.bezier_degree = bezier_degree
        self.debug = debug

        # For transforming ranges into ranges_as_vec
        self.idx_to_angle = np.array([
            self.ANGLE_MIN + idx * self.ANGLE_INCR
            for idx in range(self.SCAN_LEN)
        ])
        self.versor = np.array([
            np.array([np.cos(self.idx_to_angle[idx]), np.sin(self.idx_to_angle[idx])])
            for idx in range(self.SCAN_LEN)
        ])

    def set_bezier_degree(self, bezier_degree: int, num_alternatives: int): # TODO: alloc num_alternatives
        self.bezier_degree = bezier_degree
        num_steps_centerline = self.nums_steps[0]

        # Initialize the curr_bezier_points_delta with control points corresponding to a straight line with
        #  all Y-s equal to zero (which means: don't modify the centerline)
        zero_modifications = np.c_[np.linspace(0, 1, num_steps_centerline), np.zeros(num_steps_centerline)]
        self.curr_bezier_points_delta = np.r_[get_bezier_parameters(zero_modifications[:, 0], zero_modifications[:, 1], degree=self.bezier_degree)]

        # We'll keep a couple of alternative Bezier modifications in case the optimization gets stuck
        # TODO: could just copy the curr_bezier_points_deltas 
        self.alternative_bezier_point_deltas = np.r_[[get_bezier_parameters(zero_modifications[:, 0], zero_modifications[:, 1], degree=self.bezier_degree)]]
        self.alternative_bezier_point_deltas = np.r_[[]]
        self.num_alternatives = len(self.alternative_bezier_point_deltas)  # TODO

        self.num_bezier_points = self.bezier_degree + 1
        self.bezier_points_step_directions = np.concatenate([
            [-step * np.eye(POINT_DIM * self.num_bezier_points) for step in reversed(range(1, self.num_steps_for_grad + 1))]
            + [ step * np.eye(POINT_DIM * self.num_bezier_points) for step in range(1, self.num_steps_for_grad + 1)]
        ], axis=0)
        self.bezier_points_step_directions = self.bezier_points_step_directions.reshape(

            2 * self.num_steps_for_grad,  # "2 * " because there are two directions per dimension, i.e. (-step, +step)
            POINT_DIM * self.num_bezier_points,  # each point has POINT_DIM coordinates

            # Now, for each direction and each Bezier point coordinate there's a matrix of the 
            #  following shape (only one position has "1", there rest are "0") that
            #  will be added to the Bezier points
            self.num_bezier_points,
            POINT_DIM,
        )
        self.bezier_points_step_shape = self.bezier_points_step_directions.shape[0] * self.bezier_points_step_directions.shape[1]

        self.batch_size = 1 + self.contr_params_step_shape + self.bezier_points_step_shape + self.num_alternatives



    def plan(
            self,
            ranges: Sequence[float],
            yaw: float,
            pos_x: float,
            pos_y: float,
            linear_vel_x: float,
            linear_vel_y: float,
            angular_vel_z: float,
            lap_time: Optional[float] = None,
            **kwargs,
    ) -> Tuple[float, float]:




        # For transforming ranges into ranges_as_vec
        self.idx_to_angle = np.array([
            self.ANGLE_MIN + idx * self.ANGLE_INCR
            for idx in range(self.SCAN_LEN)
        ])
        self.versor = np.array([
            np.array([np.cos(self.idx_to_angle[idx]), np.sin(self.idx_to_angle[idx])])
            for idx in range(self.SCAN_LEN)
        ])
        self.grad_clip_threshold = 1.0

        self.alternative_better_threshold = 0.1





        position = np.array([pos_x, pos_y])
        centerline, left_bound, right_bound = self.extract_centerline_and_bounds(yaw, position)

        # This is where the prediction takes place
        trajectory_pred, actuators_pred, curr_contr_params_scaled = self.calc_predictions(
            centerline,
            linear_vel_x,
            linear_vel_y,
            angular_vel_z,
        )

        # First, let's extract the actuator values needed for driving the car
        speeds_and_deltas = self.targets_scalers['speeds_and_deltas'].inverse_transform(actuators_pred[NOT_MODIFIED_IDX]).cpu()
        first_speed_actuator_idx = 0
        first_delta_actuator_idx = len(speeds_and_deltas) // 2
        self.curr_speed = float(speeds_and_deltas[first_speed_actuator_idx])
        self.curr_delta = float(speeds_and_deltas[first_delta_actuator_idx])

        # Turn the lidar scan into 2D points in the car's reference frame
        ranges_as_vec = torch.tensor(ranges[:, np.newaxis] * self.versor, requires_grad=False, device=self.device)
        
        #                                                                                  # 
        #  Now for the heavier part: calculating the gradient of the progress and penalty  #
        #                                                                                  #
        # 1) To estimate the gradient of the progress we need the trajectory
        trajectory_pred = self.targets_scalers['trajectory'].inverse_transform(trajectory_pred)
        progress_pred, penalty_pred = calc_progress_and_penalty_while_driving(
            trajectory_pred,
            centerline,
            left_bound,
            right_bound,
            ranges=ranges_as_vec,
            penalty_sigma=self.penalty_sigma,
        )

        reward_pred = (progress_pred + self.penalty_scale_coeff * penalty_pred).cpu()
        base_reward = float(reward_pred[NOT_MODIFIED_IDX])

        # Let's check if any of the alternatives is better than the base case
        alternative_rewards = reward_pred[1:(1 + self.num_alternatives)]
        # best_alternative = torch.argmax(alternative_rewards - base_reward)
        if False: # alternative_rewards[best_alternative] > base_reward + self.alternative_better_threshold:
            # If any of the alternatives is better than the base case (to which we arrived using a gradient-based
            #  approach), we stop the optimization procedure, keep the current controller parameters, and switch
            #  to the Bezier modification that corresnponds to that alternative
            self.curr_bezier_points_delta = self.alternative_bezier_point_deltas[best_alternative].copy()

        else:
            # But if no alternative was better, we continue out gradient-based procedure
            offset = 1 + self.num_alternatives

            # 3) Finally, we need to compute the gradient of self.curr_bezier_points_delta
            num_bezier_coords = POINT_DIM * self.num_bezier_points
            grad_bezier_points_delta = np.zeros(num_bezier_coords)
            x = self.eta_for_grad * np.arange(-self.num_steps_for_grad, self.num_steps_for_grad + 1)
            for bezier_idx in range(num_bezier_coords):
                reward = []
                for step in range(2 * self.num_steps_for_grad):
                    idx = offset + bezier_idx + step * num_bezier_coords
                    reward.append(float(reward_pred[idx]))
                    if step == self.num_steps_for_grad:
                        reward.append(base_reward)    

                slope = np.polyfit(x, reward, deg=1)[0]  # Fit a line and take the slope
                grad_bezier_points_delta[bezier_idx] = np.clip(slope, -self.grad_clip_threshold, self.grad_clip_threshold)

            self.curr_bezier_points_delta += (self.eta_for_update * grad_bezier_points_delta).reshape(-1, 2)
            # self.curr_bezier_points_delta[0] = 0, 0  # TODO
            self.curr_bezier_points_delta[:, 1] = np.clip(self.curr_bezier_points_delta[:, 1], -0.4, 0.4)

            offset += self.bezier_points_step_shape

            # 2) OK, time for gradient estimation for contr_params
            # NOTE: the following code is not the fastest implementation of the gradient estimation but notice that
            #  the call above (to calc_progress_and_penalty_while_driving) is the real bottleneck. And if no GPU is available you won't
            #  get a significant speedup by optimizing the gradient estimating below
            grad_contr_param = np.zeros(self.num_contr_params)
            x = self.eta_for_grad * np.arange(-self.num_steps_for_grad, self.num_steps_for_grad + 1)
            for contr_param_idx in range(self.num_contr_params):
                reward = []
                for step in range(2 * self.num_steps_for_grad):
                    idx = offset + contr_param_idx + step * self.num_contr_params
                    reward.append(float(reward_pred[idx]))
                    if step == self.num_steps_for_grad:
                        reward.append(base_reward)
                        
                slope = np.polyfit(x, reward, deg=1)[0]  # Fit a line and take the slope
                grad_contr_param[contr_param_idx] = np.clip(slope, -self.grad_clip_threshold, self.grad_clip_threshold)

            self.curr_contr_params = self.features_scalers['contr_params'].inverse_transform(
                curr_contr_params_scaled
                + self.eta_for_update * torch.tensor(grad_contr_param, device=self.device)
            )
            # Clip the controller parameters according to their limits
            self.curr_contr_params = torch.clip(self.curr_contr_params, min=self.contr_params_limits[:, 0], max=self.contr_params_limits[:, 1])

        if self.debug:
            print(f'lookahead = {self.curr_contr_params[0]:.2f}, '
                  f'speed_setpoint = {self.curr_contr_params[1]:.2f}, '
                  f'tire_force_max = {self.curr_contr_params[2]:.2f}')

        if self.ego_data is not None:
            self._gather_data(
                yaw, position, linear_vel_x, linear_vel_y, angular_vel_z,
                self.curr_delta, self.curr_speed, lap_time,
                bezier_points_delta=self.curr_bezier_points_delta.copy(),
                centerline=centerline.cpu().numpy(),
                left_bound=left_bound.cpu().numpy(),
                right_bound=right_bound.cpu().numpy(),
                ranges=ranges_as_vec.cpu().numpy(),
                trajectory_pred=trajectory_pred[NOT_MODIFIED_IDX].cpu().numpy(),
                actuators_pred=actuators_pred[NOT_MODIFIED_IDX].cpu().numpy(),
                collision=kwargs['collision'],
            )

        return self.curr_speed, self.curr_delta

    def extract_centerline_and_bounds(
            self,
            yaw: float,
            position: np.array,
    ) -> Tuple[np.array, np.array, np.array]:
        """Extract centerline and bound slices used for prediction and reward calculations."""
        points_slices = []
        for points, num_steps in zip(self.points, self.nums_steps):
            closest_idx = closest_point_idx(position, points)
            points_slice = cyclic_slice(points, closest_idx, num_steps)
            points_slice = rotate_into_map_coord(points_slice - position, -yaw)
            points_slice = torch.tensor(points_slice, device=self.device, dtype=torch.float)
            points_slices.append(points_slice)
        centerline, left_bound, right_bound = points_slices
        return centerline, left_bound, right_bound

    def calc_predictions(
            self,
            centerline: np.array,
            linear_vel_x: float,
            linear_vel_y: float,
            angular_vel_z: float,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Calculates the predicted trajectory and actuators, as well as current controller params (scaled and tiled)."""

        state = torch.tensor(np.r_[linear_vel_x, linear_vel_y, angular_vel_z, self.curr_delta, self.curr_speed], device=self.device, dtype=torch.float)
        contr_params = self.curr_contr_params.clone().detach().float()

        state = torch.tile(state, (self.batch_size, 1))
        state_scaled = self.features_scalers['state'].transform(state)

        curr_waypoints, waypoints_modified = self._bezierize_and_modify_waypoints(centerline)
        curr_waypoints_tiled = torch.tile(curr_waypoints, (self.contr_params_step_shape, 1))
        waypoints = torch.concat([curr_waypoints[None], waypoints_modified, curr_waypoints_tiled])
        waypoints_scaled = self.features_scalers['waypoints'].transform(waypoints.to(self.device))

        curr_contr_params_scaled = self.features_scalers['contr_params'].transform(contr_params)
        contr_params_modified = curr_contr_params_scaled + self.eta_for_grad * self.contr_params_step_directions

        # OK, this is the final batch
        contr_params = torch.concat([
            curr_contr_params_scaled[None],
            torch.tile(curr_contr_params_scaled, (len(waypoints_modified), 1)),
            contr_params_modified,
        ])

        with torch.inference_mode():
            preds = self.omniward_model(state_scaled, contr_params, waypoints_scaled, None, None, None)
            trajectory_pred, actuators_pred = preds['trajectory_pred'], preds['actuators_pred']

        return trajectory_pred, actuators_pred, contr_params[NOT_MODIFIED_IDX]

    def _compose_additional_data(self) -> Dict[str, Any]:
        return {
            'lookahead_distance': float(self.curr_contr_params[0]),
            'speed_setpoint': float(self.curr_contr_params[1]),
            'tire_force_max': float(self.curr_contr_params[2]),
            'centerline': self.points[0],
            'left_bound': self.points[1],
            'right_bound': self.points[2],
        }

    def to(self, device):
        self.device = device
        self.contr_params_step_directions = self.contr_params_step_directions.to(self.device)
        self.curr_contr_params = self.curr_contr_params.to(self.device)
        self.omniward_model = self.omniward_model.to(self.device)
        self.contr_params_limits = self.contr_params_limits.to(self.device)

        for features_scaler in self.features_scalers.values():
            features_scaler.to(self.device)

        for targets_scaler in self.targets_scalers.values():
            targets_scaler.to(self.device)

    def _bezierize_and_modify_waypoints(self, centerline: torch.Tensor) -> Tuple[np.array, np.array]:
        centerline_cpu = centerline.cpu().numpy()  # TODO: ugh, can be done better
        centerline_cpu, translation, yaw = straighten_up_arc(centerline_cpu)

        # We want the unmodified curr_waypoints to be of shape:
        #   (POINT_DIM * len(centerline), )
        #  to then tile them
        curr_waypoints = self._modify_waypoints(self.curr_bezier_points_delta, centerline_cpu, translation, yaw)
        curr_waypoints = curr_waypoints.flatten()
        
        waypoints_modified = []
        # The first type of modifications comes from the alternative Bezier points
        for bezier_points in self.alternative_bezier_point_deltas:
            mod_waypoints = self._modify_waypoints(bezier_points, centerline_cpu, translation, yaw)
            waypoints_modified.append(mod_waypoints)

        # The second type of modifications comes from the step directions and will be used for estimating the gradient
        bezier_points_modified = self.curr_bezier_points_delta + self.eta_for_grad * self.bezier_points_step_directions
        bezier_points_modified = bezier_points_modified.reshape(-1, self.num_bezier_points, POINT_DIM)
        for bezier_points in bezier_points_modified:
            mod_waypoints = self._modify_waypoints(bezier_points, centerline_cpu, translation, yaw)
            waypoints_modified.append(mod_waypoints)
            
        # Stack the modified waypoints
        # TODO: you could allocate the result and avoid this copying, even in the __init__
        waypoints_modified = np.r_[waypoints_modified]

        # We want the modified waypoints be of the shape:
        #   (self.bezier_points_step_shape, POINT_DIM * len(centerline))
        #  to then concat it togher with the contr_params part of the batch
        waypoints_modified = waypoints_modified.reshape(waypoints_modified.shape[0], -1)
        
        return torch.tensor(curr_waypoints, dtype=torch.float32), torch.tensor(waypoints_modified, dtype=torch.float32)

    @staticmethod
    def _modify_waypoints(bezier_points: np.array, centerline: np.array, translation: np.array, yaw: float) -> np.array:
        bcurve = bezier_curve(bezier_points, num_points=len(centerline))
        mod_waypoints = np.c_[
            centerline[:, 0],
            centerline[:, 1] + bcurve[:, 1]
        ]
        mod_waypoints = rotate_into_map_coord(mod_waypoints, yaw) + translation
        return mod_waypoints
