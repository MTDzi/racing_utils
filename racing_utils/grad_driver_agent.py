from typing import Dict, Optional, Sequence, Tuple, Dict, Any

import numpy as np
import torch

from .torch_related import (
    TensorStandardScaler,
    calc_progress_and_penalty_while_driving,
    modify_waypoints_in_batch,
    modify_waypoints,
    straighten_up_arc,
)
from .base_agent import BaseAgent


NOT_MODIFIED_IDX = 0
POINT_DIM = 2
IDX = 0


class GradientDriverAgent(BaseAgent):
    """Controls the car using a trained model + modifying part of its input so as to maximize reward.
    Runs inference for the model and modifies the controller parameters (that are part of the input of the model)
    according to an approximated gradient of the (progress - penalty) objective function.
    Attributes:
        points: A list containing the arrays: waypoints, left_bound, and right_bound.
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


    def setup_cache(self):
        # Cache the following attributes instead of re-calculating them every time
        self.idx_to_angle = np.array([
            self.ANGLE_MIN + idx * self.ANGLE_INCR
            for idx in range(self.SCAN_LEN)
        ])
        self.versor = np.array([
            np.array([np.cos(self.idx_to_angle[idx]), np.sin(self.idx_to_angle[idx])])
            for idx in range(self.SCAN_LEN)
        ])
        

    def set_bezier_degree(self, bezier_degree: int):
        self.bezier_degree = bezier_degree

        self.num_modifiable_bezier_points = self.bezier_degree - 1  # Meaning: the terminal points will NOT be modified
        self.bezier_points_step_directions = np.concatenate([
            [-step * np.eye(self.num_modifiable_bezier_points) for step in reversed(range(1, self.num_steps_for_grad + 1))]
            + [ step * np.eye(self.num_modifiable_bezier_points) for step in range(1, self.num_steps_for_grad + 1)]
        ], axis=0)
        self.bezier_points_step_directions = self.bezier_points_step_directions.reshape(
            (2 * self.num_steps_for_grad) * self.num_modifiable_bezier_points,
            self.num_modifiable_bezier_points,
        )
        self.bezier_points_step_directions = np.column_stack([
            np.zeros(self.bezier_points_step_directions.shape[0]),
            self.bezier_points_step_directions,
            np.zeros(self.bezier_points_step_directions.shape[0]),
        ])
        self.bezier_points_step_directions = np.stack([
            np.zeros_like(self.bezier_points_step_directions),
            self.bezier_points_step_directions
        ], axis=2)

        self.bezier_points_step_directions = self.eta_for_waypoints_grad * torch.tensor(self.bezier_points_step_directions, device=self.device)
        self.batch_size = 1 + self.contr_params_step_shape + self.bezier_points_step_directions.shape[0]
    

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




        self.grad_clip_threshold = 1.0

        



        position = np.array([pos_x, pos_y])
        ranges_as_vec = torch.tensor(ranges[:, np.newaxis] * self.versor, requires_grad=False, device=self.device)

        waypoints, centerline, left_bound, right_bound, best_score = self.random_probe_planner.plan(position, yaw, ranges_as_vec)

        # This is where the prediction takes place
        trajectory_pred, actuators_pred, curr_contr_params_scaled, curr_waypoints_straight, straighting_translation, straighting_yaw = self.calc_predictions(
            waypoints,
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
        )

        reward_pred = (progress_pred + self.penalty_scale_coeff / penalty_pred).cpu()
        base_reward = float(reward_pred[NOT_MODIFIED_IDX])

        # But if no alternative was better, we continue out gradient-based procedure
        offset = 1

        # 3) Finally, we need to compute the gradient of self.curr_bezier_points_delta
        grad_bezier_points_delta = np.zeros(self.num_modifiable_bezier_points)
        x = self.eta_for_grad * np.arange(-self.num_steps_for_grad, self.num_steps_for_grad + 1)
        for bezier_idx in range(self.num_modifiable_bezier_points):
            reward = []
            for step in range(2 * self.num_steps_for_grad):
                idx = offset + bezier_idx + step * self.num_modifiable_bezier_points
                reward.append(float(reward_pred[idx]))
                if step == self.num_steps_for_grad:
                    reward.append(base_reward)    

            slope = np.polyfit(x, reward, deg=1)[0]  # Fit a line and take the slope
            grad_bezier_points_delta[bezier_idx] = np.clip(slope, -self.grad_clip_threshold, self.grad_clip_threshold)

        bezier_modification = np.r_[0, self.eta_for_waypoints_update * grad_bezier_points_delta, 0]
        bezier_modification = np.c_[np.zeros_like(bezier_modification), bezier_modification]
        bezier_modification = torch.tensor(bezier_modification, device=self.device)
        waypoints_modified = modify_waypoints(bezier_modification, curr_waypoints_straight, straighting_translation, straighting_yaw)
        waypoints[:self.modification_steps] = waypoints_modified
        
        offset += self.bezier_points_step_directions.shape[0]

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


        
        self.random_probe_planner.update_waypoints(waypoints.cpu().numpy(), position, yaw)


        if self.ego_data is not None:
            self._gather_data(
                yaw, position, linear_vel_x, linear_vel_y, angular_vel_z,
                self.curr_delta, self.curr_speed, lap_time,
                bezier_points_delta=self.curr_bezier_points_delta.copy(),
                waypoints=waypoints.cpu().numpy(),
                left_bound=left_bound.cpu().numpy(),
                right_bound=right_bound.cpu().numpy(),
                ranges=ranges_as_vec.cpu().numpy(),
                trajectory_pred=trajectory_pred[NOT_MODIFIED_IDX].cpu().numpy(),
                actuators_pred=actuators_pred[NOT_MODIFIED_IDX].cpu().numpy(),
                collision=kwargs['collision'],
            )

        return self.curr_speed, self.curr_delta


    def calc_predictions(
            self,
            curr_waypoints: torch.Tensor,
            linear_vel_x: float,
            linear_vel_y: float,
            angular_vel_z: float,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Calculates the predicted trajectory and actuators, as well as current controller params (scaled and tiled)."""

        state = torch.tensor(np.r_[linear_vel_x, linear_vel_y, angular_vel_z, self.curr_delta, self.curr_speed], device=self.device, dtype=torch.float)
        contr_params = self.curr_contr_params.clone().detach().float()

        state = torch.tile(state, (self.batch_size, 1))
        state_scaled = self.features_scalers['state'].transform(state)

        curr_waypoints_straight, translation, yaw = straighten_up_arc(curr_waypoints[:self.modification_steps].clone())
        waypoints_modified = modify_waypoints_in_batch(self.bezier_points_step_directions, curr_waypoints_straight, translation, yaw)
        waypoints_unmodified_tiled = torch.tile(curr_waypoints[self.modification_steps:], (waypoints_modified.shape[0], 1, 1))
        waypoints_modified = torch.concat([waypoints_modified, waypoints_unmodified_tiled], axis=1)

        curr_waypoints_tiled = torch.tile(curr_waypoints, (self.contr_params_step_shape, 1, 1))
        waypoints = torch.concat([
            curr_waypoints.reshape(1, -1),
            waypoints_modified.reshape(waypoints_modified.shape[0], -1),
            curr_waypoints_tiled.reshape(curr_waypoints_tiled.shape[0], -1),
        ])
        # TODO: The scaling could be done prior to the concat, might be faster 
        waypoints_scaled = self.features_scalers['waypoints'].transform(waypoints)

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

        return trajectory_pred, actuators_pred, contr_params[NOT_MODIFIED_IDX], curr_waypoints_straight, translation, yaw


    def _compose_additional_data(self) -> Dict[str, Any]:
        return {
            'lookahead_distance': float(self.curr_contr_params[0]),
            'speed_setpoint': float(self.curr_contr_params[1]),
            'tire_force_max': float(self.curr_contr_params[2]),
            'waypoints': self.points[0],
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
