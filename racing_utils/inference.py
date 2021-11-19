from typing import Dict

import numpy as np

import torch

from .torch_related import TensorStandardScaler, calc_reward_and_penalty
from .utils import rotate_into_map_coord, closest_point_idx, cyclic_slice


NOT_MODIFIED_IDX = 0


class GradientDriver:
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

            # Gradient-related
            eta: float = 0.1,
            num_steps_for_grad: int = 4,
            penalty_sigma: float = 0.4,

            device: str = 'cpu',
    ):
        self.points = [centerline, left_bound, right_bound]
        self.nums_steps = [num_steps_centerline, num_steps_ahead_bound, num_steps_ahead_bound]

        self.omniward_model = omniward_model
        self.features_scalers = features_scalers
        self.targets_scalers = targets_scalers

        self.eta = eta
        self.num_steps_for_grad = num_steps_for_grad
        self.penalty_sigma = penalty_sigma

        self.curr_delta = 0.0
        self.curr_speed = 0.0
        self.curr_contr_params = init_contr_params

        self.device = device

        self.num_contr_params = len(init_contr_params)
        self.step_directions = torch.cat(
                [torch.zeros((1, self.num_contr_params))]
                + [-step * torch.eye(self.num_contr_params) for step in reversed(range(1, num_steps_for_grad+1))]
                + [ step * torch.eye(self.num_contr_params) for step in range(1, num_steps_for_grad+1)],
            axis=0
        ).to(self.device)
        self.batch_size = self.step_directions.shape[0]


    def plan(self, ranges, yaw, pos_x, pos_y, linear_vel_x, linear_vel_y, angular_vel_z):
        position = np.array([pos_x, pos_y])
        points_slices = []
        for points, num_steps in zip(self.points, self.nums_steps):
            closest_idx = closest_point_idx(position, points)
            points_slice = cyclic_slice(points, closest_idx, num_steps)
            points_slice = rotate_into_map_coord(points_slice - position, -yaw)
            points_slice = torch.tensor(points_slice, device=self.device, dtype=torch.float)
            points_slices.append(points_slice)
        centerline, left_bound, right_bound = points_slices

        state = torch.tensor(np.r_[linear_vel_x, linear_vel_y, angular_vel_z, self.curr_delta, self.curr_speed], device=self.device, dtype=torch.float)
        contr_params = torch.tensor(self.curr_contr_params, device=self.device, dtype=torch.float)

        batch_shape = (self.batch_size, 1)
        state = torch.tile(state, batch_shape)
        contr_params = torch.tile(contr_params, batch_shape)
        centerline = torch.tile(centerline.flatten(), batch_shape)
        left_bound = torch.tile(left_bound, batch_shape)
        right_bound = torch.tile(right_bound, batch_shape)

        state_scaled = self.features_scalers['state'].transform(state)
        contr_params_scaled = self.features_scalers['contr_params'].transform(contr_params)
        centerline_scaled = self.features_scalers['centerline'].transform(centerline)

        new_contr_params = contr_params_scaled + self.eta * self.step_directions

        with torch.inference_mode():
            preds = self.omniward_model(state_scaled, new_contr_params, centerline_scaled, None, None)
            trajectory_pred, actuators_pred = preds['trajectory_pred'], preds['actuators_pred']

        # First, let's extract the actuator values needed for driving the car
        speeds_and_deltas = self.targets_scalers['speeds_and_deltas'].inverse_transform(actuators_pred[NOT_MODIFIED_IDX]).cpu()
        first_speed_actuator_idx = 0
        first_delta_actuator_idx = len(speeds_and_deltas) // 2
        self.curr_speed = float(speeds_and_deltas[first_speed_actuator_idx])
        self.curr_delta = float(speeds_and_deltas[first_delta_actuator_idx])

        # return self.curr_speed, self.curr_delta


        # TODO: make these attributes
        self.penalty_scale_coeff = -1.1
        self.penalty_sigma = 0.2
        self.contr_params_limits = torch.tensor([
            (3.0, 5.0),
            (8.0, 14.0),
            (8.0, 12.0),
        ], device=self.device)
        # self.contr_params_limits = torch.tensor([
        #     (3.0, 7.0),
        #     (8.0, 16.0),
        #     (8.0, 15.0),
        # ], device=self.device)
        self.contr_params_limits = torch.tensor([
            (3.0, 6.0),
            (8.0, 15.0),
            (8.0, 12.0),
        ], device=self.device)

        
        # Now for the heavier part: calculating the gradient of the reward and penalty

        # 1) To estimate the gradient of the reward we need the trajectory
        trajectory_pred = self.targets_scalers['trajectory'].inverse_transform(trajectory_pred)
        reward_pred, penalty_pred = calc_reward_and_penalty(trajectory_pred, centerline, left_bound, right_bound, penalty_sigma=self.penalty_sigma)

        # 2) OK, time for gradient estimation
        # NOTE: the following code is not the fastest implementation of the gradient calculation but notice that
        #  the call above (to calc_reward_and_penalty) is the real bottleneck. And if not GPU is available you won't
        #  get a significant speedup by optimizing the gradient estimating code below
        base_reward = float(reward_pred[NOT_MODIFIED_IDX].cpu())
        base_penalty = float(penalty_pred[NOT_MODIFIED_IDX].cpu())
        grad_contr_param = np.zeros(self.num_contr_params)
        x = self.eta * np.arange(-self.num_steps_for_grad, self.num_steps_for_grad + 1)
        for contr_param_idx in range(self.num_contr_params):
            objective = []
            for step in range(2 * self.num_steps_for_grad):
                common_idx = 1 + step * self.num_contr_params + contr_param_idx
                reward = float(reward_pred[common_idx].cpu())
                penalty = float(penalty_pred[common_idx].cpu())
                objective.append(reward + self.penalty_scale_coeff * penalty)
                if step == self.num_steps_for_grad:
                    objective.append(base_reward + self.penalty_scale_coeff * base_penalty)
                    
            coeffs = np.polyfit(x, objective, deg=1)  # Fit a line
            grad_contr_param[contr_param_idx] = coeffs[0]  # This is the slope of the fitted line

        self.curr_contr_params = self.features_scalers['contr_params'].inverse_transform(
            new_contr_params[NOT_MODIFIED_IDX]
            + self.eta * torch.tensor(grad_contr_param, device=self.device)
        )
        # Clip the controller parameters according to their limits
        self.curr_contr_params = torch.clip(self.curr_contr_params, min=self.contr_params_limits[:, 0], max=self.contr_params_limits[:, 1])

        print(f'lookahead = {self.curr_contr_params[0]:.2f}, '
              f'speed_setpoint = {self.curr_contr_params[1]:.2f}, '
              f'tire_force_max = {self.curr_contr_params[2]:.2f}')

        return self.curr_speed, self.curr_delta

        

