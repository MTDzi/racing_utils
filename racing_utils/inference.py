from typing import Dict, Optional, Sequence

import numpy as np

import torch

from .torch_related import TensorStandardScaler, calc_progress_and_penalty_while_driving
from .utils import rotate_into_map_coord, closest_point_idx, cyclic_slice


NOT_MODIFIED_IDX = 0


class GradientDriver:
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
            the calc_progress_and_penalty_while_driving function).
        contr_params_limits: When doing gradient descent on the controller parameters, the new values will be clipped
            by the values specified in this tensor.
    """

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
        self.step_directions = torch.cat(
                [torch.zeros((1, self.num_contr_params))]
                + [-step * torch.eye(self.num_contr_params) for step in reversed(range(1, num_steps_for_grad + 1))]
                + [ step * torch.eye(self.num_contr_params) for step in range(1, num_steps_for_grad + 1)],
            axis=0
        ).to(self.device)
        self.batch_size = self.step_directions.shape[0]

        assert penalty_scale_coeff < 0, (
            'The penalty_scale_coeff passed was positive which makes no sense given that the objective function is:\n'
            '\t progress + penalty_scale_coeff * penalty\n'
        )
        self.penalty_scale_coeff = penalty_scale_coeff
        self.penalty_sigma = penalty_sigma
        self.contr_params_limits = contr_params_limits.to(self.device)
        self.only_closest_for_penalty = only_closest_for_penalty

        self.debug = debug


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
    ):
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
        contr_params = self.curr_contr_params.clone().detach().float()

        batch_shape = (self.batch_size, 1)
        state = torch.tile(state, batch_shape)
        contr_params = torch.tile(contr_params, batch_shape)
        centerline = torch.tile(centerline.flatten(), batch_shape)
        left_bound = torch.tile(left_bound, batch_shape)
        right_bound = torch.tile(right_bound, batch_shape)

        state_scaled = self.features_scalers['state'].transform(state)
        contr_params_scaled = self.features_scalers['contr_params'].transform(contr_params)
        centerline_scaled = self.features_scalers['centerline'].transform(centerline)

        new_contr_params = contr_params_scaled + self.eta_for_grad * self.step_directions

        with torch.inference_mode():
            preds = self.omniward_model(state_scaled, new_contr_params, centerline_scaled, None, None)
            trajectory_pred, actuators_pred = preds['trajectory_pred'], preds['actuators_pred']

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
            penalty_sigma=self.penalty_sigma,
            only_closest=self.only_closest_for_penalty,
        )

        # 2) OK, time for gradient estimation
        # NOTE: the following code is not the fastest implementation of the gradient estimation but notice that
        #  the call above (to calc_progress_and_penalty_while_driving) is the real bottleneck. And if no GPU is available you won't
        #  get a significant speedup by optimizing the gradient estimating below
        base_progress = float(progress_pred[NOT_MODIFIED_IDX].cpu())
        base_penalty = float(penalty_pred[NOT_MODIFIED_IDX].cpu())
        grad_contr_param = np.zeros(self.num_contr_params)
        x = self.eta_for_grad * np.arange(-self.num_steps_for_grad, self.num_steps_for_grad + 1)
        for contr_param_idx in range(self.num_contr_params):
            reward = []
            for step in range(2 * self.num_steps_for_grad):
                common_idx = 1 + step * self.num_contr_params + contr_param_idx
                progress = float(progress_pred[common_idx].cpu())
                penalty = float(penalty_pred[common_idx].cpu())
                reward.append(progress + self.penalty_scale_coeff * penalty)
                if step == self.num_steps_for_grad:
                    reward.append(base_progress + self.penalty_scale_coeff * base_penalty)
                    
            coeffs = np.polyfit(x, reward, deg=1)  # Fit a line
            grad_contr_param[contr_param_idx] = coeffs[0]  # This is the slope of the fitted line

        self.curr_contr_params = self.features_scalers['contr_params'].inverse_transform(
            new_contr_params[NOT_MODIFIED_IDX]
            + self.eta_for_update * torch.tensor(grad_contr_param, device=self.device)
        )
        # Clip the controller parameters according to their limits
        self.curr_contr_params = torch.clip(self.curr_contr_params, min=self.contr_params_limits[:, 0], max=self.contr_params_limits[:, 1])

        if self.debug:
            print(f'lookahead = {self.curr_contr_params[0]:.2f}, '
                  f'speed_setpoint = {self.curr_contr_params[1]:.2f}, '
                  f'tire_force_max = {self.curr_contr_params[2]:.2f}')

        return self.curr_speed, self.curr_delta

    def to(self, device):
        self.device = device
        self.step_directions = self.step_directions.to(self.device)
        self.curr_contr_params = self.curr_contr_params.to(self.device)
        self.omniward_model = self.omniward_model.to(self.device)
        self.contr_params_limits = self.contr_params_limits.to(self.device)

        for features_scaler in self.features_scalers.values():
            features_scaler.to(self.device)

        for targets_scaler in self.targets_scalers.values():
            targets_scaler.to(self.device)
