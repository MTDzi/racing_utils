from typing import Dict, Sequence, Tuple

import numpy as np

import torch
import torch.nn as nn


class OmniwardModel(nn.Module):
    """
    Model predicting the future:
     * positions and
     * actuators given the current:
    given the current:
     * state of the car
     * centerline and, most importantly,
     * controller parameter values.
    """

    FULL_TRAJ_DIM = 6
    YAW_0 = torch.tensor([0.0])

    def __init__(
            self,
            middle_sizes: Sequence[int],
            actuators_head_sizes: Sequence[int],
            trajectory_head_sizes: Sequence[int],
    ):
        super().__init__()
        self.middle_encoder = self._stack_layers(middle_sizes, nn.Linear, nn.SiLU)
        self.actuators_predictor = self._stack_layers(actuators_head_sizes, nn.Linear, nn.SiLU, last_linear=True)
        self.trajectory_predictor = self._stack_layers(trajectory_head_sizes, nn.Linear, nn.SiLU, last_linear=True)

    @staticmethod
    def _stack_layers(
            sizes: Sequence,
            layer: nn.Module,
            activation: nn.Module,
            last_linear: bool =False,
    ):
        layers = []
        for enc_size_in, enc_size_out in zip(sizes[:-1], sizes[1:]):
            layers.append(layer(enc_size_in, int(enc_size_out)))
            layers.append(activation(inplace=True))
        if last_linear:
            layers.pop()
        return nn.Sequential(*layers)
        
    def forward(
            self,
            state: torch.Tensor,
            contr_params: torch.Tensor,
            waypoints: torch.Tensor,
            centerline: torch.Tensor,
            left_bound: torch.Tensor,
            right_bound: torch.Tensor,
    ):
        middle = self.middle_encoder(torch.cat([state, contr_params, waypoints], axis=1))
        trajectory_pred = self.trajectory_predictor(middle.clone())
        actuators_pred = self.actuators_predictor(middle.clone())

        return {
            'trajectory_pred': trajectory_pred,
            'actuators_pred': actuators_pred,
        }


def get_omniward_model(
        num_layers_waypoint_encoder: int,
        num_layers_middle: int,
        width_reduction: float,
        features: Dict[str, np.array],
        targets: Dict[str, np.array],
        device: str,
) -> OmniwardModel:
    """
    Determines the architecture of the model based on the feature and target sizes, creates the OmniwardModel,
    pushes it to the specified device, and returns it.
    """
    state_size = len(features['state'])
    contr_params_size = len(features['contr_params'])
    waypoints_size = len(features['waypoints_bezier'])
    trajectory_size = len(targets['trajectory'])
    actuators_size = len(targets['speeds_and_deltas'])

    waypoints_encoder_sizes = [waypoints_size]
    for _ in range(num_layers_waypoint_encoder):
        waypoints_encoder_sizes.append(int(waypoints_encoder_sizes[-1] // width_reduction))

    middle_sizes = [waypoints_encoder_sizes[-1]]
    for _ in range(num_layers_middle):
        middle_sizes.append(int(middle_sizes[-1] // width_reduction))

    output_sizes = [
        middle_sizes[-1],
        int((trajectory_size + actuators_size) // width_reduction),
    ]

    omniward_model = OmniwardModel(
        state_size,
        contr_params_size,
        waypoints_encoder_sizes,
        middle_sizes,
        output_sizes,
        trajectory_size,
        actuators_size,
    )
    omniward_model.to(device)

    return omniward_model
