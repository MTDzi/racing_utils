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

            # Input
            state_size: int,
            controller_params_size: int,
            centerline_size: int,

            # Middle part of  the architecture
            centerline_encoder_sizes: Sequence[int],
            middle_sizes: Sequence[int],
            output_sizes: Sequence[int],

            # Output
            trajectory_size: int,
            actuators_size: int,
    ):
        super().__init__()

        sizes = [centerline_size] + centerline_encoder_sizes
        self.centerline_encoder = self._stack_layers(sizes, nn.Linear, nn.SiLU)

        sizes = [state_size + centerline_encoder_sizes[-1] + controller_params_size] + middle_sizes
        self.middle_encoder = self._stack_layers(sizes, nn.Linear, nn.SiLU)

        sizes = [middle_sizes[-1]] + output_sizes + [actuators_size]
        self.actuators_predictor = self._stack_layers(sizes, nn.Linear, nn.SiLU, last_linear=True)

        sizes = [middle_sizes[-1]] + output_sizes + [trajectory_size]
        self.trajectory_predictor = self._stack_layers(sizes, nn.Linear, nn.SiLU, last_linear=True)

    @staticmethod
    def _stack_layers(
            sizes: Sequence,
            layer: nn.Module,
            activation: nn.Module,
            last_linear: bool =False,
    ):
        layers = []
        for enc_size_in, enc_size_out in zip(sizes[:-1], sizes[1:]):
            layers.append(layer(enc_size_in, enc_size_out))
            layers.append(activation(inplace=True))
        if last_linear:
            layers.pop()
        return nn.Sequential(*layers)
        
    def forward(
            self,
            state: torch.Tensor,
            contr_params: torch.Tensor,
            centerline: torch.Tensor,
            left_bound: torch.Tensor,
            right_bound: torch.Tensor,
    ):
        centerline_enc = self.centerline_encoder(centerline)

        centerline_for_middle = centerline_enc.clone()
        middle = self.middle_encoder(torch.cat([state, contr_params, centerline_for_middle], axis=1))
        middle_for_trajectory = middle.clone()
        middle_for_actuators = middle.clone()
        trajectory_pred = self.trajectory_predictor(middle_for_trajectory)
        actuators_pred = self.actuators_predictor(middle_for_actuators)

        return {
            'trajectory_pred': trajectory_pred,
            'actuators_pred': actuators_pred,
        }


def get_omniward_model(
        num_layers: int,
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
    centerline_size = len(features['centerline'])
    trajectory_size = len(targets['trajectory'])
    actuators_size = len(targets['speeds_and_deltas'])

    centerline_encoder_sizes = num_layers * [int(centerline_size // width_reduction // width_reduction)]
    middle_sizes = num_layers * [int((centerline_size + trajectory_size + actuators_size) // width_reduction // width_reduction)]
    output_sizes = [
        int(centerline_size // width_reduction),
        int((trajectory_size + actuators_size) // width_reduction // width_reduction),
        int((trajectory_size + actuators_size) // width_reduction),
    ]

    omniward_model = OmniwardModel(
        state_size,
        contr_params_size,
        centerline_size,
        centerline_encoder_sizes,
        middle_sizes,
        output_sizes,
        trajectory_size,
        actuators_size,
    )
    omniward_model.to(device)

    return omniward_model
