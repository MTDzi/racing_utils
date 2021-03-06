"""
Classes and functions that use PyTorch but are neither models nor inference utensils.
"""
from __future__ import annotations

from typing import Dict, Tuple, NewType

import torch

from sklearn.preprocessing import StandardScaler


Batch = NewType('Batch', Dict[str, torch.Tensor])


class NotTensorfiedYetException(Exception):
    pass


class TensorStandardScaler(StandardScaler):
    """
    Like the sklearn version, this class transforms features by subtracting the mean, and
     dividing by the standard deviation.

    The crucial difference is that this scaler can operate on tensors, so the computation can
     be carried out on  the GPU.
    """

    def __init__(self, device: str, **kwargs):
        super().__init__(**kwargs)
        self.device = device
        self.tensorfied = False

    def tensorfy(self):
        self.mean_ = torch.tensor(self.mean_, device=self.device, dtype=torch.float)
        self.scale_ = torch.tensor(self.scale_, device=self.device, dtype=torch.float)
        self.tensorfied = True

    def transform(self, X: torch.Tensor, copy: bool = True) -> torch.Tensor:
        if not self.tensorfied:
            raise NotTensorfiedYetException('You need to call `tensorfy` first')

        if copy is False:
            X -= self.mean_
            X /= self.scale_
        else:
            X = (X - self.mean_) / self.scale_

        return X

    def inverse_transform(self, X: torch.Tensor, copy: bool = True) -> torch.Tensor:
        if not self.tensorfied:
            raise NotTensorfiedYetException('You need to call `tensorfy` first')
            
        if copy is False:
            X *= self.scale_
            X += self.mean_
        else:
            X = self.scale_ * X + self.mean_

        return X

    def to(self, device: str) -> TensorStandardScaler:
        if not self.tensorfied:
            raise NotTensorfiedYetException('You need to call `tensorfy` first')
            
        self.device = device
        self.mean_ = self.mean_.to(device)
        self.scale_ = self.scale_.to(device)

        return self


def scale_batch_and_to_device(
        device: str,
        features_scalers: Dict[str, TensorStandardScaler],
        targets_scalers: Dict[str, TensorStandardScaler],
        features_batch: Batch,
        targets_batch: Batch,
) -> Tuple[Batch, Batch]:
    """
    Calls the .transform method of the scalers on the features and targets, and then moves
     them to the device specified.
    """
    features_batch = {
        feature_name: features_scalers[feature_name].transform(features_batch[feature_name].float().to(device))
        for feature_name in features_batch.keys()
    }
    targets_batch = {
        target_name: targets_scalers[target_name].transform(targets_batch[target_name].float().to(device))
        for target_name in targets_batch.keys()
    }
    return features_batch, targets_batch


@torch.inference_mode()
def calc_progress_and_penalty(
        trajectory: torch.Tensor,
        centerline: torch.Tensor,
        left_bound: torch.Tensor,
        right_bound: torch.Tensor,
        penalty_sigma: float = 0.4,
        only_closest: bool = False,
):
    """
    Calculates the progress along the centerline + penalty caused by closeness to any of the bounds.
    """
    batch_size = len(trajectory)
    trajectory = trajectory.reshape(batch_size, -1, 2)
    centerline = centerline.reshape(batch_size, -1, 2)
    left_bound = left_bound.reshape(batch_size, -1, 2)
    right_bound = right_bound.reshape(batch_size, -1, 2)

    last_position = trajectory[:, -1]
    dists = torch.linalg.norm(centerline - last_position[:, None], axis=2)
    closest_centerline_idx = torch.argmin(dists, axis=1)
    dists_along_centerline = torch.linalg.norm(torch.diff(centerline, axis=1), axis=2).cumsum(axis=1)
    arange = torch.arange(batch_size)
    closest_centerline_idx_for_diff = closest_centerline_idx - 1
    closest_centerline_idx_for_diff[closest_centerline_idx_for_diff == -1] = 0
    reward = dists_along_centerline[arange, closest_centerline_idx_for_diff]

    versor_to_last_position = last_position - centerline[arange, closest_centerline_idx]
    versor_to_last_position /= torch.linalg.norm(versor_to_last_position) + 1e-10

    closest_centerline_idx_for_last_centerline_vector = closest_centerline_idx + 1
    centerline_len = centerline.shape[1]
    closest_centerline_idx_for_last_centerline_vector[closest_centerline_idx_for_last_centerline_vector == centerline_len] = centerline_len - 1
    last_centerline_vector = centerline[arange, closest_centerline_idx_for_last_centerline_vector] - centerline[arange, closest_centerline_idx_for_last_centerline_vector - 1]
    
    reward -= (versor_to_last_position * last_centerline_vector).sum(axis=1)
    
    penalty = 0
    for bound in [left_bound, right_bound]:
        distances = torch.linalg.norm(bound[:, None] - trajectory[:, :, None], axis=3)
        gaussed_distances = torch.exp(-distances / penalty_sigma / penalty_sigma)
        if only_closest:
            penalty += gaussed_distances.view(-1, gaussed_distances.shape[1] * gaussed_distances.shape[2]).min(axis=1)[0]
        else:
            penalty += gaussed_distances.sum(axis=(1, 2))

    which_beyond = (penalty > reward)
    penalty[which_beyond] = reward[which_beyond]

    return reward, penalty


# TODO: this and the function above can be combined into one
@torch.inference_mode()
def calc_progress_and_penalty_while_driving(
        trajectory: torch.Tensor,
        centerline: torch.Tensor,
        left_bound: torch.Tensor,
        right_bound: torch.Tensor,
        penalty_sigma: float = 0.4,
        only_closest: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Calculates the progress along the centerline + penalty caused by closeness to any of the bounds.
    """
    batch_size = len(trajectory)
    trajectory = trajectory.reshape(batch_size, -1, 2)
    
    last_position = trajectory[:, -1]
    dists = torch.linalg.norm(centerline[None] - last_position[:, None], axis=2)
    closest_centerline_idx = torch.argmin(dists, axis=1)
    dists_along_centerline = torch.linalg.norm(torch.diff(centerline, axis=0), axis=1).cumsum(axis=0)
    dists_along_centerline = torch.tile(dists_along_centerline, (batch_size, 1))
    arange = torch.arange(batch_size)
    closest_centerline_idx_for_diff = closest_centerline_idx - 1
    closest_centerline_idx_for_diff[closest_centerline_idx_for_diff == -1] = 0
    reward = dists_along_centerline[arange, closest_centerline_idx_for_diff]

    versor_to_last_position = last_position - centerline[closest_centerline_idx]
    versor_to_last_position /= torch.linalg.norm(versor_to_last_position) + 1e-10

    closest_centerline_idx_for_last_centerline_vector = closest_centerline_idx + 1
    centerline_len = centerline.shape[1]
    closest_centerline_idx_for_last_centerline_vector[closest_centerline_idx_for_last_centerline_vector == centerline_len] = centerline_len - 1
    last_centerline_vector = centerline[closest_centerline_idx_for_last_centerline_vector] - centerline[closest_centerline_idx_for_last_centerline_vector - 1]
    
    reward -= (versor_to_last_position * last_centerline_vector).sum(axis=1)
    
    penalty = 0
    for bound in [left_bound, right_bound]:
        distances = torch.linalg.norm(bound[None, None] - trajectory[:, :, None], axis=3)
        gaussed_distances = torch.exp(-distances / penalty_sigma / penalty_sigma)
        if only_closest:
            penalty += gaussed_distances.view(-1, gaussed_distances.shape[1] * gaussed_distances.shape[2]).min(axis=1)[0]
        else:
            penalty += gaussed_distances.sum(axis=(1, 2))

    which_beyond = (penalty > reward)
    penalty[which_beyond] = reward[which_beyond]

    return reward, penalty
