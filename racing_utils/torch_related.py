from typing import Dict, Tuple, NewType

import torch

from sklearn.preprocessing import StandardScaler


Batch = NewType('Batch', Dict[str, torch.Tensor])


class TensorStandardScaler(StandardScaler):

    def __init__(self, device, **kwargs):
        super().__init__(**kwargs)
        self.device = device
        self.tensorfied = False

    def tensorfy(self):
        self.mean_ = torch.tensor(self.mean_, device=self.device, dtype=torch.float)
        self.scale_ = torch.tensor(self.scale_, device=self.device, dtype=torch.float)
        self.tensorfied = True

    def transform(self, X, copy=None):
        if not self.tensorfied:
            raise ValueError('You need to call `tensorfy` first')

        if copy is False:
            X -= self.mean_
            X /= self.scale_
        else:
            X = (X - self.mean_) / self.scale_

        return X

    def inverse_transform(self, X, copy=None):
        if not self.tensorfied:
            raise ValueError('You need to call `tensorfy` first')
            
        if copy is False:
            X *= self.scale_
            X += self.mean_
        else:
            X = self.scale_ * X + self.mean_

        return X


def scale_batch_and_to_device(
        device: str,
        features_scalers: Dict[str, TensorStandardScaler],
        targets_scalers: Dict[str, TensorStandardScaler],
        features_batch: Batch,
        targets_batch: Batch,
) -> Tuple[Batch, Batch]:
    features_batch = {
        feature_name: features_scalers[feature_name].transform(features_batch[feature_name].float().to(device))
        for feature_name in features_batch.keys()
    }
    targets_batch = {
        target_name: targets_scalers[target_name].transform(targets_batch[target_name].float().to(device))
        for target_name in targets_batch.keys()
    }
    return features_batch, targets_batch


def calc_reward_and_penalty(trajectory, centerline, left_bound, right_bound, penalty_sigma=0.4):
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
        penalty += torch.exp(-torch.linalg.norm(bound[:, None] - trajectory[:, :, None], axis=3) / penalty_sigma / penalty_sigma).sum(axis=(1, 2))

    which_beyond = (penalty > reward)
    penalty[which_beyond] = reward[which_beyond]

    return reward, penalty