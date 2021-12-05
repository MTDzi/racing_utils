from abc import ABC, abstractmethod
from typing import Tuple, Optional
from pathlib import Path

import numpy as np

from ._data import _RaceData


class BaseAgent(ABC):
    collect_data: bool
    ego_data: _RaceData

    @abstractmethod
    def plan(
            ranges: np.array,
            yaw: float,
            pos_x: float,
            pos_y: float,
            linear_vel_x: float,
            linear_vel_y: float,
            angular_vel_z: float,
            lap_time: Optional[float] = None,
    ) -> Tuple[float, float]:
        pass

    def _gather_data(
            self,
            delta: float,
            speed: float,
            target_point: np.array,
            lap_time: Optional[float] = None,
    ):
        if not self.collect_data:
            return

        if lap_time is None:
            raise ValueError('When gathering data, lap_time argument needs to be provided')

        self.ego_data.add(
            time=lap_time,
            position=self.position,
            v_x=self.linear_vel_x,
            v_y=self.linear_vel_y,
            speed_actuator=speed,
            yaw=self._angle_back_to_domain(self.yaw),
            delta=delta,
            omega=self.angular_vel_z,
            target_point=target_point,
        )

    def dump_data(self, filename: Optional[Path] = None):
        additional_data = {
            'lookahead_distance': self.lookahead_distance,
            'speed_setpoint': self.speed,
            'tire_force_max': self.tire_force_max,
            'max_allowable_twist_linear_x': self.max_allowable_twist_linear_x,
            'centerline': self.waypoints,
        }
        self.ego_data.dump(filename, additional_data)

    @staticmethod
    def _angle_back_to_domain(angle: float) -> float:
        two_pi = 2 * np.pi
        if angle > np.pi:
            angle -= two_pi
        elif angle < -np.pi:
            angle += two_pi
        return angle