from abc import ABC, abstractmethod
from typing import Tuple, Optional, Dict, Any
from pathlib import Path

import numpy as np

from ._data import _RaceData


class BaseAgent(ABC):
    collect_data: bool
    ego_data: _RaceData = None

    def setup_data_collection(self):
        self.ego_data = _RaceData()

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
            yaw: float,
            position: np.array,
            v_x: float,
            v_y: float,
            omega: float,
            delta: float,
            speed: float,
            lap_time: float,
            **additional_info
    ):
        self.ego_data.add(
            yaw=self._angle_back_to_domain(yaw),
            position=position,
            v_x=v_x,
            v_y=v_y,
            omega=omega,
            delta=delta,
            speed_actuator=speed,
            time=lap_time,
            **additional_info,
        )

    def dump_data(self, filename: Optional[Path] = None):
        additional_data = self._compose_additional_data()
        self.ego_data.dump(filename, additional_data)
        self.ego_data = _RaceData()

    @abstractmethod
    def _compose_additional_data(self) -> Dict[str, Any]:
        pass

    @staticmethod
    def _angle_back_to_domain(angle: float) -> float:
        two_pi = 2 * np.pi
        if angle > np.pi:
            angle -= two_pi
        elif angle < -np.pi:
            angle += two_pi
        return angle
