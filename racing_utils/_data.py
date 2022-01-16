from pathlib import Path
from datetime import datetime
from typing import Union, Dict, Any, Optional 

import numpy as np
import pandas as pd


class _RaceData:
    def __init__(self, dump_directory: str = '/tmp'):
        self.dump_directory = Path(dump_directory)
        self.states = {
            'yaw': [],
            'position': [],
            'v_x': [],
            'v_y': [],
            'omega': [],
            'delta': [],
            'speed_actuator': [],
            'time': [],
        }
        self.step = 0

    def add(
            self,
            yaw: float,
            position: np.array,
            v_x: float,
            v_y: float,
            omega: float,
            delta: float,
            speed_actuator: float,
            time: float,
            **kwargs,
    ):
        self.states['yaw'].append(yaw)
        self.states['position'].append(position)
        self.states['v_x'].append(v_x)
        self.states['v_y'].append(v_y)
        self.states['omega'].append(omega)
        self.states['delta'].append(delta)
        self.states['speed_actuator'].append(speed_actuator)
        self.states['time'].append(time)
        for key, value in kwargs.items():
            if key not in self.states:
                self.states[key] = []
            self.states[key].append(value)
        self.step += 1
        
    def dump(self, filename: Union[str, Path], additional_data: Optional[Dict[str, Any]] = None):
        if filename is None:
            now = str(datetime.now()).replace(' ', '_').replace(':', '_')
            filename = now + '.pkl'
        data = {
            'data': pd.DataFrame(self.states),
            'additional_data': additional_data,
        }
        # "protocol=4" because "protocol=5" works with Python 3.9 but not with Python 3.7,
        #  which, at the time of writing this, is the default version on Google Colab
        #  and that messes up pd.read_pickle there
        pd.to_pickle(data, self.dump_directory / filename, protocol=4)
