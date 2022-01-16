import time
from typing import List

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from moviepy.editor import VideoClip
from moviepy.video.io.bindings import mplfig_to_npimage


class RaceVideo:
    def __init__(self, races: List[pd.DataFrame], fps: float):
        self.races = races
        self.fps = fps
        self.cached_track_ax = None

    def make_movie(self, simulation_logger):
        trajectory, collision_log = simulation_logger.trajectory, simulation_logger.collision_log
        start_time = time.time()
        fig, ax = plt.subplots()

        duration = len(trajectory) // FPS

        def make_frame(t):
            ax.clear()
            self.track.plot_track(ax)
            t = int(FPS * t) + 1

            # TODO: can be done just once, for all t, but since it's not a bottleneck
            #  (right now), I'm leaving it as it is
            x_min = trajectory[t, 0, 0] - WINDOW_WIDTH
            x_max = x_min + 2 * WINDOW_WIDTH
            y_min = trajectory[t, 0, 1] - WINDOW_WIDTH
            y_max = y_min + 2 * WINDOW_WIDTH

            for car_id in range(self.num_cars):
                x, y, angle, speed = trajectory[t - 1, car_id]
                ax.plot(
                    trajectory[:t, car_id, 0], trajectory[:t, car_id, 1],
                    alpha=0.6, label=f'car_{car_id} {speed:.1f}[m/s]',
                )
                # TODO: same here, this rotation can be done for all t and cars at once
                # Now, plot the car as a rectangle
                x = x - HALF_CAR_LENGTH * np.cos(angle) + HALF_CAR_WIDTH * np.sin(angle)
                y = y - HALF_CAR_LENGTH * np.sin(angle) - HALF_CAR_WIDTH * np.cos(angle)
                angle = 180 / 3.14 * angle

                color = 'black'
                local_collision_log = collision_log[t]
                if local_collision_log and car_id in local_collision_log:
                    color = 'red'

                rect = patches.Rectangle(
                    (x, y),
                    width=CAR_LENGTH,
                    height=CAR_WIDTH,
                    angle=angle,
                    linewidth=1,
                    facecolor=color,
                    alpha=0.6,
                )
                ax.add_patch(rect)

            ax.set_xlim(x_min, x_max)
            ax.set_ylim(y_min, y_max)
            ax.set_aspect(1)
            ax.legend(loc='upper left')
            return mplfig_to_npimage(fig)

        animation = VideoClip(make_frame, duration=duration)
        animation.write_videofile(
            f'/{SIMULATOR_OUTPUT_PATH}/{self.scenario_name}_{PLAYBACK_SUFFIX}',
            fps=FPS,
            preset='ultrafast',
        )
        print(f'Making the video took: {time.time() - start_time:.2f}[s]')

    def _plot_one_traj(self, ax, trajectory, x_min_max, y_min_max, car_id):
        x_min, x_max = x_min_max
        y_min, y_max = y_min_max
        ax.plot(
            trajectory[:, 0], trajectory[:, 1],
            alpha=0.6, label=f'car_{car_id}',
        )
        ax.set_xlim(x_min - 0.1 * np.abs(x_min), x_max + 0.1 * np.abs(x_max))
        ax.set_ylim(y_min - 0.1 * np.abs(y_min), y_max + 0.1 * np.abs(y_max))
        ax.set_aspect(1)
        ax.legend()
