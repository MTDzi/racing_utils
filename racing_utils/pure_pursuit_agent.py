import time
import sys
from pathlib import Path
from typing import Optional, Tuple, Dict, Any

import numpy as np
import pandas as pd

from .utils import rotate_into_map_coord, closest_point_idx
from .base_agent import BaseAgent
from ._data import _RaceData


class PurePursuitAgent(BaseAgent):

    MAX_STEER_ANGLE = 0.4189
    WHEELBASE = 0.3302
    SCAN_LEN = 1080
    IGNORE_SCAN_PART = 0.2
    ANGLE_MIN = -2.34999990463
    ANGLE_MAX =  2.34999990463
    ANGLE_INCR = 0.00435185199603
    MIN_SEGMENT_LEN = 2

    SEARCH_HORIZON = 15

    QSIZE = 5

    CENTER_IN_DZIK = 0.53

    WALL_SCAN_RESOLUTION = 0.6
    WALL_SCAN_HOW_LONG_NO_NEW = 35
    MAX_RANGE_FOR_WALL_SCAN = 5  # Depending on the track this might need adjusting

    SPEED_NOISE_LEVEL = 0.1
    DELTA_NOISE_LEVEL = 0.01

    def __init__(
            self,
            csv_waypoints_path,
            lookahead_distance,
            speed,
            segment_margin,
            tire_force_max,
            pos_to_seg_coeff,
            target_to_seg_coeff,
            delta_diff_coeff,
            prev_target_coeff,
            nitro_boost,
            max_allowable_twist_linear_x,
            debug,
            scan_walls,
            collect_data,
            direction=1,
            noise=False,
    ):
        self.lookahead_distance = lookahead_distance
        self.speed = speed
        self.segment_margin = segment_margin
        self.tire_force_max = tire_force_max
        self.pos_to_seg_coeff = pos_to_seg_coeff
        self.target_to_seg_coeff = target_to_seg_coeff
        self.delta_diff_coeff = delta_diff_coeff
        self.prev_target_coeff = prev_target_coeff
        self.nitro_boost = nitro_boost
        self.max_allowable_twist_linear_x = max_allowable_twist_linear_x
        self.debug = debug
        self.scan_walls = scan_walls
        self.collect_data = collect_data
        self.noise = noise

        self.max_speed_increment = 100000
        self.max_delta_increment = 0.1

        self.waypoints = self.read_csv_as_array(csv_waypoints_path)[::direction]
        self.interior = self.read_csv_as_array(str(csv_waypoints_path).replace('SOCHI', 'interior'))[::direction]
        self.exterior = self.read_csv_as_array(str(csv_waypoints_path).replace('SOCHI', 'exterior'))[::direction]

        self.closest_point_idx = None
        self.prev_target_point = None
        self.during_avoidance = False
        self.prev_speed = 0.0
        self.prev_delta = 0.0
        self.position = np.array([0, 0])
        self.prev_position = np.array([0, 0])

        self.setup_cache()

        if self.collect_data:
            self.ego_data = _RaceData()

        if self.scan_walls:
            self.wall_scan_thus_far = None
            self.wall_scan_last_seen_new = None
            self.wall_scan_dump_filename = str(csv_waypoints_path).replace('SOCHI', 'wall_scan').replace('csv', 'npz')

    def plan(
            self,
            ranges: np.array,
            yaw: float,
            pos_x: float,
            pos_y: float,
            linear_vel_x: float,
            linear_vel_y: float,
            angular_vel_z: float,
            lap_time: Optional[float] = None,
    ) -> Tuple[float, float]:
        self.ranges = ranges
        self.yaw = yaw
        self.position = np.array([pos_x, pos_y])
        self.linear_vel_x = linear_vel_x
        self.linear_vel_y = linear_vel_y
        self.angular_vel_z = angular_vel_z

        self.keep_ranges(self.ranges, self.position, self.yaw)    

        delta, target_point, target_yaw = self.find_yaw_and_target()

        self.calculate_scan_segments()

        delta, target_point = self.correct_for_collisions(delta, target_point, target_yaw)

        delta, speed = self.correct_for_curvature_and_speed_diff(delta)

        delta, speed = self.cap_delta_and_speed(delta, speed)

        if self.ego_data is not None:
            self._gather_data(
                self.yaw, self.position, linear_vel_x, linear_vel_y, angular_vel_z, delta, speed, lap_time,
                target_point=target_point,
            )

        if self.noise is True:
            speed += self.SPEED_NOISE_LEVEL * np.random.normal()
            delta += self.DELTA_NOISE_LEVEL * np.random.normal()
            delta, speed = self.cap_delta_and_speed(delta, speed)

        return speed, delta

    @staticmethod
    def read_csv_as_array(csv_path):
        with open(csv_path, 'r') as f:
            data = f.read().split('\n')[:-1]
            data = [[float(__) for __ in _.split(',')] for _ in data]
        return np.array(data)[:, :2]

    def setup_cache(self):
        # Cache the following attributes instead of re-calculating them every time
        self.idx_to_angle = np.array([
            self.ANGLE_MIN + idx * self.ANGLE_INCR
            for idx in range(self.SCAN_LEN)
        ])
        self.versor = np.array([
            np.array([np.cos(self.idx_to_angle[idx]), np.sin(self.idx_to_angle[idx])])
            for idx in range(self.SCAN_LEN)
        ])
        self.waypoints_diffs = np.linalg.norm(
            np.diff(np.r_[self.waypoints, self.waypoints[[0]]], axis=0),
            axis=1,
        )

    def keep_ranges(self, ranges, position, yaw):
        if not self.scan_walls:
            return

        which_closest = (ranges < self.MAX_RANGE_FOR_WALL_SCAN)
        ranges_as_vec = ranges[which_closest, np.newaxis] * self.versor[which_closest]
        ranges_as_vec = position + rotate_into_map_coord(ranges_as_vec, yaw)

        if self.wall_scan_thus_far is None:
            # Decimate the first scan to create the initial wall scans
            self.wall_scan_thus_far = ranges_as_vec[::10]
            self.wall_scan_last_seen_new = time.time()
        else:
            dist_to_previous = np.linalg.norm(ranges_as_vec[np.newaxis] - self.wall_scan_thus_far[:, np.newaxis], axis=2)
            min_dist_to_previous = dist_to_previous.min(axis=0)
            which_new = (min_dist_to_previous > self.WALL_SCAN_RESOLUTION)
            if which_new.sum() > 0:
                self.wall_scan_last_seen_new = time.time()
            elif time.time() - self.wall_scan_last_seen_new > self.WALL_SCAN_HOW_LONG_NO_NEW:
                print(f'Dumping wall scan to: "{self.wall_scan_dump_filename}"')
                np.savez_compressed(self.wall_scan_dump_filename, wall_scan=self.wall_scan_thus_far, resolution=self.WALL_SCAN_RESOLUTION)
                sys.exit()

            self.wall_scan_thus_far = np.r_[ranges_as_vec[which_new], self.wall_scan_thus_far]
            print(f'Wall scans thus far: {self.wall_scan_thus_far.shape[0]}')

    def find_yaw_and_target(self):
        target_point = self.find_target_point_along_waypoints()            
        target_yaw = self.get_angle(target_point - self.position)
        yaw_correction = target_yaw - self.yaw
        if self.debug:
            self.publish_target_point(target_point, self.target_pub)
        return self._angle_back_to_domain(yaw_correction), target_point, target_yaw

    def calculate_scan_segments(self):
        # Look for segment boundaries
        segment_indices = []
        diff = np.abs(np.diff(self.ranges))
        segment_start = 0
        where_diff_break = np.r_[
            np.argwhere(diff > self.segment_margin).flatten(),
            self.SCAN_LEN - 1,
        ]
        for i in where_diff_break:
            segment_end = i
            if segment_end - segment_start < self.MIN_SEGMENT_LEN:
                segment_start = segment_end + 1
                continue

            segment_indices.append([segment_start, segment_end])
            segment_start = segment_end + 1

        # Now, based on the indices that define segment boundaries, build an array
        #  that contains start / end points of each segment
        # TODO: this can be done faster
        self.segments = []
        for pair in segment_indices:
            two_ranges = self.ranges[pair]
            if (two_ranges > self.lookahead_distance).any():
                # This situation occurs when the segment is very large
                #  (in partucular, a wall) that doesn't require avoidance
                continue
                
            # This line...
            start_end = two_ranges[:, np.newaxis] * self.versor[pair]
            # ...is equivalent to:
            # start_end = [
            #    two_ranges[0] * self.versor[pair[0]],
            #    two_ranges[1] * self.versor[pair[1]],
            # ]
            # but faster

            self.segments.append(start_end)

        if self.segments == []:
            return

        self.segments = np.array(self.segments)

        # Transform into the /map coordinate system
        self.segments = rotate_into_map_coord(self.segments, self.yaw)
        self.segments += self.position

        ok_rows = []
        for i in range(len(self.segments)):
            segment = self.segments[i]
            min_avg_interior_dist = np.linalg.norm(np.expand_dims(segment, 0) - np.expand_dims(self.interior, 1), axis=2).min(axis=0).mean()
            if min_avg_interior_dist <= self.segment_margin:
                continue

            min_avg_exterior_dist = np.linalg.norm(np.expand_dims(segment, 0) - np.expand_dims(self.exterior, 1), axis=2).min(axis=0).mean()
            if min_avg_exterior_dist <= self.segment_margin:
                continue

            ok_rows.append(i)

        self.segments = self.segments[ok_rows]

    def find_target_point_along_waypoints(self):
        if (self.closest_point_idx is not None) and (self.closest_point_idx + self.SEARCH_HORIZON < len(self.waypoints)):
            waypoints_subset = self.waypoints[self.closest_point_idx:self.closest_point_idx + self.SEARCH_HORIZON]
            idx_offset = self.closest_point_idx
        else:
            waypoints_subset = self.waypoints
            idx_offset = 0

        self.closest_point_idx = curr_idx = closest_point_idx(self.position, waypoints_subset) + idx_offset

        dist = 0
        while dist < self.lookahead_distance:
            dist += self.waypoints_diffs[curr_idx]
            curr_idx =  (curr_idx + 1) % self.waypoints.shape[0]

        dupa = curr_idx
        # TODO:
        # while dist < 2 * self.lookahead_distance:
        #     dist += self.waypoints_diffs[curr_idx]
        #     curr_idx =  (curr_idx + 1) % self.waypoints.shape[0]
        
        return self.waypoints[dupa]

    def correct_for_collisions(self, delta, target_point, target_yaw):
        if self.segments == []:
            self.prev_target_point = target_point
            return delta, target_point

        direction = self.position - target_point
        perpendicular = np.array([-direction[1], direction[0]]) / np.linalg.norm(direction)

        # First, create extended segments
        extended_segments = self._calc_extended_segments(perpendicular)

        # Now, look for intersections
        intersected_segments = self._calc_intersected_segments(
            extended_segments,
            delta,
            target_point,
            target_yaw,
        )

        if intersected_segments == []:
            self.during_avoidance = False
            self.prev_target_point = target_point
            return delta, target_point
            
        closest_segment_idx, edge_idx, _ = min(
            intersected_segments,
            key=lambda triple: triple[0],
        )
        new_target_point = extended_segments[closest_segment_idx, edge_idx]
        if self.debug:
            self.publish_target_point(new_target_point, self.corrected_target_pub)
            
        self.prev_target_point = new_target_point
        self.during_avoidance = True
        new_target_yaw = self.get_angle(new_target_point - self.position)

        # TODO: this seemed to help but not when the turn was sharp
        # yaw_correction = np.clip(new_target_yaw - self.yaw, -0.2, 0.2)
        yaw_correction = new_target_yaw - self.yaw

        new_delta = self._angle_back_to_domain(yaw_correction)

        return new_delta, new_target_point

    def _calc_extended_segments(self, perpendicular):
        extended_segments = np.zeros_like(self.segments)
        for i in range(len(self.segments)):
            start, end = self.segments[i]
            segment_vec = (end - start)
            sign = np.sign(segment_vec.dot(perpendicular))
            extended_segments[i] = (
                start - sign*perpendicular * self.segment_margin,
                end + sign*perpendicular * self.segment_margin,
            )
        return extended_segments

    def _calc_intersected_segments(self, extended_segments, delta, target_point, target_yaw):
        intersected_segments = []
        vec_to_target = np.array([self.position, target_point])
        for seg_idx in range(len(extended_segments)):
            seg_vec = extended_segments[seg_idx]
            if not self.intersects(vec_to_target, seg_vec):
                continue

            dists = (
                self.pos_to_seg_coeff * np.linalg.norm(seg_vec - self.position, axis=1)
                + self.target_to_seg_coeff * np.linalg.norm(seg_vec - target_point, axis=1)
            )

            for j, _ in enumerate(seg_vec):
                target_yaw = self.get_angle(seg_vec[j] - self.position)
                yaw_correction = target_yaw - self.yaw
                new_delta = self._angle_back_to_domain(yaw_correction)
                dists[j] += self.delta_diff_coeff * np.abs(delta - new_delta)

            which_closer = np.argmin(dists)
            intersected_segments.append([seg_idx, which_closer, dists[which_closer]])

        return intersected_segments

    def correct_for_curvature_and_speed_diff(self, delta):
        if delta > self.MAX_STEER_ANGLE:
            delta = self.MAX_STEER_ANGLE
        elif delta < -self.MAX_STEER_ANGLE:
            delta = -self.MAX_STEER_ANGLE

        kappa = np.abs(np.tan(delta) / self.WHEELBASE)
        cap_on_speed = np.sqrt(self.tire_force_max / kappa)
        new_speed = min(self.speed, cap_on_speed)

        going_straight = (np.abs(delta) < 0.01)
        speed_below_max_allowable = (self.linear_vel_x < self.max_allowable_twist_linear_x)
        not_spinning = (self.angular_vel_z < 0.001)
        if going_straight and speed_below_max_allowable and not_spinning:
            new_speed += self.nitro_boost
            delta = 0
        
        return delta, new_speed

    def cap_delta_and_speed(self, delta, speed):
        new_delta = np.clip(delta, self.prev_delta - self.max_delta_increment, self.prev_delta + self.max_delta_increment)
        new_speed = np.clip(speed, self.prev_speed - self.max_speed_increment, self.prev_speed + self.max_speed_increment)
        self.prev_delta = new_delta
        self.prev_speed = new_speed
        return new_delta, new_speed

    def _compose_additional_data(self) -> Dict[str, Any]:
        return {
            'lookahead_distance': self.lookahead_distance,
            'speed_setpoint': self.speed,
            'tire_force_max': self.tire_force_max,
            'max_allowable_twist_linear_x': self.max_allowable_twist_linear_x,
            'centerline': self.waypoints,
        }

    @staticmethod
    def intersects(segA, segB):
        # Stolen from
        #  https://stackoverflow.com/questions/3838329/how-can-i-check-if-two-segments-intersect
        dx0 = segA[1, 0] - segA[0, 0]
        dx1 = segB[1, 0] - segB[0, 0]
        dy0 = segA[1, 1] - segA[0, 1]
        dy1 = segB[1, 1] - segB[0, 1]
        p0 = dy1 * (segB[1, 0] - segA[0, 0]) - dx1 * (segB[1, 1] - segA[0, 1])
        p1 = dy1 * (segB[1, 0] - segA[1, 0]) - dx1 * (segB[1, 1] - segA[1, 1])
        p2 = dy0 * (segA[1, 0] - segB[0, 0]) - dx0 * (segA[1, 1] - segB[0, 1])
        p3 = dy0 * (segA[1, 0] - segB[1, 0]) - dx0 * (segA[1, 1] - segB[1, 1])
        return (p0 * p1 <= 0) & (p2 * p3 <= 0)
    
    @staticmethod
    def get_angle(delta):
        return np.arctan2(delta[1], delta[0])
