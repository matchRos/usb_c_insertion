#!/usr/bin/env python3

from __future__ import annotations

from dataclasses import dataclass
import math
import os
import sys
from typing import Tuple

import rospy

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from ft_interface import FTInterface
from param_utils import get_param
from robot_interface import RobotInterface
from tf_interface import TFInterface


@dataclass(frozen=True)
class ExtractionResult:
    success: bool
    reason: str
    extracted_distance: float
    pull_force: float
    lateral_force: float
    torque_norm: float
    gripper_opened: bool


class ExtractionController:
    """
    Conservative cable extraction helper with gentle wiggle motion.

    The controller pulls along the extraction axis while adding a small
    oscillation in the tool y/z directions to reduce jamming risk.
    """

    def __init__(
        self,
        robot_interface: RobotInterface,
        tf_interface: TFInterface,
        ft_interface: FTInterface,
    ):
        self._robot = robot_interface
        self._tf = tf_interface
        self._ft = ft_interface

        self._command_rate = float(get_param("~motion/command_rate", 500.0))
        self._extract_distance = float(get_param("~extract/distance", 0.04))
        self._target_pull_force = float(get_param("~extract/pull_force_target", 8.0))
        self._pull_force_tolerance = float(get_param("~extract/pull_force_tolerance", 1.0))
        self._pull_force_gain = float(get_param("~extract/pull_force_gain", 0.01))
        self._max_pull_speed = float(get_param("~extract/max_pull_speed", 0.01))
        self._pulsed_enabled = bool(get_param("~extract/pulsed_enabled", False))
        self._pulse_pull_duration = float(get_param("~extract/pulse_pull_duration", 0.25))
        self._pulse_rest_duration = float(get_param("~extract/pulse_rest_duration", 0.15))
        self._pulse_min_pull_speed = float(get_param("~extract/pulse_min_pull_speed", 0.002))
        self._pulse_pull_force_target = float(
            get_param("~extract/pulse_pull_force_target", self._target_pull_force)
        )
        self._tool_z_direction_sign = float(get_param("~extract/tool_z_direction_sign", 1.0))
        self._wiggle_speed_y = float(get_param("~extract/wiggle_speed_y", 0.002))
        self._wiggle_speed_z = float(get_param("~extract/wiggle_speed_z", 0.002))
        self._wiggle_frequency = float(get_param("~extract/wiggle_frequency", 2.0))
        self._timeout = float(get_param("~extract/timeout", 8.0))
        self._max_lateral_force = float(get_param("~extract/max_lateral_force", 20.0))
        self._max_torque_norm = float(get_param("~extract/max_torque_norm", 3.0))
        self._release_after_extract = bool(get_param("~extract/release_after_extract", True))

    def extract(self) -> ExtractionResult:
        """
        Extract the cable along tool -z with a small wiggle.
        """
        start_pose = self._tf.get_tool_pose_in_base()
        if start_pose is None:
            return ExtractionResult(False, "missing_initial_tf", 0.0, 0.0, 0.0, 0.0, False)
        if self._ft.is_wrench_stale():
            return ExtractionResult(False, "stale_wrench", 0.0, 0.0, 0.0, 0.0, False)

        deadline = rospy.Time.now() + rospy.Duration.from_sec(self._timeout)
        rate = rospy.Rate(max(1.0, self._command_rate))
        start_time = rospy.Time.now()
        rospy.loginfo(
            "[usb_c_insertion] event=extract_start mode=%s distance=%.4f target_pull_force=%.3f pulse_target_pull_force=%.3f pulse_pull_duration=%.3f pulse_rest_duration=%.3f max_pull_speed=%.4f",
            "pulsed" if self._pulsed_enabled else "continuous",
            self._extract_distance,
            self._target_pull_force,
            self._pulse_pull_force_target,
            self._pulse_pull_duration,
            self._pulse_rest_duration,
            self._max_pull_speed,
        )

        while not rospy.is_shutdown():
            if rospy.Time.now() > deadline:
                self._robot.stop_motion()
                return self._build_result(False, "timeout", start_pose, False)
            if self._ft.is_wrench_stale():
                self._robot.stop_motion()
                return self._build_result(False, "stale_wrench", start_pose, False)

            pose = self._tf.get_tool_pose_in_base()
            if pose is None:
                self._robot.stop_motion()
                return self._build_result(False, "missing_tf", start_pose, False)

            tool_y_direction, tool_z_direction = self._tool_frame_directions(pose)
            extraction_direction = tuple(self._tool_z_direction_sign * component for component in tool_z_direction)

            wrench = self._ft.get_filtered_wrench()
            pull_force = max(0.0, wrench.force_z)
            lateral_force = math.sqrt(wrench.force_x * wrench.force_x + wrench.force_y * wrench.force_y)
            torque_norm = math.sqrt(
                wrench.torque_x * wrench.torque_x
                + wrench.torque_y * wrench.torque_y
                + wrench.torque_z * wrench.torque_z
            )
            extracted_distance = self._project_displacement(
                start_pose.pose.position.x,
                start_pose.pose.position.y,
                start_pose.pose.position.z,
                pose.pose.position.x,
                pose.pose.position.y,
                pose.pose.position.z,
                extraction_direction,
            )

            if lateral_force > self._max_lateral_force or torque_norm > self._max_torque_norm:
                self._robot.stop_motion()
                return self._build_result(False, "safety_limit_exceeded", start_pose, False)

            if extracted_distance >= self._extract_distance:
                self._robot.stop_motion()
                gripper_opened = False
                if self._release_after_extract:
                    gripper_opened = self._robot.open_gripper()
                    if not gripper_opened:
                        return self._build_result(False, "gripper_open_failed", start_pose, False)
                return self._build_result(True, "completed", start_pose, gripper_opened)

            elapsed = (rospy.Time.now() - start_time).to_sec()
            pull_phase, pulse_index = self._pulse_state(elapsed)
            if pull_phase:
                bounded_pull_speed = self._compute_pull_speed(pull_force)
            else:
                bounded_pull_speed = 0.0

            phase = 2.0 * math.pi * self._wiggle_frequency * elapsed
            tool_y_velocity = self._wiggle_speed_y * math.sin(phase) if pull_phase else 0.0

            vx = extraction_direction[0] * bounded_pull_speed + tool_y_direction[0] * tool_y_velocity
            vy = extraction_direction[1] * bounded_pull_speed + tool_y_direction[1] * tool_y_velocity
            vz = extraction_direction[2] * bounded_pull_speed + tool_y_direction[2] * tool_y_velocity
            rospy.loginfo_throttle(
                0.5,
                "[usb_c_insertion] event=extract_progress mode=%s pulse=%d phase=%s extracted_distance=%.4f pull_force=%.3f target_pull_force=%.3f lateral_force=%.3f torque_norm=%.3f pull_speed=%.4f",
                "pulsed" if self._pulsed_enabled else "continuous",
                int(pulse_index),
                "pull" if pull_phase else "rest",
                extracted_distance,
                pull_force,
                self._active_pull_force_target(),
                lateral_force,
                torque_norm,
                bounded_pull_speed,
            )
            self._robot.send_twist(vx, vy, vz, 0.0, 0.0, 0.0)
            rate.sleep()

        self._robot.stop_motion()
        return self._build_result(False, "shutdown", start_pose, False)

    def _build_result(self, success, reason, start_pose, gripper_opened):
        pose = self._tf.get_tool_pose_in_base()
        wrench = self._ft.get_filtered_wrench()
        if pose is None:
            return ExtractionResult(success, reason, 0.0, max(0.0, wrench.force_z), math.sqrt(wrench.force_x * wrench.force_x + wrench.force_y * wrench.force_y), math.sqrt(wrench.torque_x * wrench.torque_x + wrench.torque_y * wrench.torque_y + wrench.torque_z * wrench.torque_z), gripper_opened)

        tool_y_direction, tool_z_direction = self._tool_frame_directions(pose)
        extraction_direction = tuple(self._tool_z_direction_sign * component for component in tool_z_direction)
        extracted_distance = self._project_displacement(
            start_pose.pose.position.x,
            start_pose.pose.position.y,
            start_pose.pose.position.z,
            pose.pose.position.x,
            pose.pose.position.y,
            pose.pose.position.z,
            extraction_direction,
        )
        pull_force = max(0.0, wrench.force_z)
        lateral_force = math.sqrt(wrench.force_x * wrench.force_x + wrench.force_y * wrench.force_y)
        torque_norm = math.sqrt(
            wrench.torque_x * wrench.torque_x
            + wrench.torque_y * wrench.torque_y
            + wrench.torque_z * wrench.torque_z
        )
        return ExtractionResult(success, reason, extracted_distance, pull_force, lateral_force, torque_norm, gripper_opened)

    def _pulse_state(self, elapsed: float) -> Tuple[bool, int]:
        if not self._pulsed_enabled:
            return True, 0
        pull_duration = max(0.01, self._pulse_pull_duration)
        rest_duration = max(0.0, self._pulse_rest_duration)
        cycle_duration = pull_duration + rest_duration
        if cycle_duration <= 0.01:
            return True, 1
        cycle_position = elapsed % cycle_duration
        pulse_index = int(elapsed / cycle_duration) + 1
        return cycle_position < pull_duration, pulse_index

    def _compute_pull_speed(self, pull_force: float) -> float:
        target_force = self._active_pull_force_target()
        force_error = target_force - pull_force
        if force_error <= -self._pull_force_tolerance:
            return 0.0
        if self._pulsed_enabled:
            proportional_speed = max(0.0, self._pull_force_gain * max(0.0, force_error))
            requested_speed = max(self._pulse_min_pull_speed, proportional_speed)
        elif abs(force_error) <= self._pull_force_tolerance:
            requested_speed = self._max_pull_speed
        else:
            requested_speed = self._pull_force_gain * force_error
        return max(0.0, min(self._max_pull_speed, requested_speed))

    def _active_pull_force_target(self) -> float:
        return self._pulse_pull_force_target if self._pulsed_enabled else self._target_pull_force

    @staticmethod
    def _project_displacement(x0, y0, z0, x1, y1, z1, direction_xyz):
        delta = (x1 - x0, y1 - y0, z1 - z0)
        return sum(delta[index] * direction_xyz[index] for index in range(3))

    @staticmethod
    def _tool_frame_directions(pose):
        qx = pose.pose.orientation.x
        qy = pose.pose.orientation.y
        qz = pose.pose.orientation.z
        qw = pose.pose.orientation.w
        y_direction = ExtractionController._rotate_vector_by_quaternion(0.0, 1.0, 0.0, qx, qy, qz, qw)
        z_direction = ExtractionController._rotate_vector_by_quaternion(0.0, 0.0, 1.0, qx, qy, qz, qw)
        return y_direction, z_direction

    @staticmethod
    def _rotate_vector_by_quaternion(vector_x, vector_y, vector_z, quat_x, quat_y, quat_z, quat_w):
        norm = math.sqrt(quat_x * quat_x + quat_y * quat_y + quat_z * quat_z + quat_w * quat_w)
        if norm <= 1e-9:
            raise ValueError("Quaternion must be non-zero.")
        qx = quat_x / norm
        qy = quat_y / norm
        qz = quat_z / norm
        qw = quat_w / norm
        rotation_matrix = (
            (
                1.0 - 2.0 * (qy * qy + qz * qz),
                2.0 * (qx * qy - qz * qw),
                2.0 * (qx * qz + qy * qw),
            ),
            (
                2.0 * (qx * qy + qz * qw),
                1.0 - 2.0 * (qx * qx + qz * qz),
                2.0 * (qy * qz - qx * qw),
            ),
            (
                2.0 * (qx * qz - qy * qw),
                2.0 * (qy * qz + qx * qw),
                1.0 - 2.0 * (qx * qx + qy * qy),
            ),
        )
        return (
            rotation_matrix[0][0] * vector_x + rotation_matrix[0][1] * vector_y + rotation_matrix[0][2] * vector_z,
            rotation_matrix[1][0] * vector_x + rotation_matrix[1][1] * vector_y + rotation_matrix[1][2] * vector_z,
            rotation_matrix[2][0] * vector_x + rotation_matrix[2][1] * vector_y + rotation_matrix[2][2] * vector_z,
        )
