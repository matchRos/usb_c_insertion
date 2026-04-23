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

        self._command_rate = float(rospy.get_param("~motion/command_rate", 500.0))
        self._extract_distance = float(rospy.get_param("~extract/distance", 0.04))
        self._target_pull_force = float(rospy.get_param("~extract/pull_force_target", 8.0))
        self._pull_force_tolerance = float(rospy.get_param("~extract/pull_force_tolerance", 1.0))
        self._pull_force_gain = float(rospy.get_param("~extract/pull_force_gain", 0.01))
        self._max_pull_speed = float(rospy.get_param("~extract/max_pull_speed", 0.01))
        self._tool_z_direction_sign = float(rospy.get_param("~extract/tool_z_direction_sign", 1.0))
        self._wiggle_speed_y = float(rospy.get_param("~extract/wiggle_speed_y", 0.002))
        self._wiggle_speed_z = float(rospy.get_param("~extract/wiggle_speed_z", 0.002))
        self._wiggle_frequency = float(rospy.get_param("~extract/wiggle_frequency", 2.0))
        self._timeout = float(rospy.get_param("~extract/timeout", 8.0))
        self._max_lateral_force = float(rospy.get_param("~extract/max_lateral_force", 20.0))
        self._max_torque_norm = float(rospy.get_param("~extract/max_torque_norm", 3.0))
        self._release_after_extract = bool(rospy.get_param("~extract/release_after_extract", True))

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

            force_error = self._target_pull_force - pull_force
            if abs(force_error) <= self._pull_force_tolerance:
                pull_speed = self._max_pull_speed
            else:
                pull_speed = self._pull_force_gain * force_error
            bounded_pull_speed = max(
                0.0,
                min(self._max_pull_speed, pull_speed),
            )

            elapsed = (rospy.Time.now() - start_time).to_sec()
            phase = 2.0 * math.pi * self._wiggle_frequency * elapsed
            tool_y_velocity = self._wiggle_speed_y * math.sin(phase)
            tool_z_velocity = self._wiggle_speed_z * math.cos(phase)

            vx = extraction_direction[0] * bounded_pull_speed + tool_y_direction[0] * tool_y_velocity
            vy = extraction_direction[1] * bounded_pull_speed + tool_y_direction[1] * tool_y_velocity
            vz = extraction_direction[2] * bounded_pull_speed + tool_y_direction[2] * tool_y_velocity
            print(f"pull_speed={bounded_pull_speed:.4f} tool_y_velocity={tool_y_velocity:.4f} tool_z_velocity={tool_z_velocity:.4f} vx={vx:.4f} vy={vy:.4f} vz={vz:.4f}")
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
