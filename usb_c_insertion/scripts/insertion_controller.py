#!/usr/bin/env python3

from __future__ import annotations

from dataclasses import dataclass
import math
import os
import sys
from typing import Optional, Tuple

import rospy

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from ft_interface import FTInterface
from robot_interface import RobotInterface
from tf_interface import TFInterface


@dataclass(frozen=True)
class InsertionResult:
    success: bool
    reason: str
    inserted_depth: float
    contact_force: float


class InsertionController:
    """
    Focused helper for the final cable insertion step.

    This class owns the insertion-axis force regulation and the final insertion
    condition check so the state machine can keep its transitions explicit.
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

        self._command_rate = float(rospy.get_param("~motion/command_rate", 100.0))
        self._insertion_depth = float(rospy.get_param("~insert/insertion_depth", 0.004))
        self._contact_force_target = float(rospy.get_param("~insert/contact_force_target", 10.0))
        self._contact_force_tolerance = float(rospy.get_param("~insert/contact_force_tolerance", 1.0))
        self._force_control_gain = float(rospy.get_param("~insert/force_control_gain", 0.001))
        self._force_control_speed_limit = float(rospy.get_param("~insert/force_control_speed_limit", 0.003))
        self._force_control_timeout = float(rospy.get_param("~insert/force_control_timeout", 4.0))
        self._min_insertion_time = float(rospy.get_param("~insert/min_insertion_time", 0.0))
        self._force_success_min_depth = float(rospy.get_param("~insert/force_success_min_depth", 0.0))
        self._wiggle_enabled = bool(rospy.get_param("~insert/wiggle_enabled", False))
        self._wiggle_speed_y = float(rospy.get_param("~insert/wiggle_speed_y", 0.0))
        self._wiggle_speed_x = float(rospy.get_param("~insert/wiggle_speed_x", 0.0))
        self._wiggle_frequency = float(rospy.get_param("~insert/wiggle_frequency", 2.0))
        self._release_force_threshold = float(rospy.get_param("~insert/release_force_threshold", 2.0))

    def insert_until_depth(
        self,
        reference_point_xyz: Tuple[float, float, float],
        insertion_direction_xyz: Tuple[float, float, float],
        force_control_timeout: Optional[float] = None,
        wiggle_y_direction_xyz: Optional[Tuple[float, float, float]] = None,
        wiggle_x_direction_xyz: Optional[Tuple[float, float, float]] = None,
    ) -> InsertionResult:
        """
        Drive along the insertion axis while regulating tool-frame contact force.
        """
        direction = self._normalize_vector(insertion_direction_xyz)
        timeout = (
            self._force_control_timeout
            if force_control_timeout is None or force_control_timeout <= 0.0
            else float(force_control_timeout)
        )
        deadline = rospy.Time.now() + rospy.Duration.from_sec(timeout)
        rate = rospy.Rate(max(1.0, self._command_rate))
        start_time = rospy.Time.now()
        wiggle_y_direction = self._prepare_lateral_direction(wiggle_y_direction_xyz, direction)
        wiggle_x_direction = self._prepare_lateral_direction(wiggle_x_direction_xyz, direction)

        while not rospy.is_shutdown():
            if rospy.Time.now() > deadline:
                self._robot.stop_motion()
                return self._build_result(False, "timeout", reference_point_xyz, direction)
            if self._ft.is_wrench_stale():
                self._robot.stop_motion()
                return self._build_result(False, "stale_wrench", reference_point_xyz, direction)

            pose = self._tf.get_tool_pose_in_base()
            if pose is None:
                self._robot.stop_motion()
                return self._build_result(False, "missing_tf", reference_point_xyz, direction)

            inserted_depth = self._project_displacement_from_reference(
                reference_point_xyz,
                direction,
                pose.pose.position.x,
                pose.pose.position.y,
                pose.pose.position.z,
            )
            contact_force = self._get_contact_force()
            force_error = self._contact_force_target - contact_force
            elapsed = (rospy.Time.now() - start_time).to_sec()
            success_reason = self._success_reason(inserted_depth, contact_force, elapsed)
            if success_reason is not None:
                self._robot.stop_motion()
                rospy.loginfo(
                    "[usb_c_insertion] event=insert_cable_complete reason=%s inserted_depth=%.4f contact_force=%.3f",
                    success_reason,
                    inserted_depth,
                    contact_force,
                )
                return InsertionResult(True, success_reason, inserted_depth, contact_force)
            if contact_force >= self._contact_force_target:
                rospy.loginfo_throttle(
                    0.5,
                    "[usb_c_insertion] event=insert_force_target_waiting inserted_depth=%.4f min_depth=%.4f elapsed=%.2f min_time=%.2f contact_force=%.3f",
                    inserted_depth,
                    self._force_success_min_depth,
                    elapsed,
                    self._min_insertion_time,
                    contact_force,
                )

            if abs(force_error) <= self._contact_force_tolerance:
                speed = self._force_control_speed_limit
            else:
                speed = self._force_control_gain * force_error

            bounded_speed = max(
                -self._force_control_speed_limit,
                min(self._force_control_speed_limit, speed),
            )
            lateral_velocity = self._compute_wiggle_velocity(
                start_time,
                wiggle_y_direction,
                wiggle_x_direction,
            )
            self._robot.send_twist(
                direction[0] * bounded_speed + lateral_velocity[0],
                direction[1] * bounded_speed + lateral_velocity[1],
                direction[2] * bounded_speed + lateral_velocity[2],
                0.0,
                0.0,
                0.0,
            )
            rospy.loginfo_throttle(
                0.5,
                "[usb_c_insertion] event=insert_progress inserted_depth=%.4f target_depth=%.4f contact_force=%.3f target_force=%.3f insert_speed=%.4f wiggle_enabled=%s",
                inserted_depth,
                self._insertion_depth,
                contact_force,
                self._contact_force_target,
                bounded_speed,
                str(self._wiggle_enabled).lower(),
            )
            rate.sleep()

        self._robot.stop_motion()
        return self._build_result(False, "shutdown", reference_point_xyz, direction)

    def check_insertion(
        self,
        reference_point_xyz: Tuple[float, float, float],
        insertion_direction_xyz: Tuple[float, float, float],
    ) -> InsertionResult:
        """
        Evaluate whether the current pose and force satisfy the insertion criteria.
        """
        direction = self._normalize_vector(insertion_direction_xyz)
        pose = self._tf.get_tool_pose_in_base()
        if pose is None:
            return InsertionResult(False, "missing_tf", 0.0, self._get_contact_force())

        inserted_depth = self._project_displacement_from_reference(
            reference_point_xyz,
            direction,
            pose.pose.position.x,
            pose.pose.position.y,
            pose.pose.position.z,
        )
        contact_force = self._get_contact_force()
        success_reason = self._success_reason(inserted_depth, contact_force)
        return InsertionResult(
            success=(success_reason is not None),
            reason=(success_reason or "threshold_not_met"),
            inserted_depth=inserted_depth,
            contact_force=contact_force,
        )

    def _build_result(
        self,
        success: bool,
        reason: str,
        reference_point_xyz: Tuple[float, float, float],
        direction_xyz: Tuple[float, float, float],
    ) -> InsertionResult:
        pose = self._tf.get_tool_pose_in_base()
        if pose is None:
            return InsertionResult(success, reason, 0.0, self._get_contact_force())

        inserted_depth = self._project_displacement_from_reference(
            reference_point_xyz,
            direction_xyz,
            pose.pose.position.x,
            pose.pose.position.y,
            pose.pose.position.z,
        )
        return InsertionResult(success, reason, inserted_depth, self._get_contact_force())

    def _get_contact_force(self) -> float:
        wrench = self._ft.get_filtered_wrench()
        return max(0.0, -wrench.force_z)

    def _success_reason(self, inserted_depth: float, contact_force: float, elapsed: Optional[float] = None):
        if inserted_depth >= self._insertion_depth:
            return "depth_reached"
        force_success_time_ready = elapsed is None or elapsed >= self._min_insertion_time
        force_success_depth_ready = inserted_depth >= self._force_success_min_depth
        if contact_force >= self._contact_force_target and force_success_time_ready and force_success_depth_ready:
            return "force_reached"
        return None

    def _compute_wiggle_velocity(
        self,
        start_time: rospy.Time,
        wiggle_y_direction: Optional[Tuple[float, float, float]],
        wiggle_x_direction: Optional[Tuple[float, float, float]],
    ) -> Tuple[float, float, float]:
        if not self._wiggle_enabled:
            return (0.0, 0.0, 0.0)
        if wiggle_y_direction is None and wiggle_x_direction is None:
            return (0.0, 0.0, 0.0)

        elapsed = (rospy.Time.now() - start_time).to_sec()
        phase = 2.0 * math.pi * self._wiggle_frequency * elapsed
        velocity = [0.0, 0.0, 0.0]
        if wiggle_y_direction is not None:
            y_speed = self._wiggle_speed_y * math.sin(phase)
            for index in range(3):
                velocity[index] += wiggle_y_direction[index] * y_speed
        if wiggle_x_direction is not None:
            x_speed = self._wiggle_speed_x * math.cos(phase)
            for index in range(3):
                velocity[index] += wiggle_x_direction[index] * x_speed
        return (velocity[0], velocity[1], velocity[2])

    @staticmethod
    def _prepare_lateral_direction(direction_xyz, insertion_direction_xyz):
        if direction_xyz is None:
            return None
        projected = sum(direction_xyz[index] * insertion_direction_xyz[index] for index in range(3))
        lateral = tuple(
            direction_xyz[index] - projected * insertion_direction_xyz[index]
            for index in range(3)
        )
        try:
            return InsertionController._normalize_vector(lateral)
        except ValueError:
            return None

    @staticmethod
    def _normalize_vector(direction_xyz: Tuple[float, float, float]) -> Tuple[float, float, float]:
        magnitude = math.sqrt(sum(component * component for component in direction_xyz))
        if magnitude <= 1e-9:
            raise ValueError("insertion_direction_xyz must be non-zero")
        return tuple(component / magnitude for component in direction_xyz)

    @staticmethod
    def _project_displacement_from_reference(
        reference_point_xyz: Tuple[float, float, float],
        direction_xyz: Tuple[float, float, float],
        x: float,
        y: float,
        z: float,
    ) -> float:
        delta = (
            x - reference_point_xyz[0],
            y - reference_point_xyz[1],
            z - reference_point_xyz[2],
        )
        return sum(delta[index] * direction_xyz[index] for index in range(3))
