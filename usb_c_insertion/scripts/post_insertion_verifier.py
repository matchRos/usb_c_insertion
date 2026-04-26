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

from ft_interface import FTInterface, WrenchData
from robot_interface import RobotInterface
from tf_interface import TFInterface


@dataclass(frozen=True)
class PostInsertionVerificationResult:
    success: bool
    reason: str
    counterforce_y: float
    counterforce_z: float


class PostInsertionVerifier:
    """
    Verify a seated insertion using tiny tool-frame perturbations.

    The verifier performs very small motions along tool y and tool z, measures
    the resulting opposing force in the tool frame, and opens the gripper when
    both checks pass.
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
        self._position_tolerance = float(rospy.get_param("~motion/pose_servo_position_tolerance", 0.0015))
        self._move_distance_y = float(rospy.get_param("~verify/move_distance_y", 0.0001))
        self._move_distance_z = float(rospy.get_param("~verify/move_distance_z", 0.0001))
        self._move_timeout = float(rospy.get_param("~verify/move_timeout", 2.0))
        self._counterforce_threshold_y = float(rospy.get_param("~verify/counterforce_threshold_y", 5.0))
        self._counterforce_threshold_z = float(rospy.get_param("~verify/counterforce_threshold_z", 5.0))
        self._settle_time = float(rospy.get_param("~verify/settle_time", 0.1))

    def verify_retention(self, move_timeout: Optional[float] = None) -> PostInsertionVerificationResult:
        """
        Run the retention check while keeping the gripper closed.
        """
        start_pose = self._tf.get_tool_pose_in_base()
        if start_pose is None:
            return PostInsertionVerificationResult(False, "missing_initial_tf", 0.0, 0.0)
        if self._ft.is_wrench_stale():
            return PostInsertionVerificationResult(False, "stale_wrench", 0.0, 0.0)

        baseline = self._ft.get_filtered_wrench()
        counterforce_y, y_ok = self._probe_counterforce(
            reference_pose=start_pose,
            baseline=baseline,
            local_offset_xyz=(0.0, self._move_distance_y, 0.0),
            force_axis="y",
            threshold=self._counterforce_threshold_y,
            move_name="verify_retention_y",
            move_timeout=move_timeout,
        )
        if not y_ok:
            return PostInsertionVerificationResult(False, "counterforce_y_not_detected", counterforce_y, 0.0)

        counterforce_z, z_ok = self._probe_counterforce(
            reference_pose=start_pose,
            baseline=baseline,
            local_offset_xyz=(0.0, 0.0, self._move_distance_z),
            force_axis="z",
            threshold=self._counterforce_threshold_z,
            move_name="verify_retention_z",
            move_timeout=move_timeout,
        )
        if not z_ok:
            return PostInsertionVerificationResult(False, "counterforce_z_not_detected", counterforce_y, counterforce_z)

        return PostInsertionVerificationResult(True, "verified", counterforce_y, counterforce_z)

    def _probe_counterforce(
        self,
        reference_pose,
        baseline: WrenchData,
        local_offset_xyz: Tuple[float, float, float],
        force_axis: str,
        threshold: float,
        move_name: str,
        move_timeout: Optional[float],
    ) -> Tuple[float, bool]:
        world_offset = self._rotate_vector_by_quaternion(
            local_offset_xyz[0],
            local_offset_xyz[1],
            local_offset_xyz[2],
            reference_pose.pose.orientation.x,
            reference_pose.pose.orientation.y,
            reference_pose.pose.orientation.z,
            reference_pose.pose.orientation.w,
        )
        target_x = reference_pose.pose.position.x + world_offset[0]
        target_y = reference_pose.pose.position.y + world_offset[1]
        target_z = reference_pose.pose.position.z + world_offset[2]
        orientation = (
            reference_pose.pose.orientation.x,
            reference_pose.pose.orientation.y,
            reference_pose.pose.orientation.z,
            reference_pose.pose.orientation.w,
        )

        if not self._move_to_pose(target_x, target_y, target_z, orientation, move_name, move_timeout):
            return 0.0, False

        rospy.sleep(max(0.0, self._settle_time))
        measured = self._ft.get_filtered_wrench()
        counterforce = self._compute_counterforce_delta(baseline, measured, force_axis)
        rospy.loginfo(
            "[usb_c_insertion] event=post_insertion_counterforce axis=%s counterforce=%.3f threshold=%.3f",
            force_axis,
            counterforce,
            threshold,
        )

        self._move_to_pose(
            reference_pose.pose.position.x,
            reference_pose.pose.position.y,
            reference_pose.pose.position.z,
            orientation,
            move_name + "_return",
            move_timeout,
        )
        return counterforce, counterforce >= threshold

    def _move_to_pose(
        self,
        x: float,
        y: float,
        z: float,
        orientation_xyzw,
        move_name: str,
        move_timeout: Optional[float],
    ) -> bool:
        self._robot.send_pose_target(
            x,
            y,
            z,
            qx=orientation_xyzw[0],
            qy=orientation_xyzw[1],
            qz=orientation_xyzw[2],
            qw=orientation_xyzw[3],
        )
        self._robot.enable_pose_servo(True)

        timeout = self._move_timeout if move_timeout is None or move_timeout <= 0.0 else float(move_timeout)
        deadline = rospy.Time.now() + rospy.Duration.from_sec(timeout)
        rate = rospy.Rate(max(1.0, self._command_rate))
        while not rospy.is_shutdown():
            if rospy.Time.now() > deadline:
                self._robot.stop_motion()
                rospy.logerr("[usb_c_insertion] event=post_insertion_move_timeout move=%s", move_name)
                return False

            pose = self._tf.get_tool_pose_in_base()
            if pose is None:
                self._robot.stop_motion()
                rospy.logerr("[usb_c_insertion] event=post_insertion_move_missing_tf move=%s", move_name)
                return False

            dx = x - pose.pose.position.x
            dy = y - pose.pose.position.y
            dz = z - pose.pose.position.z
            distance = math.sqrt(dx * dx + dy * dy + dz * dz)
            if distance <= self._position_tolerance:
                self._robot.stop_motion()
                return True
            rate.sleep()

        self._robot.stop_motion()
        return False

    @staticmethod
    def _compute_counterforce_delta(baseline: WrenchData, measured: WrenchData, force_axis: str) -> float:
        if force_axis == "y":
            return abs(measured.force_y - baseline.force_y)
        if force_axis == "z":
            return abs(measured.force_z - baseline.force_z)
        raise ValueError("Unsupported force_axis: %s" % force_axis)

    @staticmethod
    def _rotate_vector_by_quaternion(
        vector_x: float,
        vector_y: float,
        vector_z: float,
        quat_x: float,
        quat_y: float,
        quat_z: float,
        quat_w: float,
    ):
        norm = math.sqrt(quat_x * quat_x + quat_y * quat_y + quat_z * quat_z + quat_w * quat_w)
        if norm <= 1e-9:
            raise ValueError("Tool quaternion must be non-zero.")

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
