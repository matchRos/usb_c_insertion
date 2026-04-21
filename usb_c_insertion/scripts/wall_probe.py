#!/usr/bin/env python3

from __future__ import annotations

from dataclasses import dataclass
import math
import os
import sys
from typing import Optional, Tuple

from geometry_msgs.msg import PointStamped
import rospy

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from contact_detector import ContactDetector
from ft_interface import WrenchData
from robot_interface import RobotInterface
from tf_interface import TFInterface


@dataclass(frozen=True)
class ProbeResult:
    success: bool
    contact_point: Optional[PointStamped]
    wrench_snapshot: Optional[WrenchData]
    reason: str


class WallProbe:
    """
    Focused probing helper for wall contact search.

    This class owns only probing behavior: move slowly, stop on contact,
    capture the contact point, and retract by a configured distance.
    """

    def __init__(
        self,
        robot_interface: RobotInterface,
        tf_interface: TFInterface,
        contact_detector: ContactDetector,
    ):
        self._robot_interface = robot_interface
        self._tf_interface = tf_interface
        self._contact_detector = contact_detector

        self._probe_speed = float(rospy.get_param("~motion/probe_speed", 0.005))
        self._retract_speed = float(rospy.get_param("~motion/retract_speed", 0.008))
        self._command_rate = float(rospy.get_param("~motion/command_rate", 500.0))
        self._retract_distance = float(rospy.get_param("~probe/retract_distance", 0.01))
        self._max_probe_distance = float(rospy.get_param("~probe/max_probe_distance", 0.08))
        self._probe_timeout = float(rospy.get_param("~probe/probe_timeout", 10.0))
        self._force_norm_threshold = float(rospy.get_param("~contact/force_threshold_norm", 4.0))

    def probe_until_contact(
        self,
        direction_xyz: Tuple[float, float, float],
        axis_name: str,
        threshold: float,
        probe_speed: Optional[float] = None,
        retract_distance: Optional[float] = None,
        max_travel_distance: Optional[float] = None,
        timeout: Optional[float] = None,
    ) -> ProbeResult:
        """
        Move along a Cartesian direction until contact is detected.

        The routine stops on contact, timeout, stale wrench input, missing TF,
        or maximum travel distance. After a valid contact it retracts along the
        opposite direction by the requested distance.
        """
        start_pose = self._tf_interface.get_tool_pose_in_base()
        if start_pose is None:
            self._robot_interface.stop_motion()
            return ProbeResult(False, None, None, "missing_initial_tf")

        direction = self._normalize_direction(direction_xyz)
        commanded_speed = self._probe_speed if probe_speed is None else float(probe_speed)
        retract_length = self._retract_distance if retract_distance is None else float(retract_distance)
        max_travel = self._max_probe_distance if max_travel_distance is None else float(max_travel_distance)
        deadline = rospy.Time.now() + rospy.Duration.from_sec(self._probe_timeout if timeout is None else float(timeout))
        rate = rospy.Rate(max(1.0, self._command_rate))

        rospy.loginfo(
            "[usb_c_insertion] event=probe_start axis=%s threshold=%.4f speed=%.4f max_travel=%.4f timeout=%.2f start_x=%.4f start_y=%.4f start_z=%.4f",
            axis_name,
            float(threshold),
            commanded_speed,
            max_travel,
            (self._probe_timeout if timeout is None else float(timeout)),
            start_pose.pose.position.x,
            start_pose.pose.position.y,
            start_pose.pose.position.z,
        )
        self._contact_detector.update_baseline()

        while not rospy.is_shutdown():
            if rospy.Time.now() > deadline:
                self._robot_interface.stop_motion()
                return self._log_and_return(False, None, self._get_wrench_snapshot(), "timeout")

            if self._is_wrench_stale():
                self._robot_interface.stop_motion()
                return self._log_and_return(False, None, self._get_wrench_snapshot(), "stale_wrench")

            current_pose = self._tf_interface.get_tool_pose_in_base()
            if current_pose is None:
                self._robot_interface.stop_motion()
                return self._log_and_return(False, None, self._get_wrench_snapshot(), "missing_tf")

            travel_distance = self._compute_distance(
                start_pose.pose.position.x,
                start_pose.pose.position.y,
                start_pose.pose.position.z,
                current_pose.pose.position.x,
                current_pose.pose.position.y,
                current_pose.pose.position.z,
            )
            if travel_distance >= max(0.0, max_travel):
                self._robot_interface.stop_motion()
                return self._log_and_return(False, None, self._get_wrench_snapshot(), "max_travel_reached")

            axis_delta = self._contact_detector.get_force_delta_along_axis(axis_name)
            norm_delta = self._contact_detector.get_force_delta_norm()
            rospy.loginfo_throttle(
                0.5,
                "[usb_c_insertion] event=probe_progress axis=%s axis_delta=%.4f axis_threshold=%.4f norm_delta=%.4f norm_threshold=%.4f travel=%.4f",
                axis_name,
                axis_delta,
                float(threshold),
                norm_delta,
                self._force_norm_threshold,
                travel_distance,
            )

            axis_contact = self._contact_detector.detect_contact_along_axis(axis_name, threshold)
            norm_contact = self._contact_detector.detect_contact_norm(self._force_norm_threshold)
            if axis_contact or norm_contact:
                self._robot_interface.stop_motion()
                contact_point = self._pose_to_point_stamped(current_pose)
                wrench_snapshot = self._get_wrench_snapshot()
                self._retract(direction, retract_length)
                return self._log_and_return(True, contact_point, wrench_snapshot, "contact_detected")

            self._robot_interface.send_twist(
                direction[0] * commanded_speed,
                direction[1] * commanded_speed,
                direction[2] * commanded_speed,
                0.0,
                0.0,
                0.0,
            )
            rate.sleep()

        self._robot_interface.stop_motion()
        return self._log_and_return(False, None, self._get_wrench_snapshot(), "shutdown")

    def _retract(self, direction_xyz: Tuple[float, float, float], retract_distance: float) -> None:
        """
        Retract along the opposite of the probe direction by a bounded distance.
        """
        if retract_distance <= 0.0:
            self._robot_interface.stop_motion()
            return

        start_pose = self._tf_interface.get_tool_pose_in_base()
        if start_pose is None:
            self._robot_interface.stop_motion()
            return

        opposite_direction = (-direction_xyz[0], -direction_xyz[1], -direction_xyz[2])
        rospy.loginfo(
            "[usb_c_insertion] event=probe_retract_start distance=%.4f speed=%.4f",
            retract_distance,
            self._retract_speed,
        )
        rate = rospy.Rate(max(1.0, self._command_rate))
        while not rospy.is_shutdown():
            current_pose = self._tf_interface.get_tool_pose_in_base()
            if current_pose is None:
                break

            travel_distance = self._compute_distance(
                start_pose.pose.position.x,
                start_pose.pose.position.y,
                start_pose.pose.position.z,
                current_pose.pose.position.x,
                current_pose.pose.position.y,
                current_pose.pose.position.z,
            )
            if travel_distance >= retract_distance:
                break

            self._robot_interface.send_twist(
                opposite_direction[0] * self._retract_speed,
                opposite_direction[1] * self._retract_speed,
                opposite_direction[2] * self._retract_speed,
                0.0,
                0.0,
                0.0,
            )
            rate.sleep()

        self._robot_interface.stop_motion()
        rospy.loginfo("[usb_c_insertion] event=probe_retract_complete")

    @staticmethod
    def _normalize_direction(direction_xyz: Tuple[float, float, float]) -> Tuple[float, float, float]:
        magnitude = math.sqrt(
            direction_xyz[0] * direction_xyz[0]
            + direction_xyz[1] * direction_xyz[1]
            + direction_xyz[2] * direction_xyz[2]
        )
        if magnitude <= 1e-9:
            raise ValueError("Probe direction must be non-zero.")
        return (
            direction_xyz[0] / magnitude,
            direction_xyz[1] / magnitude,
            direction_xyz[2] / magnitude,
        )

    @staticmethod
    def _compute_distance(x0: float, y0: float, z0: float, x1: float, y1: float, z1: float) -> float:
        return math.sqrt((x1 - x0) ** 2 + (y1 - y0) ** 2 + (z1 - z0) ** 2)

    @staticmethod
    def _pose_to_point_stamped(pose) -> PointStamped:
        point = PointStamped()
        point.header = pose.header
        point.point = pose.pose.position
        return point

    def _is_wrench_stale(self) -> bool:
        """
        Read stale state from the connected force-torque interface.
        """
        return self._contact_detector._ft_interface.is_wrench_stale()

    def _get_wrench_snapshot(self) -> WrenchData:
        """
        Return the current filtered wrench snapshot used by the detector.
        """
        return self._contact_detector._ft_interface.get_filtered_wrench()

    @staticmethod
    def _log_and_return(
        success: bool,
        contact_point: Optional[PointStamped],
        wrench_snapshot: Optional[WrenchData],
        reason: str,
    ) -> ProbeResult:
        if contact_point is None:
            rospy.loginfo(
                "[usb_c_insertion] event=probe_result success=%s reason=%s",
                str(success).lower(),
                reason,
            )
        else:
            if wrench_snapshot is None:
                rospy.loginfo(
                    "[usb_c_insertion] event=probe_result success=%s reason=%s contact_x=%.4f contact_y=%.4f contact_z=%.4f",
                    str(success).lower(),
                    reason,
                    contact_point.point.x,
                    contact_point.point.y,
                    contact_point.point.z,
                )
            else:
                rospy.loginfo(
                    "[usb_c_insertion] event=probe_result success=%s reason=%s contact_x=%.4f contact_y=%.4f contact_z=%.4f force_x=%.4f force_y=%.4f force_z=%.4f torque_x=%.4f torque_y=%.4f torque_z=%.4f",
                    str(success).lower(),
                    reason,
                    contact_point.point.x,
                    contact_point.point.y,
                    contact_point.point.z,
                    wrench_snapshot.force_x,
                    wrench_snapshot.force_y,
                    wrench_snapshot.force_z,
                    wrench_snapshot.torque_x,
                    wrench_snapshot.torque_y,
                    wrench_snapshot.torque_z,
                )
        return ProbeResult(success, contact_point, wrench_snapshot, reason)
