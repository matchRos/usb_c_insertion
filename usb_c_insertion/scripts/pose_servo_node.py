#!/usr/bin/env python3

from __future__ import annotations

import math
import os
import sys
from typing import Optional

from geometry_msgs.msg import PoseStamped
import rospy
from std_msgs.msg import Bool

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from robot_interface import RobotInterface
from tf_interface import TFInterface
from usb_c_insertion.msg import PoseServoStatus


class PoseServoNode:
    """
    Pose servo controller that moves smoothly to a requested target pose.

    The node is enabled only while a target is active. It computes Cartesian
    Twist requests from TF pose error and sends them into the smoothing node.
    """

    def __init__(self):
        self._target_topic = rospy.get_param("~topics/pose_target", "/usb_c_insertion/pose_target")
        self._enable_topic = rospy.get_param("~topics/pose_servo_enable", "/usb_c_insertion/pose_servo_enable")
        self._status_topic = rospy.get_param("~topics/pose_servo_status", "/usb_c_insertion/pose_servo_status")

        self._command_rate = float(rospy.get_param("~motion/command_rate", 500.0))
        self._position_gain = float(rospy.get_param("~motion/pose_servo_position_gain", 1.5))
        self._orientation_gain = float(rospy.get_param("~motion/pose_servo_orientation_gain", 1.0))
        self._position_tolerance = float(rospy.get_param("~motion/pose_servo_position_tolerance", 0.0005))
        self._orientation_tolerance = float(rospy.get_param("~motion/pose_servo_orientation_tolerance", 0.03))
        self._max_linear_speed = float(rospy.get_param("~motion/max_linear_speed", 0.05))
        self._max_angular_speed = float(rospy.get_param("~motion/max_angular_speed", 0.25))

        self._tf = TFInterface()
        self._robot = RobotInterface()
        self._enabled = False
        self._target_pose: Optional[PoseStamped] = None
        self._target_stamp = rospy.Time(0)
        self._zero_twist_sent = False
        self._goal_reached_latched = False
        self._status_publisher = rospy.Publisher(self._status_topic, PoseServoStatus, queue_size=10)

        self._target_subscriber = rospy.Subscriber(self._target_topic, PoseStamped, self._target_callback, queue_size=1)
        self._enable_subscriber = rospy.Subscriber(self._enable_topic, Bool, self._enable_callback, queue_size=1)

    def spin(self) -> None:
        rate = rospy.Rate(max(1.0, self._command_rate))
        while not rospy.is_shutdown():
            if not self._enabled:
                self._publish_status(None, 0.0, 0.0, self._goal_reached_latched)
                self._send_zero_twist_once()
                rate.sleep()
                continue

            if self._target_pose is None:
                self._publish_status(None, 0.0, 0.0, False)
                rate.sleep()
                continue

            current_pose = self._tf.get_tool_pose_in_base()
            if current_pose is None:
                rospy.logwarn_throttle(
                    1.0,
                    "[usb_c_insertion] event=pose_servo_missing_tf",
                )
                self._publish_status(None, 0.0, 0.0, False)
                rate.sleep()
                continue

            target = self._target_pose.pose
            error_x = target.position.x - current_pose.pose.position.x
            error_y = target.position.y - current_pose.pose.position.y
            error_z = target.position.z - current_pose.pose.position.z
            distance = math.sqrt(error_x * error_x + error_y * error_y + error_z * error_z)

            orientation_error = self._quaternion_error_vector(
                current_pose.pose.orientation,
                target.orientation,
            )
            orientation_error_norm = math.sqrt(
                orientation_error[0] * orientation_error[0]
                + orientation_error[1] * orientation_error[1]
                + orientation_error[2] * orientation_error[2]
            )

            if distance <= self._position_tolerance and orientation_error_norm <= self._orientation_tolerance:
                self._goal_reached_latched = True
                self._publish_status(current_pose, distance, orientation_error_norm, True)
                rospy.loginfo(
                    "[usb_c_insertion] event=pose_servo_disabled_internal reason=tolerance_reached pos_err=%.4f ori_err=%.4f",
                    distance,
                    orientation_error_norm,
                )
                self._enabled = False
                self._send_zero_twist_once()
                rate.sleep()
                continue

            self._publish_status(current_pose, distance, orientation_error_norm, False)

            linear_speed = min(self._max_linear_speed, self._position_gain * distance)
            if distance <= 1e-9:
                vx = 0.0
                vy = 0.0
                vz = 0.0
            else:
                scale = linear_speed / distance
                vx = error_x * scale
                vy = error_y * scale
                vz = error_z * scale

            angular_velocity = self._limit_angular_vector(
                (
                    self._orientation_gain * orientation_error[0],
                    self._orientation_gain * orientation_error[1],
                    self._orientation_gain * orientation_error[2],
                )
            )
            rospy.loginfo_throttle(
                1.0,
                "[usb_c_insertion] event=pose_servo_cmd vx=%.4f vy=%.4f vz=%.4f wx=%.4f wy=%.4f wz=%.4f pos_err=%.4f ori_err=%.4f",
                vx,
                vy,
                vz,
                angular_velocity[0],
                angular_velocity[1],
                angular_velocity[2],
                distance,
                orientation_error_norm,
            )
            self._robot.send_twist(vx, vy, vz, angular_velocity[0], angular_velocity[1], angular_velocity[2])
            self._zero_twist_sent = False
            rate.sleep()

    def _target_callback(self, msg: PoseStamped) -> None:
        self._target_pose = msg
        self._target_stamp = rospy.Time.now()
        self._zero_twist_sent = False
        self._goal_reached_latched = False

    def _enable_callback(self, msg: Bool) -> None:
        if self._enabled and not bool(msg.data):
            rospy.logwarn(
                "[usb_c_insertion] event=pose_servo_disabled_external"
            )
        self._enabled = bool(msg.data)
        if self._enabled:
            self._zero_twist_sent = False
            self._goal_reached_latched = False

    def _publish_status(
        self,
        current_pose: Optional[PoseStamped],
        position_error: float,
        orientation_error: float,
        goal_reached: bool,
    ) -> None:
        status = PoseServoStatus()
        status.header.stamp = rospy.Time.now()
        status.enabled = bool(self._enabled)
        status.has_target = self._target_pose is not None
        status.goal_reached = bool(goal_reached)
        status.position_error = float(position_error)
        status.orientation_error = float(orientation_error)
        if current_pose is not None:
            status.current_pose = current_pose
        self._status_publisher.publish(status)

    def _send_zero_twist_once(self) -> None:
        """
        Send a single explicit stop command when servoing becomes inactive.

        The downstream twist controller already has a watchdog, so repeatedly
        publishing zero here only creates command chattering.
        """
        if self._zero_twist_sent:
            return
        self._robot.send_zero_twist()
        self._zero_twist_sent = True

    def _limit_angular_vector(self, angular_xyz):
        norm = math.sqrt(sum(component * component for component in angular_xyz))
        if norm <= 1e-9:
            return (0.0, 0.0, 0.0)
        scale = min(1.0, self._max_angular_speed / norm)
        return tuple(component * scale for component in angular_xyz)

    @staticmethod
    def _quaternion_error_vector(current_orientation, target_orientation):
        current = PoseServoNode._normalize_quaternion(
            (
                current_orientation.x,
                current_orientation.y,
                current_orientation.z,
                current_orientation.w,
            )
        )
        target = PoseServoNode._normalize_quaternion(
            (
                target_orientation.x,
                target_orientation.y,
                target_orientation.z,
                target_orientation.w,
            )
        )

        error = PoseServoNode._quaternion_multiply(target, PoseServoNode._quaternion_conjugate(current))
        if error[3] < 0.0:
            error = (-error[0], -error[1], -error[2], -error[3])

        vector_norm = math.sqrt(error[0] * error[0] + error[1] * error[1] + error[2] * error[2])
        if vector_norm <= 1e-9:
            return (0.0, 0.0, 0.0)

        angle = 2.0 * math.atan2(vector_norm, error[3])
        axis = (error[0] / vector_norm, error[1] / vector_norm, error[2] / vector_norm)
        return (axis[0] * angle, axis[1] * angle, axis[2] * angle)

    @staticmethod
    def _normalize_quaternion(quaternion_xyzw):
        norm = math.sqrt(sum(component * component for component in quaternion_xyzw))
        if norm <= 1e-9:
            return (0.0, 0.0, 0.0, 1.0)
        return tuple(component / norm for component in quaternion_xyzw)

    @staticmethod
    def _quaternion_conjugate(quaternion_xyzw):
        return (-quaternion_xyzw[0], -quaternion_xyzw[1], -quaternion_xyzw[2], quaternion_xyzw[3])

    @staticmethod
    def _quaternion_multiply(first_xyzw, second_xyzw):
        x1, y1, z1, w1 = first_xyzw
        x2, y2, z2, w2 = second_xyzw
        return (
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        )


def main() -> None:
    rospy.init_node("usb_c_insertion_pose_servo")
    PoseServoNode().spin()


if __name__ == "__main__":
    main()
