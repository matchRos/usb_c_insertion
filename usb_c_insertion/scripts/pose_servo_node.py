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


class PoseServoNode:
    """
    Pose servo controller that moves smoothly to a requested target pose.

    The node is enabled only while a target is active. It computes Cartesian
    Twist requests from TF pose error and sends them into the smoothing node.
    """

    def __init__(self):
        self._target_topic = rospy.get_param("~topics/pose_target", "/usb_c_insertion/pose_target")
        self._enable_topic = rospy.get_param("~topics/pose_servo_enable", "/usb_c_insertion/pose_servo_enable")

        self._command_rate = float(rospy.get_param("~motion/command_rate", 500.0))
        self._position_gain = float(rospy.get_param("~motion/pose_servo_position_gain", 1.5))
        self._yaw_gain = float(rospy.get_param("~motion/pose_servo_yaw_gain", 1.0))
        self._position_tolerance = float(rospy.get_param("~motion/pose_servo_position_tolerance", 0.0015))
        self._yaw_tolerance = float(rospy.get_param("~motion/pose_servo_yaw_tolerance", 0.03))
        self._max_linear_speed = float(rospy.get_param("~motion/max_linear_speed", 0.01))
        self._max_angular_speed = float(rospy.get_param("~motion/max_angular_speed", 0.05))

        self._tf = TFInterface()
        self._robot = RobotInterface()
        self._enabled = False
        self._target_pose: Optional[PoseStamped] = None
        self._target_stamp = rospy.Time(0)
        self._zero_twist_sent = False

        self._target_subscriber = rospy.Subscriber(self._target_topic, PoseStamped, self._target_callback, queue_size=1)
        self._enable_subscriber = rospy.Subscriber(self._enable_topic, Bool, self._enable_callback, queue_size=1)

    def spin(self) -> None:
        rate = rospy.Rate(max(1.0, self._command_rate))
        while not rospy.is_shutdown():
            if not self._enabled:
                self._send_zero_twist_once()
                rate.sleep()
                continue

            if self._target_pose is None:
                rate.sleep()
                continue

            current_pose = self._tf.get_tool_pose_in_base()
            if current_pose is None:
                rate.sleep()
                continue

            target = self._target_pose.pose
            error_x = target.position.x - current_pose.pose.position.x
            error_y = target.position.y - current_pose.pose.position.y
            error_z = target.position.z - current_pose.pose.position.z
            distance = math.sqrt(error_x * error_x + error_y * error_y + error_z * error_z)

            current_yaw = self._yaw_from_quaternion(current_pose.pose.orientation)
            target_yaw = self._yaw_from_quaternion(target.orientation)
            yaw_error = self._normalize_angle(target_yaw - current_yaw)

            if distance <= self._position_tolerance and abs(yaw_error) <= self._yaw_tolerance:
                self._enabled = False
                self._send_zero_twist_once()
                rate.sleep()
                continue

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

            wz = max(-self._max_angular_speed, min(self._max_angular_speed, self._yaw_gain * yaw_error))
            self._robot.send_twist(vx, vy, vz, 0.0, 0.0, wz)
            self._zero_twist_sent = False
            rate.sleep()

    def _target_callback(self, msg: PoseStamped) -> None:
        self._target_pose = msg
        self._target_stamp = rospy.Time.now()
        self._zero_twist_sent = False

    def _enable_callback(self, msg: Bool) -> None:
        self._enabled = bool(msg.data)
        if self._enabled:
            self._zero_twist_sent = False

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

    @staticmethod
    def _yaw_from_quaternion(quaternion) -> float:
        siny_cosp = 2.0 * (quaternion.w * quaternion.z + quaternion.x * quaternion.y)
        cosy_cosp = 1.0 - 2.0 * (quaternion.y * quaternion.y + quaternion.z * quaternion.z)
        return math.atan2(siny_cosp, cosy_cosp)

    @staticmethod
    def _normalize_angle(angle: float) -> float:
        while angle > math.pi:
            angle -= 2.0 * math.pi
        while angle < -math.pi:
            angle += 2.0 * math.pi
        return angle


def main() -> None:
    rospy.init_node("usb_c_insertion_pose_servo")
    PoseServoNode().spin()


if __name__ == "__main__":
    main()
