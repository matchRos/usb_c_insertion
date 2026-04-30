#!/usr/bin/env python3

from __future__ import annotations

import math
import os
import sys

from geometry_msgs.msg import Twist
import rospy
from std_msgs.msg import Bool

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from param_utils import required_float_param, required_str_param


class TwistControllerNode:
    """
    Smooth and watchdog-protect Twist commands before they reach the robot.

    The node applies synchronous scaling across all linear axes and all angular
    axes so commanded direction is preserved while speed and acceleration stay
    inside conservative limits.
    """

    def __init__(self):
        self._input_topic = required_str_param("~topics/raw_twist_cmd")
        self._output_topic = required_str_param("~topics/twist_cmd")
        self._micro_motion_active_topic = required_str_param("~topics/micro_motion_active")

        self._command_rate = required_float_param("~motion/command_rate")
        self._watchdog_timeout = required_float_param("~motion/watchdog_timeout")
        self._max_linear_speed = required_float_param("~motion/max_linear_speed")
        self._max_angular_speed = required_float_param("~motion/max_angular_speed")
        self._max_linear_acceleration = required_float_param("~motion/max_linear_acceleration")
        self._max_angular_acceleration = required_float_param("~motion/max_angular_acceleration")
        self._twist_alpha = required_float_param("~motion/twist_smoothing_alpha")

        self._target_twist = Twist()
        self._current_twist = Twist()
        self._last_command_time = rospy.Time(0)
        self._micro_motion_active = False

        self._publisher = rospy.Publisher(self._output_topic, Twist, queue_size=10)
        self._subscriber = rospy.Subscriber(self._input_topic, Twist, self._command_callback, queue_size=10)
        self._micro_motion_active_subscriber = rospy.Subscriber(
            self._micro_motion_active_topic,
            Bool,
            self._micro_motion_active_callback,
            queue_size=1,
        )
        rospy.on_shutdown(self._handle_shutdown)

    def spin(self) -> None:
        rate = rospy.Rate(max(1.0, self._command_rate))
        dt = 1.0 / max(1.0, self._command_rate)
        while not rospy.is_shutdown():
            if self._micro_motion_active:
                self._target_twist = Twist()
                self._current_twist = Twist()
                self._last_command_time = rospy.Time(0)
                rate.sleep()
                continue

            desired_twist = self._get_watchdog_safe_target()
            limited_target = self._apply_speed_limits(desired_twist)
            filtered_target = self._blend_twists(self._current_twist, limited_target, self._twist_alpha)
            self._current_twist = self._apply_acceleration_limits(self._current_twist, filtered_target, dt)
            self._publisher.publish(self._to_controller_frame(self._current_twist))
            rate.sleep()

    def _command_callback(self, msg: Twist) -> None:
        self._target_twist = msg
        self._last_command_time = rospy.Time.now()

    def _micro_motion_active_callback(self, msg: Bool) -> None:
        self._micro_motion_active = bool(msg.data)
        if self._micro_motion_active:
            self._target_twist = Twist()
            self._current_twist = Twist()
            self._last_command_time = rospy.Time(0)
            self._publisher.publish(self._to_controller_frame(Twist()))

    def _get_watchdog_safe_target(self) -> Twist:
        if self._last_command_time == rospy.Time(0):
            return Twist()
        age = (rospy.Time.now() - self._last_command_time).to_sec()
        if age > self._watchdog_timeout:
            return Twist()
        return self._target_twist

    def _apply_speed_limits(self, twist: Twist) -> Twist:
        limited = Twist()
        linear_scale = self._compute_group_scale(
            (twist.linear.x, twist.linear.y, twist.linear.z),
            self._max_linear_speed,
        )
        angular_scale = self._compute_group_scale(
            (twist.angular.x, twist.angular.y, twist.angular.z),
            self._max_angular_speed,
        )

        limited.linear.x = twist.linear.x * linear_scale
        limited.linear.y = twist.linear.y * linear_scale
        limited.linear.z = twist.linear.z * linear_scale
        limited.angular.x = twist.angular.x * angular_scale
        limited.angular.y = twist.angular.y * angular_scale
        limited.angular.z = twist.angular.z * angular_scale
        return limited

    def _apply_acceleration_limits(self, current: Twist, target: Twist, dt: float) -> Twist:
        result = Twist()
        linear_dx = target.linear.x - current.linear.x
        linear_dy = target.linear.y - current.linear.y
        linear_dz = target.linear.z - current.linear.z
        angular_dx = target.angular.x - current.angular.x
        angular_dy = target.angular.y - current.angular.y
        angular_dz = target.angular.z - current.angular.z

        linear_scale = self._compute_group_scale(
            (linear_dx, linear_dy, linear_dz),
            self._max_linear_acceleration * dt,
        )
        angular_scale = self._compute_group_scale(
            (angular_dx, angular_dy, angular_dz),
            self._max_angular_acceleration * dt,
        )

        result.linear.x = current.linear.x + linear_dx * linear_scale
        result.linear.y = current.linear.y + linear_dy * linear_scale
        result.linear.z = current.linear.z + linear_dz * linear_scale
        result.angular.x = current.angular.x + angular_dx * angular_scale
        result.angular.y = current.angular.y + angular_dy * angular_scale
        result.angular.z = current.angular.z + angular_dz * angular_scale
        return result

    @staticmethod
    def _blend_twists(current: Twist, target: Twist, alpha: float) -> Twist:
        clamped_alpha = max(0.0, min(1.0, alpha))
        result = Twist()
        result.linear.x = current.linear.x + clamped_alpha * (target.linear.x - current.linear.x)
        result.linear.y = current.linear.y + clamped_alpha * (target.linear.y - current.linear.y)
        result.linear.z = current.linear.z + clamped_alpha * (target.linear.z - current.linear.z)
        result.angular.x = current.angular.x + clamped_alpha * (target.angular.x - current.angular.x)
        result.angular.y = current.angular.y + clamped_alpha * (target.angular.y - current.angular.y)
        result.angular.z = current.angular.z + clamped_alpha * (target.angular.z - current.angular.z)
        return result

    @staticmethod
    def _compute_group_scale(components, limit: float) -> float:
        norm = math.sqrt(sum(component * component for component in components))
        if norm <= 1e-9:
            return 1.0
        return min(1.0, max(0.0, float(limit)) / norm)

    @staticmethod
    def _to_controller_frame(twist: Twist) -> Twist:
        """
        Convert the internal base_link twist convention to the controller convention.

        The controller expects x and y signs flipped relative to the planning
        frame used by the rest of this package. The same frame rotation must
        also be applied to angular velocity, so wx and wy change sign while wz
        remains unchanged.
        """
        converted = Twist()
        converted.linear.x = -twist.linear.x
        converted.linear.y = -twist.linear.y
        converted.linear.z = twist.linear.z
        converted.angular.x = -twist.angular.x
        converted.angular.y = -twist.angular.y
        converted.angular.z = twist.angular.z
        return converted

    def _handle_shutdown(self) -> None:
        """
        Publish explicit zero commands during shutdown so the robot stops.

        The watchdog usually catches missing commands, but sending a final stop
        command here reduces the chance of a short residual motion if this node
        is terminated while non-zero twists are still buffered downstream.
        """
        zero_twist = self._to_controller_frame(Twist())
        for _ in range(10):
            self._publisher.publish(zero_twist)
            rospy.sleep(0.01)


def main() -> None:
    rospy.init_node("usb_c_insertion_twist_controller")
    TwistControllerNode().spin()


if __name__ == "__main__":
    main()
