#!/usr/bin/env python3

from __future__ import annotations

import math
import os
import sys
from typing import Optional, Tuple

import actionlib
from geometry_msgs.msg import PoseStamped, Twist
import rospy
from std_msgs.msg import Bool

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from robot_interface import RobotInterface
from tf_interface import TFInterface
from usb_c_insertion.msg import (
    MicroMoveAction,
    MicroMoveFeedback,
    MicroMoveResult,
)


class MicroMoveActionServer:
    """
    Execute very short Cartesian moves with a direct jerk-continuous velocity profile.

    This action intentionally bypasses the normal twist smoothing node. It owns
    speed, acceleration, and jerk limits itself and raises a micro-motion-active
    flag so the regular twist controller stops publishing during the move.
    """

    _MAX_SMOOTHSTEP_VELOCITY = 2.1875
    _MAX_SMOOTHSTEP_ACCELERATION = 7.5131884044
    _MAX_SMOOTHSTEP_JERK = 52.5

    def __init__(self):
        self._action_name = str(rospy.get_param("~action_name", "micro_move")).strip()
        self._base_frame = str(rospy.get_param("~frames/base_frame", "base_link")).strip()
        self._twist_topic = str(rospy.get_param("~topics/twist_cmd", "/twist_controller/command")).strip()
        self._active_topic = str(
            rospy.get_param("~topics/micro_motion_active", "/usb_c_insertion/micro_motion_active")
        ).strip()

        self._command_rate = float(rospy.get_param("~motion/command_rate", 500.0))
        self._max_distance = float(rospy.get_param("~micro_motion/max_distance", 0.02))
        self._default_max_velocity = float(rospy.get_param("~micro_motion/max_velocity", 0.04))
        self._default_max_acceleration = float(rospy.get_param("~micro_motion/max_acceleration", 0.5))
        self._default_max_jerk = float(rospy.get_param("~micro_motion/max_jerk", 20.0))
        self._min_duration = float(rospy.get_param("~micro_motion/min_duration", 0.04))
        self._max_duration = float(rospy.get_param("~micro_motion/max_duration", 1.0))
        self._default_monitor_tf = bool(rospy.get_param("~micro_motion/monitor_tf", True))
        self._max_overshoot = float(rospy.get_param("~micro_motion/max_overshoot", 0.001))

        self._tf = TFInterface()
        self._robot = RobotInterface()
        self._publisher = rospy.Publisher(self._twist_topic, Twist, queue_size=10)
        self._active_publisher = rospy.Publisher(self._active_topic, Bool, queue_size=1, latch=True)
        self._server = actionlib.SimpleActionServer(
            self._action_name,
            MicroMoveAction,
            execute_cb=self._execute,
            auto_start=False,
        )
        rospy.on_shutdown(self._handle_shutdown)
        self._active_publisher.publish(Bool(data=False))
        self._server.start()
        rospy.loginfo("[usb_c_insertion] event=micro_move_action_ready action=%s", self._action_name)

    def _execute(self, goal) -> None:
        displacement = (
            float(goal.displacement.x),
            float(goal.displacement.y),
            float(goal.displacement.z),
        )
        distance = self._norm(displacement)
        if distance <= 1e-9:
            self._server.set_succeeded(self._make_result(True, "zero_distance", "", 0.0, 0.0, 0.0))
            return
        if distance > self._max_distance:
            self._server.set_aborted(
                self._make_result(
                    False,
                    "distance_exceeds_limit",
                    "distance_exceeds_limit",
                    distance,
                    0.0,
                    0.0,
                )
            )
            return

        direction = tuple(component / distance for component in displacement)
        max_velocity = self._goal_or_default(goal.max_velocity, self._default_max_velocity)
        max_acceleration = self._goal_or_default(goal.max_acceleration, self._default_max_acceleration)
        max_jerk = self._goal_or_default(goal.max_jerk, self._default_max_jerk)
        monitor_tf = self._default_monitor_tf if not bool(goal.monitor_tf) else bool(goal.monitor_tf)
        duration = self._compute_duration(distance, max_velocity, max_acceleration, max_jerk)
        timeout = self._goal_or_default(goal.timeout, duration + 0.25)

        if duration > self._max_duration:
            self._server.set_aborted(
                self._make_result(
                    False,
                    "duration_exceeds_limit",
                    "duration_exceeds_limit",
                    distance,
                    0.0,
                    duration,
                )
            )
            return

        start_pose = self._tf.get_tool_pose_in_base() if monitor_tf else None
        start_xyz = self._pose_xyz(start_pose) if start_pose is not None else None

        rospy.loginfo(
            "[usb_c_insertion] event=micro_move_start distance=%.5f duration=%.4f vmax=%.3f amax=%.3f jmax=%.3f dx=%.5f dy=%.5f dz=%.5f",
            distance,
            duration,
            max_velocity,
            max_acceleration,
            max_jerk,
            displacement[0],
            displacement[1],
            displacement[2],
        )

        started_at = rospy.Time.now()
        deadline = started_at + rospy.Duration.from_sec(max(timeout, duration + 0.05))
        measured_distance = 0.0
        self._set_micro_motion_active(True)

        try:
            rate = rospy.Rate(max(1.0, self._command_rate))
            while not rospy.is_shutdown():
                if self._server.is_preempt_requested():
                    self._publish_zero_twist()
                    self._server.set_preempted(
                        self._make_result(False, "preempted", "preempted", distance, measured_distance, duration)
                    )
                    return

                now = rospy.Time.now()
                elapsed = (now - started_at).to_sec()
                if now > deadline:
                    self._publish_zero_twist()
                    self._server.set_aborted(
                        self._make_result(
                            False,
                            "micro_move_timeout",
                            "micro_move_timeout",
                            distance,
                            measured_distance,
                            duration,
                        )
                    )
                    return

                if elapsed >= duration:
                    break

                u = self._clamp(elapsed / duration, 0.0, 1.0)
                commanded_speed = distance * self._smoothstep7_derivative(u) / duration
                twist = Twist()
                twist.linear.x = direction[0] * commanded_speed
                twist.linear.y = direction[1] * commanded_speed
                twist.linear.z = direction[2] * commanded_speed
                self._publisher.publish(self._to_controller_frame(twist))

                if start_xyz is not None:
                    current_pose = self._tf.get_tool_pose_in_base()
                    if current_pose is not None:
                        measured_distance = self._project_displacement(start_xyz, direction, self._pose_xyz(current_pose))
                        if measured_distance > distance + self._max_overshoot:
                            self._publish_zero_twist()
                            self._server.set_aborted(
                                self._make_result(
                                    False,
                                    "micro_move_overshoot",
                                    "micro_move_overshoot",
                                    distance,
                                    measured_distance,
                                    duration,
                                )
                            )
                            return

                self._publish_feedback(elapsed, duration, distance, measured_distance, commanded_speed)
                rate.sleep()

            self._publish_zero_twist()
            final_pose = self._tf.get_tool_pose_in_base()
            if start_xyz is not None and final_pose is not None:
                measured_distance = self._project_displacement(start_xyz, direction, self._pose_xyz(final_pose))
            result = self._make_result(True, "micro_move_complete", "", distance, measured_distance, duration)
            if final_pose is not None:
                result.final_pose = final_pose
            self._server.set_succeeded(result)
        finally:
            self._publish_zero_twist()
            self._set_micro_motion_active(False)

    def _compute_duration(self, distance: float, max_velocity: float, max_acceleration: float, max_jerk: float) -> float:
        duration = max(0.0, self._min_duration)
        if max_velocity > 0.0:
            duration = max(duration, distance * self._MAX_SMOOTHSTEP_VELOCITY / max_velocity)
        if max_acceleration > 0.0:
            duration = max(duration, math.sqrt(distance * self._MAX_SMOOTHSTEP_ACCELERATION / max_acceleration))
        if max_jerk > 0.0:
            duration = max(duration, (distance * self._MAX_SMOOTHSTEP_JERK / max_jerk) ** (1.0 / 3.0))
        return duration

    def _publish_feedback(
        self,
        elapsed: float,
        duration: float,
        commanded_distance: float,
        measured_distance: float,
        commanded_speed: float,
    ) -> None:
        feedback = MicroMoveFeedback()
        feedback.stage = "moving"
        feedback.elapsed = float(elapsed)
        feedback.duration = float(duration)
        feedback.commanded_distance = float(commanded_distance)
        feedback.measured_distance = float(measured_distance)
        feedback.commanded_speed = float(commanded_speed)
        self._server.publish_feedback(feedback)

    def _make_result(
        self,
        success: bool,
        message: str,
        error_code: str,
        commanded_distance: float,
        measured_distance: float,
        duration: float,
    ) -> MicroMoveResult:
        result = MicroMoveResult()
        result.success = bool(success)
        result.message = str(message)
        result.error_code = str(error_code)
        result.commanded_distance = float(commanded_distance)
        result.measured_distance = float(measured_distance)
        result.duration = float(duration)
        result.final_pose.header.stamp = rospy.Time.now()
        result.final_pose.header.frame_id = self._base_frame
        return result

    def _set_micro_motion_active(self, active: bool) -> None:
        self._robot.enable_pose_servo(False)
        self._active_publisher.publish(Bool(data=bool(active)))
        rospy.sleep(0.01)

    def _publish_zero_twist(self) -> None:
        zero_twist = self._to_controller_frame(Twist())
        for _ in range(3):
            self._publisher.publish(zero_twist)

    def _handle_shutdown(self) -> None:
        self._publish_zero_twist()
        self._set_micro_motion_active(False)

    @staticmethod
    def _smoothstep7_derivative(u: float) -> float:
        return 140.0 * u ** 3 - 420.0 * u ** 4 + 420.0 * u ** 5 - 140.0 * u ** 6

    @staticmethod
    def _goal_or_default(value: float, default: float) -> float:
        return float(value) if float(value) > 0.0 else float(default)

    @staticmethod
    def _norm(vector_xyz) -> float:
        return math.sqrt(sum(component * component for component in vector_xyz))

    @staticmethod
    def _pose_xyz(pose: Optional[PoseStamped]) -> Tuple[float, float, float]:
        return (
            pose.pose.position.x,
            pose.pose.position.y,
            pose.pose.position.z,
        )

    @staticmethod
    def _project_displacement(start_xyz, direction_xyz, current_xyz) -> float:
        delta = (
            current_xyz[0] - start_xyz[0],
            current_xyz[1] - start_xyz[1],
            current_xyz[2] - start_xyz[2],
        )
        return sum(delta[index] * direction_xyz[index] for index in range(3))

    @staticmethod
    def _to_controller_frame(twist: Twist) -> Twist:
        converted = Twist()
        converted.linear.x = -twist.linear.x
        converted.linear.y = -twist.linear.y
        converted.linear.z = twist.linear.z
        converted.angular.x = -twist.angular.x
        converted.angular.y = -twist.angular.y
        converted.angular.z = twist.angular.z
        return converted

    @staticmethod
    def _clamp(value: float, lower: float, upper: float) -> float:
        return max(lower, min(upper, value))


def main() -> None:
    rospy.init_node("usb_c_insertion_micro_move_action_server")
    MicroMoveActionServer()
    rospy.spin()


if __name__ == "__main__":
    main()
