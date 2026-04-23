#!/usr/bin/env python3

from __future__ import annotations

import math
import os
import sys
from typing import Optional

import actionlib
from geometry_msgs.msg import PoseStamped
import rospy

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from robot_interface import RobotInterface
from tf_interface import TFInterface
from usb_c_insertion.msg import (
    MoveToPoseAction,
    MoveToPoseFeedback,
    MoveToPoseResult,
)


class MoveToPoseActionServer:
    """
    Small action server that exposes pose servoing to an external workflow.

    The server publishes a pose target, enables the existing pose servo node,
    and reports completion once the robot is both inside tolerance and settled
    for a short dwell time.
    """

    def __init__(self):
        self._action_name = str(rospy.get_param("~action_name", "move_to_pose")).strip()
        self._default_position_tolerance = float(
            rospy.get_param("~motion/action_position_tolerance", 0.002)
        )
        self._default_orientation_tolerance = float(
            rospy.get_param("~motion/action_orientation_tolerance", 0.05)
        )
        self._default_settle_time = float(
            rospy.get_param("~motion/action_settle_time", 0.4)
        )
        self._default_timeout = float(
            rospy.get_param("~motion/action_timeout", 30.0)
        )
        self._feedback_rate = float(
            rospy.get_param("~motion/action_feedback_rate", 20.0)
        )
        self._pipeline_wait_timeout = float(
            rospy.get_param("~motion/action_pipeline_wait_timeout", 2.0)
        )

        self._robot = RobotInterface()
        self._tf = TFInterface()
        self._server = actionlib.SimpleActionServer(
            self._action_name,
            MoveToPoseAction,
            execute_cb=self._execute,
            auto_start=False,
        )
        self._server.start()
        rospy.loginfo(
            "[usb_c_insertion] event=move_to_pose_action_ready action=%s",
            self._action_name,
        )

    def _execute(self, goal) -> None:
        if not self._robot.wait_for_motion_pipeline(
            timeout=self._pipeline_wait_timeout,
            require_pose_servo=True,
        ):
            self._robot.stop_motion()
            self._set_aborted("motion_pipeline_not_ready", None)
            return

        timeout = self._goal_or_default(goal.timeout, self._default_timeout)
        position_tolerance = self._goal_or_default(
            goal.position_tolerance, self._default_position_tolerance
        )
        orientation_tolerance = self._goal_or_default(
            goal.orientation_tolerance, self._default_orientation_tolerance
        )
        settle_time = self._goal_or_default(goal.settle_time, self._default_settle_time)

        self._robot.send_pose_target(
            goal.target_pose.pose.position.x,
            goal.target_pose.pose.position.y,
            goal.target_pose.pose.position.z,
            goal.target_pose.pose.orientation.x,
            goal.target_pose.pose.orientation.y,
            goal.target_pose.pose.orientation.z,
            goal.target_pose.pose.orientation.w,
            frame_id=goal.target_pose.header.frame_id,
        )
        self._robot.enable_pose_servo(True)

        rate = rospy.Rate(max(1.0, self._feedback_rate))
        deadline = rospy.Time.now() + rospy.Duration.from_sec(max(0.1, timeout))
        settled_since: Optional[rospy.Time] = None

        while not rospy.is_shutdown():
            if self._server.is_preempt_requested():
                self._robot.stop_motion()
                self._server.set_preempted(self._make_result(False, "preempted", None))
                return

            current_pose = self._tf.get_tool_pose_in_base()
            if current_pose is None:
                rate.sleep()
                continue

            position_error = self._position_error(goal.target_pose, current_pose)
            orientation_error = self._orientation_error(goal.target_pose, current_pose)
            reached_position = position_error <= position_tolerance
            reached_orientation = orientation_error <= orientation_tolerance

            feedback = MoveToPoseFeedback()
            feedback.current_pose = current_pose
            feedback.position_error = position_error
            feedback.orientation_error = orientation_error
            feedback.reached_position = reached_position
            feedback.reached_orientation = reached_orientation
            self._server.publish_feedback(feedback)

            if reached_position and reached_orientation:
                if settled_since is None:
                    settled_since = rospy.Time.now()
                elif (rospy.Time.now() - settled_since).to_sec() >= settle_time:
                    self._robot.enable_pose_servo(False)
                    self._server.set_succeeded(
                        self._make_result(True, "target_reached", current_pose)
                    )
                    return
            else:
                settled_since = None

            if rospy.Time.now() >= deadline:
                self._robot.stop_motion()
                self._set_aborted("move_to_pose_timeout", current_pose)
                return

            rate.sleep()

        self._robot.stop_motion()

    def _set_aborted(self, message: str, pose: Optional[PoseStamped]) -> None:
        self._server.set_aborted(self._make_result(False, message, pose))

    @staticmethod
    def _goal_or_default(value: float, default: float) -> float:
        return float(value) if float(value) > 0.0 else float(default)

    @staticmethod
    def _make_result(success: bool, message: str, pose: Optional[PoseStamped]) -> MoveToPoseResult:
        result = MoveToPoseResult()
        result.success = bool(success)
        result.message = str(message)
        if pose is not None:
            result.final_pose = pose
        return result

    @staticmethod
    def _position_error(target_pose: PoseStamped, current_pose: PoseStamped) -> float:
        dx = target_pose.pose.position.x - current_pose.pose.position.x
        dy = target_pose.pose.position.y - current_pose.pose.position.y
        dz = target_pose.pose.position.z - current_pose.pose.position.z
        return math.sqrt(dx * dx + dy * dy + dz * dz)

    @staticmethod
    def _orientation_error(target_pose: PoseStamped, current_pose: PoseStamped) -> float:
        error = MoveToPoseActionServer._quaternion_error_vector(
            current_pose.pose.orientation,
            target_pose.pose.orientation,
        )
        return math.sqrt(error[0] * error[0] + error[1] * error[1] + error[2] * error[2])

    @staticmethod
    def _quaternion_error_vector(current_orientation, target_orientation):
        current = MoveToPoseActionServer._normalize_quaternion(
            (
                current_orientation.x,
                current_orientation.y,
                current_orientation.z,
                current_orientation.w,
            )
        )
        target = MoveToPoseActionServer._normalize_quaternion(
            (
                target_orientation.x,
                target_orientation.y,
                target_orientation.z,
                target_orientation.w,
            )
        )

        error = MoveToPoseActionServer._quaternion_multiply(
            target,
            MoveToPoseActionServer._quaternion_conjugate(current),
        )
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
    rospy.init_node("usb_c_insertion_move_to_pose_action_server")
    MoveToPoseActionServer()
    rospy.spin()


if __name__ == "__main__":
    main()
