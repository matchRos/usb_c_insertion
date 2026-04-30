#!/usr/bin/env python3

from __future__ import annotations

import os
import sys
from typing import Optional

import actionlib
from geometry_msgs.msg import PoseStamped
import rospy

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from param_utils import required_bool_param, required_float_param, required_str_param
from robot_interface import RobotInterface
from usb_c_insertion.msg import (
    MoveToPoseAction,
    MoveToPoseFeedback,
    MoveToPoseResult,
    PoseServoStatus,
)


class MoveToPoseActionServer:
    """
    Small action server that exposes pose servoing to an external workflow.

    The server publishes a pose target, enables the existing pose servo node,
    and reports completion once the robot is both inside tolerance and settled
    for a short dwell time.
    """

    def __init__(self):
        self._action_name = "move_to_pose"
        self._base_frame = required_str_param("~frames/base_frame")
        self._default_settle_time = required_float_param("~motion/action_settle_time")
        self._default_timeout = required_float_param("~motion/action_timeout")
        self._feedback_rate = required_float_param("~motion/action_feedback_rate")
        self._pipeline_wait_timeout = required_float_param("~motion/action_pipeline_wait_timeout")
        self._enforce_workspace_limits = required_bool_param("~motion/enforce_workspace_limits")
        self._min_target_x = required_float_param("~motion/min_target_x")
        self._min_target_z = required_float_param("~motion/min_target_z")
        self._status_topic = required_str_param("~topics/pose_servo_status")

        self._robot = RobotInterface()
        self._latest_status: Optional[PoseServoStatus] = None
        self._status_subscriber = rospy.Subscriber(
            self._status_topic,
            PoseServoStatus,
            self._status_callback,
            queue_size=10,
        )
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
        frame_id = goal.target_pose.header.frame_id.strip() or self._base_frame
        if frame_id != self._base_frame:
            self._robot.stop_motion()
            self._set_aborted("unsupported_target_pose_frame", None)
            return

        if not self._is_target_allowed(goal.target_pose):
            self._robot.stop_motion()
            self._set_aborted("target_pose_not_allowed", None)
            return

        if not self._robot.wait_for_motion_pipeline(
            timeout=self._pipeline_wait_timeout,
            require_pose_servo=True,
        ):
            self._robot.stop_motion()
            self._set_aborted("motion_pipeline_not_ready", None)
            return

        timeout = self._goal_or_default(goal.timeout, self._default_timeout)
        settle_time = self._goal_or_default(goal.settle_time, self._default_settle_time)

        self._robot.send_pose_target(
            goal.target_pose.pose.position.x,
            goal.target_pose.pose.position.y,
            goal.target_pose.pose.position.z,
            goal.target_pose.pose.orientation.x,
            goal.target_pose.pose.orientation.y,
            goal.target_pose.pose.orientation.z,
            goal.target_pose.pose.orientation.w,
            frame_id=frame_id,
        )
        self._robot.enable_pose_servo(True)

        rate = rospy.Rate(max(1.0, self._feedback_rate))
        deadline = rospy.Time.now() + rospy.Duration.from_sec(max(0.1, timeout))
        goal_start_time = rospy.Time.now()
        settled_since: Optional[rospy.Time] = None

        while not rospy.is_shutdown():
            if self._server.is_preempt_requested():
                self._robot.stop_motion()
                self._server.set_preempted(self._make_result(False, "preempted", "preempted", None))
                return

            status = self._latest_status
            if status is None or status.header.stamp < goal_start_time:
                rate.sleep()
                continue

            feedback = MoveToPoseFeedback()
            feedback.current_pose = status.current_pose
            feedback.position_error = status.position_error
            feedback.orientation_error = status.orientation_error
            feedback.reached_position = bool(status.goal_reached)
            feedback.reached_orientation = bool(status.goal_reached)
            self._server.publish_feedback(feedback)

            if bool(status.goal_reached):
                if settled_since is None:
                    settled_since = rospy.Time.now()
                elif (rospy.Time.now() - settled_since).to_sec() >= settle_time:
                    self._robot.enable_pose_servo(False)
                    self._server.set_succeeded(
                        self._make_result(True, "target_reached", "", status.current_pose)
                    )
                    return
            else:
                settled_since = None

            if rospy.Time.now() >= deadline:
                self._robot.stop_motion()
                self._set_aborted(
                    "move_to_pose_timeout",
                    status.current_pose if status is not None else None,
                )
                return

            rate.sleep()

        self._robot.stop_motion()

    def _set_aborted(self, message: str, pose: Optional[PoseStamped]) -> None:
        self._server.set_aborted(self._make_result(False, message, message, pose))

    def _status_callback(self, msg: PoseServoStatus) -> None:
        self._latest_status = msg

    @staticmethod
    def _goal_or_default(value: float, default: float) -> float:
        return float(value) if float(value) > 0.0 else float(default)

    def _is_target_allowed(self, target_pose: PoseStamped) -> bool:
        if not self._enforce_workspace_limits:
            return True
        return (
            target_pose.pose.position.x >= self._min_target_x
            and target_pose.pose.position.z >= self._min_target_z
        )

    @staticmethod
    def _make_result(
        success: bool,
        message: str,
        error_code: str,
        pose: Optional[PoseStamped],
    ) -> MoveToPoseResult:
        result = MoveToPoseResult()
        result.success = bool(success)
        result.message = str(message)
        result.error_code = str(error_code)
        if pose is not None:
            result.final_pose = pose
        return result

def main() -> None:
    rospy.init_node("usb_c_insertion_move_to_pose_action_server")
    MoveToPoseActionServer()
    rospy.spin()


if __name__ == "__main__":
    main()
