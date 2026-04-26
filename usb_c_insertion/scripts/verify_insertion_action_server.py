#!/usr/bin/env python3

from __future__ import annotations

import os
import sys

import actionlib
import rospy

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from ft_interface import FTInterface
from post_insertion_verifier import PostInsertionVerifier
from robot_interface import RobotInterface
from tf_interface import TFInterface
from usb_c_insertion.msg import (
    VerifyInsertionAction,
    VerifyInsertionFeedback,
    VerifyInsertionResult,
)


class VerifyInsertionActionServer:
    """
    Action wrapper around the post-insertion retention check.

    A non-verified insertion is a normal result, not an action transport error.
    Callers should inspect result.verified to decide whether to accept a port
    candidate or keep searching.
    """

    def __init__(self):
        self._action_name = str(
            rospy.get_param(
                "~verify/action_name",
                rospy.get_param("~action_name", "verify_insertion"),
            )
        ).strip()
        self._default_timeout = float(rospy.get_param("~verify/action_timeout", 2.0))
        self._default_zero_ft = bool(rospy.get_param("~verify/zero_ft_before_verify", False))
        self._base_frame = str(rospy.get_param("~frames/base_frame", "base_link")).strip()

        self._robot = RobotInterface()
        self._tf = TFInterface()
        self._ft = FTInterface(
            wrench_topic=rospy.get_param("~topics/wrench", "/wrench"),
            filter_window_size=rospy.get_param("~contact/baseline_window", 20),
            wrench_timeout=rospy.get_param("~contact/wrench_timeout", 0.2),
        )
        self._verifier = PostInsertionVerifier(self._robot, self._tf, self._ft)
        self._server = actionlib.SimpleActionServer(
            self._action_name,
            VerifyInsertionAction,
            execute_cb=self._execute,
            auto_start=False,
        )
        self._server.start()
        rospy.loginfo("[usb_c_insertion] event=verify_insertion_action_ready action=%s", self._action_name)

    def _execute(self, goal) -> None:
        started_at = rospy.Time.now()
        timeout = self._goal_or_default(goal.timeout, self._default_timeout)
        zero_ft = self._default_zero_ft or bool(goal.zero_ft_before_verify)

        self._publish_feedback("starting", started_at, 0.0, 0.0)
        if not self._robot.wait_for_motion_pipeline(timeout=1.0, require_pose_servo=True):
            self._server.set_aborted(
                self._make_result(
                    False,
                    False,
                    "motion_pipeline_unavailable",
                    "motion_pipeline_unavailable",
                    "motion_pipeline_unavailable",
                    0.0,
                    0.0,
                )
            )
            return

        if zero_ft:
            self._publish_feedback("zero_ft", started_at, 0.0, 0.0)
            if not self._ft.zero_sensor():
                self._server.set_aborted(
                    self._make_result(
                        False,
                        False,
                        "zero_ft_failed",
                        "zero_ft_failed",
                        "zero_ft_failed",
                        0.0,
                        0.0,
                    )
                )
                return
            rospy.sleep(0.2)

        if self._server.is_preempt_requested():
            self._robot.stop_motion()
            self._server.set_preempted(
                self._make_result(False, False, "preempted", "preempted", "preempted", 0.0, 0.0)
            )
            return

        self._publish_feedback("verify_retention", started_at, 0.0, 0.0)
        result = self._verifier.verify_retention(move_timeout=timeout)
        self._publish_feedback(
            "complete",
            started_at,
            result.counterforce_y,
            result.counterforce_z,
        )

        action_result = self._make_result(
            True,
            result.success,
            result.reason,
            "",
            "" if result.success else result.reason,
            result.counterforce_y,
            result.counterforce_z,
        )
        current_pose = self._tf.get_tool_pose_in_base()
        if current_pose is not None:
            action_result.verified_pose = current_pose
        else:
            action_result.verified_pose.header.stamp = rospy.Time.now()
            action_result.verified_pose.header.frame_id = self._base_frame

        self._server.set_succeeded(action_result)

    def _publish_feedback(
        self,
        stage: str,
        started_at: rospy.Time,
        counterforce_y: float,
        counterforce_z: float,
    ) -> None:
        feedback = VerifyInsertionFeedback()
        feedback.stage = stage
        feedback.elapsed = float((rospy.Time.now() - started_at).to_sec())
        feedback.counterforce_y = float(counterforce_y)
        feedback.counterforce_z = float(counterforce_z)
        self._server.publish_feedback(feedback)

    @staticmethod
    def _make_result(
        success: bool,
        verified: bool,
        message: str,
        error_code: str,
        failure_reason: str,
        counterforce_y: float,
        counterforce_z: float,
    ) -> VerifyInsertionResult:
        result = VerifyInsertionResult()
        result.success = bool(success)
        result.verified = bool(verified)
        result.message = str(message)
        result.error_code = str(error_code)
        result.failure_reason = str(failure_reason)
        result.counterforce_y = float(counterforce_y)
        result.counterforce_z = float(counterforce_z)
        return result

    @staticmethod
    def _goal_or_default(value: float, default: float) -> float:
        return float(value) if float(value) > 0.0 else float(default)


def main() -> None:
    rospy.init_node("usb_c_insertion_verify_insertion_action_server")
    VerifyInsertionActionServer()
    rospy.spin()


if __name__ == "__main__":
    main()
