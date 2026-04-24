#!/usr/bin/env python3

from __future__ import annotations

import os
import sys

import actionlib
from geometry_msgs.msg import PoseStamped
import rospy

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from prepose_planner import normalize_quaternion, quaternion_from_yaw, quaternion_multiply
from usb_c_insertion.msg import (
    ApplyYawCorrectionAction,
    ApplyYawCorrectionFeedback,
    ApplyYawCorrectionResult,
    MoveToPoseAction,
    MoveToPoseGoal,
)


class ApplyYawCorrectionActionServer:
    """
    Apply a yaw correction around base_link.z to a reference pose.

    The action keeps the Cartesian position fixed, rotates the reference pose
    around the world z axis, and then delegates motion execution to the
    existing move_to_pose action.
    """

    def __init__(self):
        self._action_name = str(rospy.get_param("~action_name", "apply_yaw_correction")).strip()
        self._move_action_name = str(rospy.get_param("~move_action_name", "move_to_pose")).strip()
        self._base_frame = str(rospy.get_param("~frames/base_frame", "base_link"))
        self._default_settle_time = float(
            rospy.get_param("~motion/action_settle_time", 0.4)
        )
        self._default_timeout = float(
            rospy.get_param("~motion/action_timeout", 20.0)
        )

        self._move_client = actionlib.SimpleActionClient(self._move_action_name, MoveToPoseAction)
        self._server = actionlib.SimpleActionServer(
            self._action_name,
            ApplyYawCorrectionAction,
            execute_cb=self._execute,
            auto_start=False,
        )
        self._server.start()
        rospy.loginfo(
            "[usb_c_insertion] event=apply_yaw_correction_action_ready action=%s",
            self._action_name,
        )

    def _execute(self, goal) -> None:
        if not self._move_client.wait_for_server(rospy.Duration.from_sec(5.0)):
            self._abort("move_action_unavailable", None)
            return

        frame_id = goal.reference_pose.header.frame_id.strip() or self._base_frame
        if frame_id != self._base_frame:
            self._abort("unsupported_reference_pose_frame", None)
            return

        corrected_pose = self._build_corrected_pose(goal.reference_pose, goal.yaw_correction_rad)
        self._publish_stage("move_to_corrected_pose")

        move_goal = MoveToPoseGoal()
        move_goal.target_pose = corrected_pose
        move_goal.settle_time = self._goal_or_default(goal.settle_time, self._default_settle_time)
        move_goal.timeout = self._goal_or_default(goal.timeout, self._default_timeout)

        self._move_client.send_goal(move_goal)
        finished = self._move_client.wait_for_result(
            rospy.Duration.from_sec(max(1.0, move_goal.timeout + 5.0))
        )
        if not finished:
            self._move_client.cancel_goal()
            self._abort("move_goal_wait_timeout", corrected_pose)
            return

        result = self._move_client.get_result()
        if result is None or not bool(result.success):
            message = result.message if result is not None else "no_result"
            error_code = result.error_code if result is not None else "move_goal_no_result"
            self._abort("move_goal_failed: %s" % message, corrected_pose, error_code)
            return

        action_result = ApplyYawCorrectionResult()
        action_result.success = True
        action_result.message = "yaw_correction_applied"
        action_result.error_code = ""
        action_result.corrected_pose = result.final_pose
        self._server.set_succeeded(action_result)

    def _build_corrected_pose(self, reference_pose: PoseStamped, yaw_correction_rad: float) -> PoseStamped:
        corrected_pose = PoseStamped()
        corrected_pose.header.stamp = rospy.Time.now()
        corrected_pose.header.frame_id = self._base_frame
        corrected_pose.pose.position = reference_pose.pose.position

        reference_quaternion = (
            reference_pose.pose.orientation.x,
            reference_pose.pose.orientation.y,
            reference_pose.pose.orientation.z,
            reference_pose.pose.orientation.w,
        )
        yaw_quaternion = quaternion_from_yaw(float(yaw_correction_rad))
        corrected_quaternion = normalize_quaternion(
            quaternion_multiply(yaw_quaternion, reference_quaternion)
        )
        corrected_pose.pose.orientation.x = corrected_quaternion[0]
        corrected_pose.pose.orientation.y = corrected_quaternion[1]
        corrected_pose.pose.orientation.z = corrected_quaternion[2]
        corrected_pose.pose.orientation.w = corrected_quaternion[3]
        return corrected_pose

    def _publish_stage(self, stage: str) -> None:
        feedback = ApplyYawCorrectionFeedback()
        feedback.stage = stage
        self._server.publish_feedback(feedback)

    def _abort(
        self,
        message: str,
        corrected_pose: PoseStamped | None,
        error_code: str | None = None,
    ) -> None:
        result = ApplyYawCorrectionResult()
        result.success = False
        result.message = str(message)
        result.error_code = str(error_code or message)
        if corrected_pose is not None:
            result.corrected_pose = corrected_pose
        self._server.set_aborted(result)

    @staticmethod
    def _goal_or_default(value: float, default: float) -> float:
        return float(value) if float(value) > 0.0 else float(default)


def main() -> None:
    rospy.init_node("usb_c_insertion_apply_yaw_correction_action_server")
    ApplyYawCorrectionActionServer()
    rospy.spin()


if __name__ == "__main__":
    main()
