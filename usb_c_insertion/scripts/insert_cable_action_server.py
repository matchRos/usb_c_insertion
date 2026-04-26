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
from insertion_controller import InsertionController
from prepose_planner import rotate_vector_by_quaternion
from robot_interface import RobotInterface
from tf_interface import TFInterface
from usb_c_insertion.msg import (
    InsertCableAction,
    InsertCableFeedback,
    InsertCableResult,
)


class InsertCableActionServer:
    """
    Action wrapper for the final force-guided insertion motion.

    The goal provides the wall contact reference point and a reference pose.
    The server uses the reference pose's tool-z axis as the insertion direction.
    """

    def __init__(self):
        self._action_name = str(
            rospy.get_param(
                "~insert/action_name",
                rospy.get_param("~action_name", "insert_cable"),
            )
        ).strip()
        self._base_frame = str(rospy.get_param("~frames/base_frame", "base_link")).strip()
        self._default_timeout = float(rospy.get_param("~insert/force_control_timeout", 8.0))
        self._default_zero_ft = bool(rospy.get_param("~insert/zero_ft_before_insert", False))

        self._robot = RobotInterface()
        self._tf = TFInterface()
        self._ft = FTInterface(
            wrench_topic=rospy.get_param("~topics/wrench", "/wrench"),
            filter_window_size=rospy.get_param("~contact/baseline_window", 20),
            wrench_timeout=rospy.get_param("~contact/wrench_timeout", 0.2),
        )
        self._controller = InsertionController(self._robot, self._tf, self._ft)
        self._server = actionlib.SimpleActionServer(
            self._action_name,
            InsertCableAction,
            execute_cb=self._execute,
            auto_start=False,
        )
        self._server.start()
        rospy.loginfo("[usb_c_insertion] event=insert_cable_action_ready action=%s", self._action_name)

    def _execute(self, goal) -> None:
        started_at = rospy.Time.now()
        timeout = self._goal_or_default(goal.timeout, self._default_timeout)
        zero_ft = self._default_zero_ft or bool(goal.zero_ft_before_insert)

        if not self._validate_goal_frames(goal):
            return

        if not self._robot.wait_for_motion_pipeline(timeout=1.0, require_pose_servo=False):
            self._server.set_aborted(
                self._make_result(
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
                    self._make_result(False, "zero_ft_failed", "zero_ft_failed", "zero_ft_failed", 0.0, 0.0)
                )
                return
            rospy.sleep(0.2)

        try:
            insertion_direction = self._tool_z_direction(goal.reference_pose)
            wiggle_y_direction = self._tool_axis_direction(goal.reference_pose, (0.0, 1.0, 0.0))
            wiggle_x_direction = self._tool_axis_direction(goal.reference_pose, (1.0, 0.0, 0.0))
        except ValueError as exc:
            self._server.set_aborted(
                self._make_result(False, str(exc), "insert_geometry_failed", "insert_geometry_failed", 0.0, 0.0)
            )
            return

        reference_point = (
            goal.reference_point.point.x,
            goal.reference_point.point.y,
            goal.reference_point.point.z,
        )

        self._publish_feedback("insert", started_at, 0.0, 0.0)
        result = self._controller.insert_until_depth(
            reference_point,
            insertion_direction,
            force_control_timeout=timeout,
            wiggle_y_direction_xyz=wiggle_y_direction,
            wiggle_x_direction_xyz=wiggle_x_direction,
        )
        action_result = self._make_result(
            result.success,
            result.reason,
            "" if result.success else result.reason,
            "" if result.success else result.reason,
            result.inserted_depth,
            result.contact_force,
        )
        final_pose = self._tf.get_tool_pose_in_base()
        if final_pose is not None:
            action_result.final_pose = final_pose
        else:
            action_result.final_pose.header.stamp = rospy.Time.now()
            action_result.final_pose.header.frame_id = self._base_frame

        self._publish_feedback("complete", started_at, result.inserted_depth, result.contact_force)
        if result.success:
            self._server.set_succeeded(action_result)
        else:
            self._server.set_aborted(action_result)

    def _validate_goal_frames(self, goal) -> bool:
        point_frame = goal.reference_point.header.frame_id.strip() or self._base_frame
        pose_frame = goal.reference_pose.header.frame_id.strip() or self._base_frame
        if point_frame == self._base_frame and pose_frame == self._base_frame:
            return True

        message = "unsupported_goal_frame"
        rospy.logerr(
            "[usb_c_insertion] event=insert_cable_goal_invalid reason=%s point_frame=%s pose_frame=%s base_frame=%s",
            message,
            point_frame,
            pose_frame,
            self._base_frame,
        )
        self._server.set_aborted(
            self._make_result(False, message, message, message, 0.0, 0.0)
        )
        return False

    def _publish_feedback(
        self,
        stage: str,
        started_at: rospy.Time,
        inserted_depth: float,
        contact_force: float,
    ) -> None:
        feedback = InsertCableFeedback()
        feedback.stage = stage
        feedback.elapsed = float((rospy.Time.now() - started_at).to_sec())
        feedback.inserted_depth = float(inserted_depth)
        feedback.contact_force = float(contact_force)
        self._server.publish_feedback(feedback)

    def _make_result(
        self,
        success: bool,
        message: str,
        error_code: str,
        failure_reason: str,
        inserted_depth: float,
        contact_force: float,
    ) -> InsertCableResult:
        result = InsertCableResult()
        result.success = bool(success)
        result.message = str(message)
        result.error_code = str(error_code)
        result.failure_reason = str(failure_reason)
        result.inserted_depth = float(inserted_depth)
        result.contact_force = float(contact_force)
        return result

    @staticmethod
    def _tool_z_direction(reference_pose):
        return InsertCableActionServer._tool_axis_direction(reference_pose, (0.0, 0.0, 1.0))

    @staticmethod
    def _tool_axis_direction(reference_pose, axis_xyz):
        quaternion = (
            reference_pose.pose.orientation.x,
            reference_pose.pose.orientation.y,
            reference_pose.pose.orientation.z,
            reference_pose.pose.orientation.w,
        )
        return rotate_vector_by_quaternion(axis_xyz[0], axis_xyz[1], axis_xyz[2], *quaternion)

    @staticmethod
    def _goal_or_default(value: float, default: float) -> float:
        return float(value) if float(value) > 0.0 else float(default)


def main() -> None:
    rospy.init_node("usb_c_insertion_insert_cable_action_server")
    InsertCableActionServer()
    rospy.spin()


if __name__ == "__main__":
    main()
