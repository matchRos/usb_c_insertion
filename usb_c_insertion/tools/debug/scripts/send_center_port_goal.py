#!/usr/bin/env python3

from __future__ import annotations

import os
import sys

import actionlib
import rospy

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PACKAGE_SCRIPTS_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "..", "..", "scripts"))
if PACKAGE_SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, PACKAGE_SCRIPTS_DIR)

from param_utils import get_param
from usb_c_insertion.msg import CenterPortInImageAction, CenterPortInImageGoal


class CenterPortGoalClient:
    def __init__(self):
        self._action_name = str(get_param("~center_port/action_name", "center_port_in_image")).strip()
        self._wait_timeout = float(get_param("~center_port/client_wait_timeout", 5.0))
        self._result_timeout = float(
            get_param("~center_port/client_result_timeout", get_param("~center_port/timeout", 10.0) + 2.0)
        )
        self._client = actionlib.SimpleActionClient(self._action_name, CenterPortInImageAction)

    def run(self) -> bool:
        rospy.loginfo(
            "[usb_c_insertion] event=center_port_goal_client_wait action=%s timeout=%.1f",
            self._action_name,
            self._wait_timeout,
        )
        if not self._client.wait_for_server(rospy.Duration.from_sec(max(0.1, self._wait_timeout))):
            rospy.logerr(
                "[usb_c_insertion] event=center_port_goal_client_failed reason=action_server_unavailable action=%s",
                self._action_name,
            )
            return False

        goal = self._build_goal()
        rospy.loginfo(
            "[usb_c_insertion] event=center_port_goal_send image_topic=%s timeout=%.1f pixel_tolerance=%.1f max_velocity=%.4f gain=%.7f min_blob_area=%.1f",
            goal.image_topic,
            goal.timeout,
            goal.pixel_tolerance,
            goal.max_velocity,
            goal.gain,
            goal.min_blob_area,
        )
        self._client.send_goal(goal, feedback_cb=self._feedback_callback)

        finished = self._client.wait_for_result(rospy.Duration.from_sec(max(0.1, self._result_timeout)))
        if not finished:
            self._client.cancel_goal()
            rospy.logerr("[usb_c_insertion] event=center_port_goal_client_failed reason=result_timeout")
            return False

        result = self._client.get_result()
        if result is None:
            rospy.logerr("[usb_c_insertion] event=center_port_goal_client_failed reason=no_result")
            return False

        log_fn = rospy.loginfo if result.success else rospy.logerr
        log_fn(
            "[usb_c_insertion] event=center_port_goal_result success=%s message=%s error_code=%s error_norm=%.2f blob_center=(%.1f,%.1f) image_size=%ux%u elapsed=%.2f",
            str(bool(result.success)).lower(),
            result.message,
            result.error_code,
            float(result.error_norm),
            float(result.blob_center_x),
            float(result.blob_center_y),
            int(result.image_width),
            int(result.image_height),
            float(result.elapsed),
        )
        return bool(result.success)

    def _build_goal(self) -> CenterPortInImageGoal:
        goal = CenterPortInImageGoal()
        goal.image_topic = str(
            get_param("~image_topic", get_param("~center_port/image_topic", "/zedm/zed_node/left/image_rect_color"))
        ).strip()
        goal.timeout = float(get_param("~center_port/timeout", 10.0))
        goal.pixel_tolerance = float(get_param("~center_port/pixel_tolerance", 12.0))
        goal.stable_time = float(get_param("~center_port/stable_time", 0.35))
        goal.max_velocity = float(get_param("~center_port/max_velocity", 0.006))
        goal.gain = float(get_param("~center_port/gain", 0.00002))
        goal.min_blob_area = float(get_param("~center_port/min_blob_area", 120.0))
        return goal

    @staticmethod
    def _feedback_callback(feedback) -> None:
        rospy.loginfo_throttle(
            0.5,
            "[usb_c_insertion] event=center_port_goal_feedback stage=%s error_x=%.1f error_y=%.1f error_norm=%.1f blob_area=%.1f command_tool_x=%.4f command_tool_y=%.4f elapsed=%.2f",
            feedback.stage,
            float(feedback.error_x),
            float(feedback.error_y),
            float(feedback.error_norm),
            float(feedback.blob_area),
            float(feedback.command_tool_x),
            float(feedback.command_tool_y),
            float(feedback.elapsed),
        )


def main() -> None:
    rospy.init_node("usb_c_insertion_center_port_goal_client")
    success = CenterPortGoalClient().run()
    if not success:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
