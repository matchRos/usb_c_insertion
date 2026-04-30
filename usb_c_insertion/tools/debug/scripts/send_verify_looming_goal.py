#!/usr/bin/env python3

from __future__ import annotations

import actionlib
import rospy

from usb_c_insertion.msg import VerifyLoomingAction, VerifyLoomingGoal


class VerifyLoomingGoalClient:
    def __init__(self):
        self._action_name = str(rospy.get_param("~looming/action_name", "verify_looming")).strip()
        self._wait_timeout = float(rospy.get_param("~looming/client_wait_timeout", 5.0))
        self._result_timeout = float(
            rospy.get_param("~looming/client_result_timeout", rospy.get_param("~looming/timeout", 8.0) + 2.0)
        )
        self._client = actionlib.SimpleActionClient(self._action_name, VerifyLoomingAction)

    def run(self) -> bool:
        rospy.loginfo(
            "[usb_c_insertion] event=verify_looming_goal_client_wait action=%s timeout=%.1f",
            self._action_name,
            self._wait_timeout,
        )
        if not self._client.wait_for_server(rospy.Duration.from_sec(max(0.1, self._wait_timeout))):
            rospy.logerr(
                "[usb_c_insertion] event=verify_looming_goal_client_failed reason=action_server_unavailable action=%s",
                self._action_name,
            )
            return False

        goal = self._build_goal()
        rospy.loginfo(
            "[usb_c_insertion] event=verify_looming_goal_send image_topic=%s travel_distance=%.4f travel_speed=%.4f direction_sign=%.1f min_scale_ratio=%.3f max_center_shift_px=%.1f",
            goal.image_topic,
            goal.travel_distance,
            goal.travel_speed,
            goal.tool_z_direction_sign,
            goal.min_scale_ratio,
            goal.max_center_shift_px,
        )
        self._client.send_goal(goal, feedback_cb=self._feedback_callback)

        finished = self._client.wait_for_result(rospy.Duration.from_sec(max(0.1, self._result_timeout)))
        if not finished:
            self._client.cancel_goal()
            rospy.logerr("[usb_c_insertion] event=verify_looming_goal_client_failed reason=result_timeout")
            return False

        result = self._client.get_result()
        if result is None:
            rospy.logerr("[usb_c_insertion] event=verify_looming_goal_client_failed reason=no_result")
            return False

        log_fn = rospy.loginfo if result.success else rospy.logerr
        log_fn(
            "[usb_c_insertion] event=verify_looming_goal_result success=%s message=%s error_code=%s scale_ratio=%.3f center_shift_px=%.2f traveled_distance=%.4f elapsed=%.2f",
            str(bool(result.success)).lower(),
            result.message,
            result.error_code,
            float(result.scale_ratio),
            float(result.center_shift_px),
            float(result.traveled_distance),
            float(result.elapsed),
        )
        return bool(result.success)

    def _build_goal(self) -> VerifyLoomingGoal:
        goal = VerifyLoomingGoal()
        goal.image_topic = str(rospy.get_param("~looming/image_topic", "/zedm/zed_node/left/image_rect_color")).strip()
        goal.travel_distance = float(rospy.get_param("~looming/travel_distance", 0.025))
        goal.travel_speed = float(rospy.get_param("~looming/travel_speed", 0.006))
        goal.timeout = float(rospy.get_param("~looming/timeout", 8.0))
        goal.min_blob_area = float(rospy.get_param("~looming/min_blob_area", 120.0))
        goal.min_scale_ratio = float(rospy.get_param("~looming/min_scale_ratio", 1.12))
        goal.max_center_shift_px = float(rospy.get_param("~looming/max_center_shift_px", 10.0))
        goal.max_aspect_ratio_change = float(rospy.get_param("~looming/max_aspect_ratio_change", 0.35))
        goal.tool_z_direction_sign = float(rospy.get_param("~looming/tool_z_direction_sign", 1.0))
        return goal

    @staticmethod
    def _feedback_callback(feedback) -> None:
        rospy.loginfo_throttle(
            0.5,
            "[usb_c_insertion] event=verify_looming_goal_feedback stage=%s traveled_distance=%.4f scale_ratio=%.3f center_shift_px=%.2f aspect_ratio_change=%.3f command_tool_z=%.4f elapsed=%.2f",
            feedback.stage,
            float(feedback.traveled_distance),
            float(feedback.scale_ratio),
            float(feedback.center_shift_px),
            float(feedback.aspect_ratio_change),
            float(feedback.command_tool_z),
            float(feedback.elapsed),
        )


def main() -> None:
    rospy.init_node("usb_c_insertion_verify_looming_goal_client")
    success = VerifyLoomingGoalClient().run()
    if not success:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
