#!/usr/bin/env python3

from __future__ import annotations

import math

import actionlib
import rospy

from usb_c_insertion.msg import AlignHousingYawAction, AlignHousingYawGoal


class AlignHousingYawGoalClient:
    def __init__(self):
        self._action_name = str(rospy.get_param("~align_housing_yaw/action_name", "align_housing_yaw")).strip()
        self._wait_timeout = float(rospy.get_param("~align_housing_yaw/client_wait_timeout", 5.0))
        self._client = actionlib.SimpleActionClient(self._action_name, AlignHousingYawAction)

    def run(self) -> bool:
        rospy.loginfo(
            "[usb_c_insertion] event=align_housing_yaw_goal_client_wait action=%s timeout=%.1f",
            self._action_name,
            self._wait_timeout,
        )
        if not self._client.wait_for_server(rospy.Duration.from_sec(max(0.1, self._wait_timeout))):
            rospy.logerr(
                "[usb_c_insertion] event=align_housing_yaw_goal_client_failed reason=action_server_unavailable action=%s",
                self._action_name,
            )
            return False

        goal = self._build_goal()
        result_timeout = self._result_timeout(goal)
        rospy.loginfo(
            "[usb_c_insertion] event=align_housing_yaw_goal_send image_topic=%s cloud_topic=%s yaw_tolerance_deg=%.2f max_iterations=%d max_yaw_step_deg=%.2f",
            goal.image_topic,
            goal.cloud_topic,
            math.degrees(goal.yaw_tolerance_rad),
            int(goal.max_iterations),
            math.degrees(goal.max_yaw_step_rad),
        )
        self._client.send_goal(goal, feedback_cb=self._feedback_callback)

        finished = self._client.wait_for_result(rospy.Duration.from_sec(max(0.1, result_timeout)))
        if not finished:
            self._client.cancel_goal()
            rospy.logerr("[usb_c_insertion] event=align_housing_yaw_goal_client_failed reason=result_timeout")
            return False

        result = self._client.get_result()
        if result is None:
            rospy.logerr("[usb_c_insertion] event=align_housing_yaw_goal_client_failed reason=no_result")
            return False

        log_fn = rospy.loginfo if result.success else rospy.logerr
        log_fn(
            "[usb_c_insertion] event=align_housing_yaw_goal_result success=%s message=%s error_code=%s iterations=%d initial_yaw_error_deg=%.2f final_yaw_error_deg=%.2f total_yaw_command_deg=%.2f normal_base=(%.4f,%.4f,%.4f)",
            str(bool(result.success)).lower(),
            result.message,
            result.error_code,
            int(result.iterations),
            math.degrees(result.initial_yaw_error_rad),
            math.degrees(result.final_yaw_error_rad),
            math.degrees(result.total_yaw_command_rad),
            result.final_normal_base.x,
            result.final_normal_base.y,
            result.final_normal_base.z,
        )
        return bool(result.success)

    def _build_goal(self) -> AlignHousingYawGoal:
        goal = AlignHousingYawGoal()
        goal.image_topic = str(
            rospy.get_param("~align_housing_yaw/image_topic", rospy.get_param("~housing_plane/image_topic", ""))
        ).strip()
        goal.cloud_topic = str(
            rospy.get_param("~align_housing_yaw/cloud_topic", rospy.get_param("~housing_plane/cloud_topic", ""))
        ).strip()
        goal.estimate_timeout = float(rospy.get_param("~align_housing_yaw/estimate_timeout", 3.0))
        goal.yaw_tolerance_rad = float(rospy.get_param("~align_housing_yaw/yaw_tolerance_rad", 0.0175))
        goal.max_iterations = int(rospy.get_param("~align_housing_yaw/max_iterations", 10))
        goal.max_yaw_step_rad = float(rospy.get_param("~align_housing_yaw/max_yaw_step_rad", 0.25))
        goal.settle_time = float(rospy.get_param("~align_housing_yaw/settle_time", 0.15))
        goal.move_timeout = float(rospy.get_param("~align_housing_yaw/move_timeout", 20.0))
        return goal

    @staticmethod
    def _result_timeout(goal: AlignHousingYawGoal) -> float:
        estimate_timeout = max(0.1, float(goal.estimate_timeout) + 2.0)
        move_timeout = max(0.1, float(goal.move_timeout) + 5.0)
        return estimate_timeout + float(max(0, int(goal.max_iterations))) * (estimate_timeout + move_timeout)

    @staticmethod
    def _feedback_callback(feedback) -> None:
        rospy.loginfo(
            "[usb_c_insertion] event=align_housing_yaw_goal_feedback stage=%s iteration=%d yaw_error_deg=%.2f yaw_command_deg=%.2f total_yaw_command_deg=%.2f normal_base=(%.4f,%.4f,%.4f) ratio=%.3f rms=%.4f",
            feedback.stage,
            int(feedback.iteration),
            math.degrees(feedback.yaw_error_rad),
            math.degrees(feedback.yaw_command_rad),
            math.degrees(feedback.total_yaw_command_rad),
            feedback.normal_base.x,
            feedback.normal_base.y,
            feedback.normal_base.z,
            feedback.inlier_ratio,
            feedback.rms_error,
        )


def main() -> None:
    rospy.init_node("usb_c_insertion_align_housing_yaw_goal_client")
    success = AlignHousingYawGoalClient().run()
    if not success:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
