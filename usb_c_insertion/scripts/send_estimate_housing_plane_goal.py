#!/usr/bin/env python3

from __future__ import annotations

import actionlib
import rospy

from usb_c_insertion.msg import EstimateHousingPlaneAction, EstimateHousingPlaneGoal


class EstimateHousingPlaneGoalClient:
    def __init__(self):
        self._action_name = str(rospy.get_param("~housing_plane/action_name", "estimate_housing_plane")).strip()
        self._wait_timeout = float(rospy.get_param("~housing_plane/client_wait_timeout", 5.0))
        self._result_timeout = float(
            rospy.get_param(
                "~housing_plane/client_result_timeout",
                rospy.get_param("~housing_plane/timeout", 3.0) + 2.0,
            )
        )
        self._client = actionlib.SimpleActionClient(self._action_name, EstimateHousingPlaneAction)

    def run(self) -> bool:
        rospy.loginfo(
            "[usb_c_insertion] event=estimate_housing_plane_goal_client_wait action=%s timeout=%.1f",
            self._action_name,
            self._wait_timeout,
        )
        if not self._client.wait_for_server(rospy.Duration.from_sec(max(0.1, self._wait_timeout))):
            rospy.logerr(
                "[usb_c_insertion] event=estimate_housing_plane_goal_client_failed reason=action_server_unavailable action=%s",
                self._action_name,
            )
            return False

        goal = self._build_goal()
        rospy.loginfo(
            "[usb_c_insertion] event=estimate_housing_plane_goal_send image_topic=%s cloud_topic=%s base_transform_frame=%s cloud_to_base_transform_rotation=%s roi_radius_px=%d roi_stride_px=%d ransac_iterations=%d use_svd_refit=%s",
            goal.image_topic,
            goal.cloud_topic,
            str(rospy.get_param("~housing_plane/base_transform_frame", "")).strip() or "<cloud_frame>",
            str(rospy.get_param("~housing_plane/cloud_to_base_transform_rotation", "identity")).strip(),
            int(goal.roi_radius_px),
            int(goal.roi_stride_px),
            int(goal.ransac_iterations),
            str(bool(goal.use_svd_refit)).lower(),
        )
        self._client.send_goal(goal, feedback_cb=self._feedback_callback)

        finished = self._client.wait_for_result(rospy.Duration.from_sec(max(0.1, self._result_timeout)))
        if not finished:
            self._client.cancel_goal()
            rospy.logerr("[usb_c_insertion] event=estimate_housing_plane_goal_client_failed reason=result_timeout")
            return False

        result = self._client.get_result()
        if result is None:
            rospy.logerr("[usb_c_insertion] event=estimate_housing_plane_goal_client_failed reason=no_result")
            return False

        log_fn = rospy.loginfo if result.success else rospy.logerr
        log_fn(
            "[usb_c_insertion] event=estimate_housing_plane_goal_result success=%s message=%s error_code=%s cloud_frame=%s base_frame=%s inliers=%d filtered=%d ratio=%.3f rms=%.4f normal_cloud=(%.4f,%.4f,%.4f) normal_base=(%.4f,%.4f,%.4f) point_base=(%.4f,%.4f,%.4f) marker_point_base=(%.4f,%.4f,%.4f) elapsed=%.2f",
            str(bool(result.success)).lower(),
            result.message,
            result.error_code,
            result.plane_point.header.frame_id,
            result.plane_point_base.header.frame_id,
            int(result.inlier_count),
            int(result.filtered_point_count),
            float(result.inlier_ratio),
            float(result.rms_error),
            float(result.plane_normal.x),
            float(result.plane_normal.y),
            float(result.plane_normal.z),
            float(result.plane_normal_base.x),
            float(result.plane_normal_base.y),
            float(result.plane_normal_base.z),
            float(result.plane_point_base.point.x),
            float(result.plane_point_base.point.y),
            float(result.plane_point_base.point.z),
            float(result.marker_plane_point_base.point.x),
            float(result.marker_plane_point_base.point.y),
            float(result.marker_plane_point_base.point.z),
            float(result.elapsed),
        )
        return bool(result.success)

    def _build_goal(self) -> EstimateHousingPlaneGoal:
        goal = EstimateHousingPlaneGoal()
        goal.image_topic = str(
            rospy.get_param("~housing_plane/image_topic", "/zedm/zed_node/left/image_rect_color")
        ).strip()
        goal.cloud_topic = str(
            rospy.get_param("~housing_plane/cloud_topic", "/zedm/zed_node/point_cloud/cloud_registered")
        ).strip()
        goal.timeout = float(rospy.get_param("~housing_plane/timeout", 3.0))
        goal.min_blob_area = float(rospy.get_param("~housing_plane/min_blob_area", 120.0))
        goal.roi_radius_px = int(rospy.get_param("~housing_plane/roi_radius_px", 70))
        goal.roi_stride_px = int(rospy.get_param("~housing_plane/roi_stride_px", 2))
        goal.depth_window_m = float(rospy.get_param("~housing_plane/depth_window_m", 0.06))
        goal.ransac_iterations = int(rospy.get_param("~housing_plane/ransac_iterations", 120))
        goal.ransac_distance_threshold = float(rospy.get_param("~housing_plane/ransac_distance_threshold", 0.004))
        goal.min_inliers = int(rospy.get_param("~housing_plane/min_inliers", 120))
        goal.use_svd_refit = bool(rospy.get_param("~housing_plane/use_svd_refit", True))
        goal.use_largest_component = bool(rospy.get_param("~housing_plane/use_largest_component", True))
        return goal

    @staticmethod
    def _feedback_callback(feedback) -> None:
        rospy.loginfo_throttle(
            0.5,
            "[usb_c_insertion] event=estimate_housing_plane_goal_feedback stage=%s marker=(%.1f,%.1f) raw=(%.1f,%.1f) roi=%d filtered=%d inliers=%d ratio=%.3f rms=%.4f elapsed=%.2f",
            feedback.stage,
            float(feedback.marker_center_x),
            float(feedback.marker_center_y),
            float(feedback.raw_center_u),
            float(feedback.raw_center_v),
            int(feedback.roi_point_count),
            int(feedback.filtered_point_count),
            int(feedback.inlier_count),
            float(feedback.inlier_ratio),
            float(feedback.rms_error),
            float(feedback.elapsed),
        )


def main() -> None:
    rospy.init_node("usb_c_insertion_estimate_housing_plane_goal_client")
    success = EstimateHousingPlaneGoalClient().run()
    if not success:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
