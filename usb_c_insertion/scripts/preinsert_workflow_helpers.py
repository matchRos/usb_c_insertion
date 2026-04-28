#!/usr/bin/env python3

from __future__ import annotations

import math
import os
import sys
from typing import Optional, Tuple

import actionlib
from geometry_msgs.msg import PoseStamped
import rospy

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from prepose_planner import (
    euler_from_quaternion,
    normalize_angle,
    normalize_quaternion,
    normalize_vector,
    quaternion_from_euler,
    rotate_vector_by_quaternion,
)
from tf_interface import TFInterface
from usb_c_insertion.msg import (
    AlignHousingYawAction,
    AlignHousingYawGoal,
    CenterPortInImageAction,
    CenterPortInImageGoal,
    EstimateHousingPlaneAction,
    EstimateHousingPlaneGoal,
    MoveToPoseAction,
    MoveToPoseGoal,
    VerifyLoomingAction,
    VerifyLoomingGoal,
)
from usb_c_insertion.srv import RunVision, RunVisionRequest


class PreinsertWorkflowHelpers:
    """
    Mechanical helpers for the high-level pre-insertion workflow.

    The orchestration file should read like the actual process. This class owns
    parameter loading, action calls, and frame math so the workflow stays clean.
    """

    def __init__(self):
        self.base_frame = str(rospy.get_param("~frames/base_frame", "base_link")).strip()
        self.tool_frame = str(rospy.get_param("~frames/tool_frame", "tool0_controller")).strip()

        self._move_action_name = str(rospy.get_param("~move_action_name", "move_to_pose")).strip()
        self._vision_service_name = str(rospy.get_param("~vision_service_name", "run_vision")).strip()
        self._align_action_name = str(
            rospy.get_param("~align_housing_yaw/action_name", "align_housing_yaw")
        ).strip()
        self._center_action_name = str(rospy.get_param("~center_port/action_name", "center_port_in_image")).strip()
        self._estimate_action_name = str(
            rospy.get_param("~housing_plane/action_name", "estimate_housing_plane")
        ).strip()
        self._looming_action_name = str(rospy.get_param("~looming/action_name", "verify_looming")).strip()

        self._move_settle_time = float(rospy.get_param("~photo_pose/settle_time", 0.4))
        self._move_timeout = float(rospy.get_param("~photo_pose/timeout", 20.0))
        self._require_fresh_vision = bool(rospy.get_param("~photo_pose/require_fresh_vision_result", True))
        self._refine_camera_frame = str(
            rospy.get_param("~workflow/refine_camera_frame", "usb_c_zedm_left_camera_optical_frame")
        ).strip()
        self._refine_camera_distance = float(rospy.get_param("~workflow/refine_camera_distance", 0.18))
        self._refine_yaw_delta_sign = float(rospy.get_param("~workflow/refine_yaw_delta_sign", 1.0))
        self._refine_yaw_max_delta_deg = float(rospy.get_param("~workflow/refine_yaw_max_delta_deg", 45.0))
        self._precontact_offset_tool_x = float(rospy.get_param("~state_machine/precontact_offset_tool_x", 0.0))
        self._precontact_offset_tool_y = float(rospy.get_param("~state_machine/precontact_offset_tool_y", 0.0))
        self._precontact_offset_tool_z = float(rospy.get_param("~state_machine/precontact_offset_tool_z", 0.010))
        self._target_offset_tool_x = float(rospy.get_param("~state_machine/target_offset_tool_x", 0.0))
        self._target_offset_tool_y = float(rospy.get_param("~state_machine/target_offset_tool_y", 0.0))

        self._tf = TFInterface()
        self._move_client = actionlib.SimpleActionClient(self._move_action_name, MoveToPoseAction)
        self._align_client = actionlib.SimpleActionClient(self._align_action_name, AlignHousingYawAction)
        self._center_client = actionlib.SimpleActionClient(self._center_action_name, CenterPortInImageAction)
        self._estimate_client = actionlib.SimpleActionClient(self._estimate_action_name, EstimateHousingPlaneAction)
        self._looming_client = actionlib.SimpleActionClient(self._looming_action_name, VerifyLoomingAction)

    def wait_for_dependencies(self) -> bool:
        checks = (
            (self._move_client, self._move_action_name, "move_action_unavailable"),
            (self._align_client, self._align_action_name, "align_housing_yaw_action_unavailable"),
            (self._center_client, self._center_action_name, "center_port_action_unavailable"),
            (self._estimate_client, self._estimate_action_name, "estimate_housing_plane_action_unavailable"),
            (self._looming_client, self._looming_action_name, "verify_looming_action_unavailable"),
        )
        for client, name, error_code in checks:
            if not client.wait_for_server(rospy.Duration.from_sec(5.0)):
                rospy.logerr("[usb_c_insertion] event=preinsert_workflow_failed reason=%s action=%s", error_code, name)
                return False
        try:
            rospy.wait_for_service(self._vision_service_name, timeout=5.0)
        except rospy.ROSException:
            rospy.logerr(
                "[usb_c_insertion] event=preinsert_workflow_failed reason=vision_service_unavailable service=%s",
                self._vision_service_name,
            )
            return False
        return True

    def load_overview_pose(self) -> PoseStamped:
        pose = PoseStamped()
        pose.header.frame_id = str(rospy.get_param("~photo_pose/frame_id", self.base_frame))
        pose.pose.position.x = float(rospy.get_param("~photo_pose/x", 0.50))
        pose.pose.position.y = float(rospy.get_param("~photo_pose/y", 0.1))
        pose.pose.position.z = float(rospy.get_param("~photo_pose/z", 0.3))
        qx = float(rospy.get_param("~photo_pose/qx", 0.708))
        qy = float(rospy.get_param("~photo_pose/qy", 0.0))
        qz = float(rospy.get_param("~photo_pose/qz", 0.0))
        qw = float(rospy.get_param("~photo_pose/qw", 0.705))
        qx, qy, qz, qw = normalize_quaternion((qx, qy, qz, qw))
        pose.pose.orientation.x = qx
        pose.pose.orientation.y = qy
        pose.pose.orientation.z = qz
        pose.pose.orientation.w = qw
        return pose

    def move_to_pose(self, pose: PoseStamped, name: str, timeout: Optional[float] = None) -> Optional[PoseStamped]:
        goal = MoveToPoseGoal()
        goal.target_pose = pose
        goal.settle_time = self._move_settle_time
        goal.timeout = self._move_timeout if timeout is None else float(timeout)
        rospy.loginfo(
            "[usb_c_insertion] event=preinsert_move_goal_sent name=%s x=%.4f y=%.4f z=%.4f",
            name,
            pose.pose.position.x,
            pose.pose.position.y,
            pose.pose.position.z,
        )
        self._move_client.send_goal(goal, feedback_cb=self._move_feedback_callback)
        finished = self._move_client.wait_for_result(rospy.Duration.from_sec(max(1.0, goal.timeout + 5.0)))
        if not finished:
            self._move_client.cancel_goal()
            rospy.logerr("[usb_c_insertion] event=preinsert_workflow_failed reason=move_timeout name=%s", name)
            return None
        result = self._move_client.get_result()
        if result is None or not bool(result.success):
            message = result.message if result is not None else "no_result"
            error_code = result.error_code if result is not None else "move_no_result"
            rospy.logerr(
                "[usb_c_insertion] event=preinsert_workflow_failed reason=move_failed name=%s error_code=%s message=%s",
                name,
                error_code,
                message,
            )
            return None
        return result.final_pose

    def run_overview_vision(self) -> Optional[PoseStamped]:
        service = rospy.ServiceProxy(self._vision_service_name, RunVision)
        response = service(RunVisionRequest(require_fresh_result=self._require_fresh_vision))
        if not bool(response.success):
            rospy.logerr(
                "[usb_c_insertion] event=preinsert_workflow_failed reason=overview_vision_failed message=%s",
                response.message,
            )
            return None
        rospy.loginfo(
            "[usb_c_insertion] event=preinsert_overview_port_pose x=%.4f y=%.4f z=%.4f",
            response.port_pose.pose.position.x,
            response.port_pose.pose.position.y,
            response.port_pose.pose.position.z,
        )
        return response.port_pose

    def plan_camera_pose_from_port(self, port_pose: PoseStamped) -> Optional[PoseStamped]:
        frame_id = port_pose.header.frame_id.strip() or self.base_frame
        if frame_id != self.base_frame:
            rospy.logerr(
                "[usb_c_insertion] event=preinsert_workflow_failed reason=unsupported_port_pose_frame frame=%s",
                frame_id,
            )
            return None
        tool_to_camera = self._tf.lookup_transform(self.tool_frame, self._refine_camera_frame)
        if tool_to_camera is None:
            rospy.logerr(
                "[usb_c_insertion] event=preinsert_workflow_failed reason=camera_tf_unavailable tool_frame=%s camera_frame=%s",
                self.tool_frame,
                self._refine_camera_frame,
            )
            return None
        current_tool_pose = self._tf.get_tool_pose_in_base()
        if current_tool_pose is None:
            rospy.logerr("[usb_c_insertion] event=preinsert_workflow_failed reason=current_tool_pose_unavailable")
            return None
        current_quaternion = self._pose_quaternion(current_tool_pose)

        try:
            target_camera_xyz, plane_normal_yaw = self._desired_camera_position_and_yaw(port_pose)
            target_tool_quaternion = self._keep_current_roll_pitch_with_target_tool_z_yaw(
                current_quaternion,
                plane_normal_yaw,
            )
            target_tool_xyz = self._camera_target_to_tool_position(
                target_camera_xyz,
                target_tool_quaternion,
                tool_to_camera,
            )
        except ValueError as exc:
            rospy.logerr(
                "[usb_c_insertion] event=preinsert_workflow_failed reason=plan_camera_pose_failed message=%s",
                exc,
            )
            return None

        target_pose = PoseStamped()
        target_pose.header.stamp = rospy.Time.now()
        target_pose.header.frame_id = self.base_frame
        target_pose.pose.position.x = target_tool_xyz[0]
        target_pose.pose.position.y = target_tool_xyz[1]
        target_pose.pose.position.z = target_tool_xyz[2]
        target_pose.pose.orientation.x = target_tool_quaternion[0]
        target_pose.pose.orientation.y = target_tool_quaternion[1]
        target_pose.pose.orientation.z = target_tool_quaternion[2]
        target_pose.pose.orientation.w = target_tool_quaternion[3]
        rospy.loginfo(
            "[usb_c_insertion] event=preinsert_camera_pose_planned camera_frame=%s distance=%.4f tool_x=%.4f tool_y=%.4f tool_z=%.4f",
            self._refine_camera_frame,
            self._refine_camera_distance,
            target_tool_xyz[0],
            target_tool_xyz[1],
            target_tool_xyz[2],
        )
        return target_pose

    def align_housing_yaw(self) -> bool:
        goal = AlignHousingYawGoal()
        goal.image_topic = str(rospy.get_param("~align_housing_yaw/image_topic", "")).strip()
        goal.cloud_topic = str(rospy.get_param("~align_housing_yaw/cloud_topic", "")).strip()
        goal.estimate_timeout = float(rospy.get_param("~align_housing_yaw/estimate_timeout", 3.0))
        goal.yaw_tolerance_rad = float(rospy.get_param("~align_housing_yaw/yaw_tolerance_rad", 0.0175))
        goal.max_iterations = int(rospy.get_param("~align_housing_yaw/max_iterations", 10))
        goal.max_yaw_step_rad = float(rospy.get_param("~align_housing_yaw/max_yaw_step_rad", 0.25))
        goal.settle_time = float(rospy.get_param("~align_housing_yaw/settle_time", self._move_settle_time))
        goal.move_timeout = float(rospy.get_param("~align_housing_yaw/move_timeout", self._move_timeout))
        self._align_client.send_goal(goal)
        finished = self._align_client.wait_for_result(
            rospy.Duration.from_sec(max(1.0, goal.max_iterations * (goal.move_timeout + goal.estimate_timeout + 7.0)))
        )
        if not finished:
            self._align_client.cancel_goal()
            rospy.logerr("[usb_c_insertion] event=preinsert_workflow_failed reason=align_housing_yaw_timeout")
            return False
        result = self._align_client.get_result()
        if result is None or not bool(result.success):
            message = result.message if result is not None else "no_result"
            error_code = result.error_code if result is not None else "align_no_result"
            rospy.logerr(
                "[usb_c_insertion] event=preinsert_workflow_failed reason=align_housing_yaw_failed error_code=%s message=%s",
                error_code,
                message,
            )
            return False
        rospy.loginfo(
            "[usb_c_insertion] event=preinsert_align_housing_yaw_complete iterations=%d final_error_deg=%.2f",
            int(result.iterations),
            math.degrees(result.final_yaw_error_rad),
        )
        return True

    def center_port_in_image(self) -> Optional[PoseStamped]:
        goal = CenterPortInImageGoal()
        goal.image_topic = str(rospy.get_param("~center_port/image_topic", "")).strip()
        goal.timeout = float(rospy.get_param("~center_port/timeout", 60.0))
        goal.pixel_tolerance = float(rospy.get_param("~center_port/pixel_tolerance", 2.0))
        goal.stable_time = float(rospy.get_param("~center_port/stable_time", 0.35))
        goal.max_velocity = float(rospy.get_param("~center_port/max_velocity", 0.012))
        goal.gain = float(rospy.get_param("~center_port/gain", 0.0001))
        goal.min_blob_area = float(rospy.get_param("~center_port/min_blob_area", 120.0))
        self._center_client.send_goal(goal)
        finished = self._center_client.wait_for_result(rospy.Duration.from_sec(max(1.0, goal.timeout + 5.0)))
        if not finished:
            self._center_client.cancel_goal()
            rospy.logerr("[usb_c_insertion] event=preinsert_workflow_failed reason=center_port_timeout")
            return None
        result = self._center_client.get_result()
        if result is None or not bool(result.success):
            message = result.message if result is not None else "no_result"
            error_code = result.error_code if result is not None else "center_no_result"
            rospy.logerr(
                "[usb_c_insertion] event=preinsert_workflow_failed reason=center_port_failed error_code=%s message=%s",
                error_code,
                message,
            )
            return None
        rospy.loginfo(
            "[usb_c_insertion] event=preinsert_center_port_complete error_norm=%.2f blob=(%.1f,%.1f)",
            result.error_norm,
            result.blob_center_x,
            result.blob_center_y,
        )
        return result.final_pose

    def estimate_housing_plane(self, label: str):
        goal = self._build_estimate_goal()
        self._estimate_client.send_goal(goal)
        finished = self._estimate_client.wait_for_result(rospy.Duration.from_sec(max(0.1, goal.timeout + 2.0)))
        if not finished:
            self._estimate_client.cancel_goal()
            rospy.logerr("[usb_c_insertion] event=preinsert_workflow_failed reason=estimate_plane_timeout label=%s", label)
            return None
        result = self._estimate_client.get_result()
        if result is None or not bool(result.success):
            message = result.message if result is not None else "no_result"
            error_code = result.error_code if result is not None else "estimate_no_result"
            rospy.logerr(
                "[usb_c_insertion] event=preinsert_workflow_failed reason=estimate_plane_failed label=%s error_code=%s message=%s",
                label,
                error_code,
                message,
            )
            return None
        rospy.loginfo(
            "[usb_c_insertion] event=preinsert_plane_estimated label=%s ratio=%.3f rms=%.4f marker_point_base=(%.4f,%.4f,%.4f) normal_base=(%.4f,%.4f,%.4f)",
            label,
            result.inlier_ratio,
            result.rms_error,
            result.marker_plane_point_base.point.x,
            result.marker_plane_point_base.point.y,
            result.marker_plane_point_base.point.z,
            result.plane_normal_base.x,
            result.plane_normal_base.y,
            result.plane_normal_base.z,
        )
        return result

    def verify_looming(self):
        goal = VerifyLoomingGoal()
        goal.image_topic = str(rospy.get_param("~looming/image_topic", "")).strip()
        goal.travel_distance = float(rospy.get_param("~looming/travel_distance", 0.025))
        goal.travel_speed = float(rospy.get_param("~looming/travel_speed", 0.006))
        goal.timeout = float(rospy.get_param("~looming/timeout", 8.0))
        goal.min_blob_area = float(rospy.get_param("~looming/min_blob_area", 120.0))
        goal.min_scale_ratio = float(rospy.get_param("~looming/min_scale_ratio", 1.12))
        goal.max_center_shift_px = float(rospy.get_param("~looming/max_center_shift_px", 10.0))
        goal.max_aspect_ratio_change = float(rospy.get_param("~looming/max_aspect_ratio_change", 0.35))
        goal.tool_z_direction_sign = float(rospy.get_param("~looming/tool_z_direction_sign", 1.0))
        self._looming_client.send_goal(goal)
        finished = self._looming_client.wait_for_result(rospy.Duration.from_sec(max(1.0, goal.timeout + 5.0)))
        if not finished:
            self._looming_client.cancel_goal()
            rospy.logerr("[usb_c_insertion] event=preinsert_workflow_failed reason=looming_timeout")
            return None
        result = self._looming_client.get_result()
        if result is None or not bool(result.success):
            message = result.message if result is not None else "no_result"
            error_code = result.error_code if result is not None else "looming_no_result"
            rospy.logerr(
                "[usb_c_insertion] event=preinsert_workflow_failed reason=looming_failed error_code=%s message=%s",
                error_code,
                message,
            )
            return None
        rospy.loginfo(
            "[usb_c_insertion] event=preinsert_looming_verified scale_ratio=%.3f center_shift_px=%.2f traveled=%.4f",
            result.scale_ratio,
            result.center_shift_px,
            result.traveled_distance,
        )
        return result

    def validate_plane_quality(self, plane_result, label: str) -> bool:
        min_ratio = float(rospy.get_param("~workflow/min_plane_inlier_ratio", 0.80))
        max_rms = float(rospy.get_param("~workflow/max_plane_rms_error", 0.004))
        if float(plane_result.inlier_ratio) < min_ratio:
            rospy.logerr(
                "[usb_c_insertion] event=preinsert_workflow_failed reason=plane_inlier_ratio_too_low label=%s ratio=%.3f min_ratio=%.3f",
                label,
                plane_result.inlier_ratio,
                min_ratio,
            )
            return False
        if float(plane_result.rms_error) > max_rms:
            rospy.logerr(
                "[usb_c_insertion] event=preinsert_workflow_failed reason=plane_rms_too_high label=%s rms=%.4f max_rms=%.4f",
                label,
                plane_result.rms_error,
                max_rms,
            )
            return False
        return True

    def build_updated_port_pose(self, plane_result) -> Optional[PoseStamped]:
        current_pose = self._tf.get_tool_pose_in_base()
        if current_pose is None:
            rospy.logerr("[usb_c_insertion] event=preinsert_workflow_failed reason=updated_port_pose_tool_pose_unavailable")
            return None
        port_pose = PoseStamped()
        port_pose.header.stamp = rospy.Time.now()
        port_pose.header.frame_id = self.base_frame
        port_pose.pose.position = plane_result.marker_plane_point_base.point
        port_pose.pose.orientation = current_pose.pose.orientation
        rospy.loginfo(
            "[usb_c_insertion] event=preinsert_updated_port_pose x=%.4f y=%.4f z=%.4f source=marker_plane_point_base",
            port_pose.pose.position.x,
            port_pose.pose.position.y,
            port_pose.pose.position.z,
        )
        return port_pose

    def plan_tcp_precontact_pose(self, port_pose: PoseStamped) -> Optional[PoseStamped]:
        current_pose = self._tf.get_tool_pose_in_base()
        if current_pose is None:
            rospy.logerr("[usb_c_insertion] event=preinsert_workflow_failed reason=precontact_tool_pose_unavailable")
            return None
        tool_quaternion = self._pose_quaternion(current_pose)
        offset_base = rotate_vector_by_quaternion(
            self._target_offset_tool_x + self._precontact_offset_tool_x,
            self._target_offset_tool_y + self._precontact_offset_tool_y,
            self._precontact_offset_tool_z,
            *tool_quaternion,
        )
        target_pose = PoseStamped()
        target_pose.header.stamp = rospy.Time.now()
        target_pose.header.frame_id = self.base_frame
        target_pose.pose.position.x = port_pose.pose.position.x + offset_base[0]
        target_pose.pose.position.y = port_pose.pose.position.y + offset_base[1]
        target_pose.pose.position.z = port_pose.pose.position.z + offset_base[2]
        target_pose.pose.orientation = current_pose.pose.orientation
        rospy.loginfo(
            "[usb_c_insertion] event=preinsert_tcp_precontact_planned x=%.4f y=%.4f z=%.4f offset_base=(%.4f,%.4f,%.4f)",
            target_pose.pose.position.x,
            target_pose.pose.position.y,
            target_pose.pose.position.z,
            offset_base[0],
            offset_base[1],
            offset_base[2],
        )
        return target_pose

    def _build_estimate_goal(self) -> EstimateHousingPlaneGoal:
        goal = EstimateHousingPlaneGoal()
        goal.image_topic = str(rospy.get_param("~housing_plane/image_topic", "")).strip()
        goal.cloud_topic = str(rospy.get_param("~housing_plane/cloud_topic", "")).strip()
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

    def _desired_camera_position_and_yaw(self, port_pose: PoseStamped):
        port_quaternion = self._pose_quaternion(port_pose)
        port_x_axis = rotate_vector_by_quaternion(1.0, 0.0, 0.0, *port_quaternion)
        camera_z_axis = normalize_vector((-port_x_axis[0], -port_x_axis[1], -port_x_axis[2]))
        camera_xyz = (
            port_pose.pose.position.x - camera_z_axis[0] * self._refine_camera_distance,
            port_pose.pose.position.y - camera_z_axis[1] * self._refine_camera_distance,
            port_pose.pose.position.z - camera_z_axis[2] * self._refine_camera_distance,
        )
        plane_normal_yaw = math.atan2(port_x_axis[1], port_x_axis[0])
        return camera_xyz, plane_normal_yaw

    def _keep_current_roll_pitch_with_target_tool_z_yaw(self, current_tool_quaternion, plane_normal_yaw: float):
        current_roll, current_pitch, current_yaw = euler_from_quaternion(current_tool_quaternion)
        tool_z_axis = rotate_vector_by_quaternion(0.0, 0.0, 1.0, *current_tool_quaternion)
        current_tool_z_yaw = self._yaw_of_projected_vector(tool_z_axis)
        desired_tool_z_yaw = normalize_angle(plane_normal_yaw + math.pi)
        raw_yaw_delta = normalize_angle(desired_tool_z_yaw - current_tool_z_yaw)
        applied_yaw_delta = normalize_angle(self._refine_yaw_delta_sign * raw_yaw_delta)
        applied_yaw_delta_deg = math.degrees(applied_yaw_delta)
        if abs(applied_yaw_delta_deg) > self._refine_yaw_max_delta_deg:
            raise ValueError(
                "camera_yaw_delta_exceeds_limit: applied_delta_deg=%.2f max_delta_deg=%.2f"
                % (applied_yaw_delta_deg, self._refine_yaw_max_delta_deg)
            )
        return quaternion_from_euler(current_roll, current_pitch, normalize_angle(current_yaw + applied_yaw_delta))

    @staticmethod
    def _camera_target_to_tool_position(camera_xyz, tool_quaternion, tool_to_camera):
        tool_camera_translation = (
            tool_to_camera.transform.translation.x,
            tool_to_camera.transform.translation.y,
            tool_to_camera.transform.translation.z,
        )
        offset_base = rotate_vector_by_quaternion(
            tool_camera_translation[0],
            tool_camera_translation[1],
            tool_camera_translation[2],
            *tool_quaternion,
        )
        return camera_xyz[0] - offset_base[0], camera_xyz[1] - offset_base[1], camera_xyz[2] - offset_base[2]

    @staticmethod
    def _yaw_of_projected_vector(vector_xyz) -> float:
        if math.hypot(vector_xyz[0], vector_xyz[1]) <= 1e-9:
            raise ValueError("projected_vector_too_small")
        return math.atan2(vector_xyz[1], vector_xyz[0])

    @staticmethod
    def _pose_quaternion(pose: PoseStamped):
        return normalize_quaternion(
            (
                pose.pose.orientation.x,
                pose.pose.orientation.y,
                pose.pose.orientation.z,
                pose.pose.orientation.w,
            )
        )

    @staticmethod
    def _move_feedback_callback(feedback) -> None:
        rospy.loginfo_throttle(
            1.0,
            "[usb_c_insertion] event=preinsert_move_feedback pos_err=%.4f ori_err=%.4f reached=%s",
            float(feedback.position_error),
            float(feedback.orientation_error),
            str(bool(feedback.reached_position and feedback.reached_orientation)).lower(),
        )
