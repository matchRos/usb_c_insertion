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
        self.base_frame = self._required_str_param("~frames/base_frame")
        self.tool_frame = self._required_str_param("~frames/tool_frame")

        self._move_action_name = self._required_str_param("~workflow/move_action_name")
        self._accurate_move_action_name = self._required_str_param("~workflow/accurate_move_action_name")
        self._vision_service_name = "run_vision"
        self._align_action_name = self._required_str_param("~align_housing_yaw/action_name")
        self._center_action_name = self._required_str_param("~center_port/action_name")
        self._estimate_action_name = self._required_str_param("~housing_plane/action_name")
        self._looming_action_name = self._required_str_param("~looming/action_name")

        self._move_settle_time = self._required_float_param("~photo_pose/settle_time")
        self._move_timeout = self._required_float_param("~photo_pose/timeout")
        self._accurate_move_settle_time = self._required_float_param("~workflow/accurate_move_settle_time")
        self._accurate_move_timeout = self._required_float_param("~workflow/accurate_move_timeout")
        self._require_fresh_vision = self._required_bool_param("~photo_pose/require_fresh_vision_result")
        self._refine_camera_frame = self._required_str_param("~workflow/refine_camera_frame")
        self._refine_camera_distance = self._required_float_param("~workflow/refine_camera_distance")
        self._refine_yaw_delta_sign = self._required_float_param("~workflow/refine_yaw_delta_sign")
        self._refine_yaw_max_delta_deg = self._required_float_param("~workflow/refine_yaw_max_delta_deg")
        self._precontact_offset_tool_x = self._required_float_param("~precontact/precontact_offset_tool_x")
        self._precontact_offset_tool_y = self._required_float_param("~precontact/precontact_offset_tool_y")
        self._precontact_offset_tool_z = self._required_float_param("~precontact/precontact_offset_tool_z")
        self._target_offset_tool_x = self._required_float_param("~precontact/target_offset_tool_x")
        self._target_offset_tool_y = self._required_float_param("~precontact/target_offset_tool_y")
        self._looming_tool_z_direction_sign = self._sign(
            self._required_float_param("~looming/tool_z_direction_sign")
        )
        self._enforce_precontact_standoff = self._required_bool_param(
            "~precontact/enforce_precontact_standoff"
        )
        self._min_precontact_standoff = abs(
            self._required_float_param("~precontact/min_precontact_standoff")
        )

        self._tf = TFInterface()
        self._move_client = actionlib.SimpleActionClient(self._move_action_name, MoveToPoseAction)
        self._accurate_move_client = actionlib.SimpleActionClient(
            self._accurate_move_action_name,
            MoveToPoseAction,
        )
        self._align_client = actionlib.SimpleActionClient(self._align_action_name, AlignHousingYawAction)
        self._center_client = actionlib.SimpleActionClient(self._center_action_name, CenterPortInImageAction)
        self._estimate_client = actionlib.SimpleActionClient(self._estimate_action_name, EstimateHousingPlaneAction)
        self._looming_client = actionlib.SimpleActionClient(self._looming_action_name, VerifyLoomingAction)
        self._last_center_port_result = None

    @staticmethod
    def _required_param(name: str):
        if name.startswith("~"):
            global_name = "/" + name[1:].lstrip("/")
            if rospy.has_param(global_name):
                return rospy.get_param(global_name)
        if rospy.has_param(name):
            return rospy.get_param(name)
        resolved_name = rospy.resolve_name(name)
        global_name = None
        global_available = False
        if name.startswith("~"):
            global_name = "/" + name[1:].lstrip("/")
            global_available = rospy.has_param(global_name)
        rospy.logerr(
            "[usb_c_insertion] event=preinsert_missing_required_param param=%s resolved_param=%s global_param=%s global_available=%s",
            name,
            resolved_name,
            global_name or "",
            str(global_available).lower(),
        )
        raise RuntimeError("Missing required ROS parameter: %s (%s)" % (name, resolved_name))

    @classmethod
    def _required_str_param(cls, name: str) -> str:
        return str(cls._required_param(name)).strip()

    @classmethod
    def _required_float_param(cls, name: str) -> float:
        return float(cls._required_param(name))

    @classmethod
    def _required_int_param(cls, name: str) -> int:
        return int(cls._required_param(name))

    @classmethod
    def _required_bool_param(cls, name: str) -> bool:
        value = cls._required_param(name)
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            normalized = value.strip().lower()
            if normalized in ("true", "1", "yes", "on"):
                return True
            if normalized in ("false", "0", "no", "off"):
                return False
            resolved_name = rospy.resolve_name(name)
            raise ValueError("Invalid boolean ROS parameter: %s (%s)=%r" % (name, resolved_name, value))
        return bool(value)

    def wait_for_dependencies(self) -> bool:
        checks = (
            (self._move_client, self._move_action_name, "move_action_unavailable"),
            (
                self._accurate_move_client,
                self._accurate_move_action_name,
                "accurate_move_action_unavailable",
            ),
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
        pose.header.frame_id = self._required_str_param("~photo_pose/frame_id")
        pose.pose.position.x = self._required_float_param("~photo_pose/x")
        pose.pose.position.y = self._required_float_param("~photo_pose/y")
        pose.pose.position.z = self._required_float_param("~photo_pose/z")
        qx = self._required_float_param("~photo_pose/qx")
        qy = self._required_float_param("~photo_pose/qy")
        qz = self._required_float_param("~photo_pose/qz")
        qw = self._required_float_param("~photo_pose/qw")
        qx, qy, qz, qw = normalize_quaternion((qx, qy, qz, qw))
        pose.pose.orientation.x = qx
        pose.pose.orientation.y = qy
        pose.pose.orientation.z = qz
        pose.pose.orientation.w = qw
        return pose

    def move_to_pose(
        self,
        pose: PoseStamped,
        name: str,
        timeout: Optional[float] = None,
        accurate: bool = False,
    ) -> Optional[PoseStamped]:
        goal = MoveToPoseGoal()
        goal.target_pose = pose
        goal.settle_time = self._accurate_move_settle_time if accurate else self._move_settle_time
        default_timeout = self._accurate_move_timeout if accurate else self._move_timeout
        goal.timeout = default_timeout if timeout is None else float(timeout)
        client = self._accurate_move_client if accurate else self._move_client
        action_name = self._accurate_move_action_name if accurate else self._move_action_name
        rospy.loginfo(
            "[usb_c_insertion] event=preinsert_move_goal_sent name=%s action=%s accurate=%s x=%.4f y=%.4f z=%.4f settle_time=%.2f timeout=%.2f",
            name,
            action_name,
            str(bool(accurate)).lower(),
            pose.pose.position.x,
            pose.pose.position.y,
            pose.pose.position.z,
            goal.settle_time,
            goal.timeout,
        )
        client.send_goal(goal, feedback_cb=self._move_feedback_callback)
        finished = client.wait_for_result(rospy.Duration.from_sec(max(1.0, goal.timeout + 5.0)))
        if not finished:
            client.cancel_goal()
            rospy.logerr("[usb_c_insertion] event=preinsert_workflow_failed reason=move_timeout name=%s", name)
            return None
        result = client.get_result()
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
        goal.image_topic = self._required_str_param("~align_housing_yaw/image_topic")
        goal.cloud_topic = self._required_str_param("~align_housing_yaw/cloud_topic")
        goal.estimate_timeout = self._required_float_param("~align_housing_yaw/estimate_timeout")
        goal.yaw_tolerance_rad = self._required_float_param("~align_housing_yaw/yaw_tolerance_rad")
        goal.max_iterations = self._required_int_param("~align_housing_yaw/max_iterations")
        goal.max_yaw_step_rad = self._required_float_param("~align_housing_yaw/max_yaw_step_rad")
        goal.settle_time = self._required_float_param("~align_housing_yaw/settle_time")
        goal.move_timeout = self._required_float_param("~align_housing_yaw/move_timeout")
        rospy.loginfo(
            "[usb_c_insertion] event=preinsert_align_housing_yaw_goal image_topic=%s cloud_topic=%s estimate_timeout=%.2f yaw_tolerance_rad=%.4f max_iterations=%d max_yaw_step_rad=%.4f settle_time=%.2f move_timeout=%.2f",
            goal.image_topic,
            goal.cloud_topic,
            goal.estimate_timeout,
            goal.yaw_tolerance_rad,
            int(goal.max_iterations),
            goal.max_yaw_step_rad,
            goal.settle_time,
            goal.move_timeout,
        )
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
        goal.image_topic = self._required_str_param("~center_port/image_topic")
        goal.timeout = self._required_float_param("~center_port/timeout")
        goal.pixel_tolerance = self._required_float_param("~center_port/pixel_tolerance")
        goal.stable_time = self._required_float_param("~center_port/stable_time")
        goal.max_velocity = self._required_float_param("~center_port/max_velocity")
        goal.gain = self._required_float_param("~center_port/gain")
        goal.min_blob_area = self._required_float_param("~center_port/min_blob_area")
        self._center_client.send_goal(goal)
        finished = self._center_client.wait_for_result(rospy.Duration.from_sec(max(1.0, goal.timeout + 5.0)))
        if not finished:
            self._center_client.cancel_goal()
            rospy.logerr("[usb_c_insertion] event=preinsert_workflow_failed reason=center_port_timeout")
            return None
        result = self._center_client.get_result()
        self._last_center_port_result = result
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

    def latest_center_port_result(self):
        return self._last_center_port_result

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
        goal.image_topic = self._required_str_param("~looming/image_topic")
        goal.travel_distance = self._required_float_param("~looming/travel_distance")
        goal.travel_speed = self._required_float_param("~looming/travel_speed")
        goal.timeout = self._required_float_param("~looming/timeout")
        goal.min_blob_area = self._required_float_param("~looming/min_blob_area")
        goal.min_scale_ratio = self._required_float_param("~looming/min_scale_ratio")
        goal.max_center_shift_px = self._required_float_param("~looming/max_center_shift_px")
        goal.max_aspect_ratio_change = self._required_float_param("~looming/max_aspect_ratio_change")
        goal.tool_z_direction_sign = self._required_float_param("~looming/tool_z_direction_sign")
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

    def recenter_after_looming_if_needed(self, looming_result) -> Tuple[bool, Optional[object]]:
        if not self._required_bool_param("~workflow/looming_recenter_enabled"):
            return True, None

        center_shift_px = float(getattr(looming_result, "center_shift_px", 0.0))
        min_shift_px = max(0.0, self._required_float_param("~workflow/looming_recenter_min_shift_px"))
        if center_shift_px < min_shift_px:
            rospy.loginfo(
                "[usb_c_insertion] event=preinsert_looming_recenter_skipped center_shift_px=%.2f min_shift_px=%.2f",
                center_shift_px,
                min_shift_px,
            )
            return True, None

        max_attempts = max(1, self._required_int_param("~workflow/looming_recenter_max_attempts"))
        required = self._required_bool_param("~workflow/looming_recenter_required")
        rospy.loginfo(
            "[usb_c_insertion] event=preinsert_looming_recenter_started center_shift_px=%.2f min_shift_px=%.2f max_attempts=%d",
            center_shift_px,
            min_shift_px,
            max_attempts,
        )

        for attempt in range(1, max_attempts + 1):
            final_pose = self.center_port_in_image()
            center_result = self.latest_center_port_result()
            if final_pose is not None and center_result is not None and bool(center_result.success):
                rospy.loginfo(
                    "[usb_c_insertion] event=preinsert_looming_recenter_complete attempt=%d error_norm=%.2f blob=(%.1f,%.1f)",
                    attempt,
                    center_result.error_norm,
                    center_result.blob_center_x,
                    center_result.blob_center_y,
                )
                return True, center_result
            rospy.logwarn(
                "[usb_c_insertion] event=preinsert_looming_recenter_attempt_failed attempt=%d",
                attempt,
            )

        if required:
            rospy.logerr("[usb_c_insertion] event=preinsert_workflow_failed reason=looming_recenter_failed")
            return False, None

        rospy.logwarn("[usb_c_insertion] event=preinsert_looming_recenter_failed continuing=true")
        return True, None

    def validate_plane_quality(self, plane_result, label: str) -> bool:
        min_ratio = self._required_float_param("~workflow/min_plane_inlier_ratio")
        max_rms = self._required_float_param("~workflow/max_plane_rms_error")
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
        standoff = self._precontact_standoff_distance()
        if not self._is_precontact_standoff_safe(standoff):
            return None

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
            "[usb_c_insertion] event=preinsert_tcp_precontact_planned x=%.4f y=%.4f z=%.4f offset_tool=(%.4f,%.4f,%.4f) standoff=%.4f offset_base=(%.4f,%.4f,%.4f)",
            target_pose.pose.position.x,
            target_pose.pose.position.y,
            target_pose.pose.position.z,
            self._target_offset_tool_x + self._precontact_offset_tool_x,
            self._target_offset_tool_y + self._precontact_offset_tool_y,
            self._precontact_offset_tool_z,
            standoff,
            offset_base[0],
            offset_base[1],
            offset_base[2],
        )
        return target_pose

    def _precontact_standoff_distance(self) -> float:
        return -self._precontact_offset_tool_z * self._looming_tool_z_direction_sign

    def _is_precontact_standoff_safe(self, standoff: float) -> bool:
        if not self._enforce_precontact_standoff:
            return True
        if standoff >= self._min_precontact_standoff:
            return True
        rospy.logerr(
            "[usb_c_insertion] event=preinsert_workflow_failed reason=unsafe_precontact_offset "
            "precontact_offset_tool_z=%.4f looming_tool_z_direction_sign=%.1f standoff=%.4f min_standoff=%.4f",
            self._precontact_offset_tool_z,
            self._looming_tool_z_direction_sign,
            standoff,
            self._min_precontact_standoff,
        )
        return False

    def _build_estimate_goal(self) -> EstimateHousingPlaneGoal:
        goal = EstimateHousingPlaneGoal()
        goal.image_topic = self._required_str_param("~housing_plane/image_topic")
        goal.cloud_topic = self._required_str_param("~housing_plane/cloud_topic")
        goal.timeout = self._required_float_param("~housing_plane/timeout")
        goal.min_blob_area = self._required_float_param("~housing_plane/min_blob_area")
        goal.roi_radius_px = self._required_int_param("~housing_plane/roi_radius_px")
        goal.roi_stride_px = self._required_int_param("~housing_plane/roi_stride_px")
        goal.depth_window_m = self._required_float_param("~housing_plane/depth_window_m")
        goal.ransac_iterations = self._required_int_param("~housing_plane/ransac_iterations")
        goal.ransac_distance_threshold = self._required_float_param("~housing_plane/ransac_distance_threshold")
        goal.min_inliers = self._required_int_param("~housing_plane/min_inliers")
        goal.use_svd_refit = self._required_bool_param("~housing_plane/use_svd_refit")
        goal.use_largest_component = self._required_bool_param("~housing_plane/use_largest_component")
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
    def _sign(value) -> float:
        return 1.0 if float(value) >= 0.0 else -1.0

    @staticmethod
    def _move_feedback_callback(feedback) -> None:
        rospy.loginfo_throttle(
            1.0,
            "[usb_c_insertion] event=preinsert_move_feedback pos_err=%.4f ori_err=%.4f reached=%s",
            float(feedback.position_error),
            float(feedback.orientation_error),
            str(bool(feedback.reached_position and feedback.reached_orientation)).lower(),
        )
