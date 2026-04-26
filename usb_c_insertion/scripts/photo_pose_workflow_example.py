#!/usr/bin/env python3

from __future__ import annotations

import math
import os
import sys
from typing import Tuple

import actionlib
from geometry_msgs.msg import PoseStamped
import rospy

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from prepose_planner import (
    euler_from_quaternion,
    normalize_quaternion,
    normalize_vector,
    normalize_angle,
    quaternion_from_euler,
    rotate_vector_by_quaternion,
)
from tf_interface import TFInterface
from usb_c_insertion.msg import (
    ApplyYawCorrectionAction,
    ApplyYawCorrectionGoal,
    InsertCableAction,
    InsertCableGoal,
    MoveToPoseAction,
    MoveToPoseGoal,
    ProbeSurfaceAction,
    ProbeSurfaceGoal,
    SearchPortAction,
    SearchPortGoal,
    VerifyInsertionAction,
    VerifyInsertionGoal,
)
from usb_c_insertion.srv import ComputePrePose, ComputePrePoseRequest, RunVision, RunVisionRequest


class PhotoPoseWorkflowExample:
    """
    Minimal example for an external orchestration step.

    The node sends a photo pose goal to the move action, waits for the robot
    to settle, and then triggers the existing vision pipeline through the ROS
    service wrapper.
    """

    def __init__(self):
        self._action_name = str(rospy.get_param("~move_action_name", "move_to_pose")).strip()
        self._vision_service_name = str(rospy.get_param("~vision_service_name", "run_vision")).strip()
        self._prepose_service_name = str(rospy.get_param("~prepose_service_name", "compute_prepose")).strip()
        self._probe_action_name = str(rospy.get_param("~probe_action_name", "probe_surface")).strip()
        self._apply_yaw_action_name = str(
            rospy.get_param("~apply_yaw_action_name", "apply_yaw_correction")
        ).strip()
        self._search_port_action_name = str(
            rospy.get_param("~search_port_action_name", "search_port")
        ).strip()
        self._insert_cable_action_name = str(
            rospy.get_param(
                "~insert_cable_action_name",
                rospy.get_param("~insert/action_name", "insert_cable"),
            )
        ).strip()
        self._verify_insertion_action_name = str(
            rospy.get_param(
                "~verify_insertion_action_name",
                rospy.get_param("~verify/action_name", "verify_insertion"),
            )
        ).strip()
        self._base_frame = str(rospy.get_param("~frames/base_frame", "base_link"))

        self._settle_time = float(rospy.get_param("~photo_pose/settle_time", 0.4))
        self._timeout = float(rospy.get_param("~photo_pose/timeout", 20.0))
        self._require_fresh_vision = bool(rospy.get_param("~photo_pose/require_fresh_vision_result", True))
        self._refine_vision_enabled = bool(rospy.get_param("~workflow/refine_vision_enabled", True))
        self._refine_vision_required = bool(rospy.get_param("~workflow/refine_vision_required", True))
        self._refine_camera_frame = str(
            rospy.get_param("~workflow/refine_camera_frame", "zedm_left_camera_optical_frame")
        ).strip()
        self._tool_frame = str(rospy.get_param("~frames/tool_frame", "tool0_controller")).strip()
        self._refine_camera_distance = float(rospy.get_param("~workflow/refine_camera_distance", 0.12))
        self._refine_yaw_delta_sign = float(rospy.get_param("~workflow/refine_yaw_delta_sign", 1.0))
        self._refine_yaw_max_delta_deg = float(rospy.get_param("~workflow/refine_yaw_max_delta_deg", 45.0))
        self._yaw_correction_use_current_position = bool(
            rospy.get_param("~workflow/yaw_correction_use_current_position", True)
        )

        self._goal = self._load_goal_pose()
        self._tf = TFInterface()
        self._client = actionlib.SimpleActionClient(self._action_name, MoveToPoseAction)
        self._probe_client = actionlib.SimpleActionClient(self._probe_action_name, ProbeSurfaceAction)
        self._apply_yaw_client = actionlib.SimpleActionClient(
            self._apply_yaw_action_name,
            ApplyYawCorrectionAction,
        )
        self._search_port_client = actionlib.SimpleActionClient(
            self._search_port_action_name,
            SearchPortAction,
        )
        self._insert_cable_client = actionlib.SimpleActionClient(
            self._insert_cable_action_name,
            InsertCableAction,
        )
        self._verify_insertion_client = actionlib.SimpleActionClient(
            self._verify_insertion_action_name,
            VerifyInsertionAction,
        )

    def run(self) -> bool:
        rospy.loginfo(
            "[usb_c_insertion] event=photo_pose_workflow_start action=%s service=%s",
            self._action_name,
            self._vision_service_name,
        )

        if not self._client.wait_for_server(rospy.Duration.from_sec(5.0)):
            rospy.logerr("[usb_c_insertion] event=photo_pose_workflow_failed reason=move_action_unavailable")
            return False

        if not self._move_to_photo_pose():
            return False

        port_pose = self._run_vision("coarse")
        if port_pose is None:
            return False

        refined_port_pose = self._run_refined_vision(port_pose)
        if refined_port_pose is None:
            return False
        port_pose = refined_port_pose

        pre_pose = self._compute_prepose(port_pose)
        if pre_pose is None:
            return False

        if not self._move_to_named_pose(pre_pose, "prepose"):
            return False

        yaw_correction = self._probe_surface(port_pose)
        if yaw_correction is None:
            return False

        yaw_reference_pose = self._build_yaw_correction_reference_pose(pre_pose)
        if yaw_reference_pose is None:
            return False

        corrected_pose = self._apply_yaw_correction(yaw_reference_pose, yaw_correction)
        if corrected_pose is None:
            return False

        search_result = self._search_port(port_pose, corrected_pose)
        if search_result is None or not bool(search_result.port_found):
            return False

        insert_result = self._insert_cable(search_result, corrected_pose)
        if insert_result is None or not bool(insert_result.success):
            return False

        return self._verify_insertion()

    def _move_to_photo_pose(self) -> bool:
        return self._move_to_named_pose(self._goal, "photo_pose")

    def _move_to_named_pose(self, pose: PoseStamped, pose_name: str) -> bool:
        goal = MoveToPoseGoal()
        goal.target_pose = pose
        goal.settle_time = self._settle_time
        goal.timeout = self._timeout

        rospy.loginfo(
            "[usb_c_insertion] event=move_goal_sent name=%s x=%.4f y=%.4f z=%.4f",
            pose_name,
            goal.target_pose.pose.position.x,
            goal.target_pose.pose.position.y,
            goal.target_pose.pose.position.z,
        )
        self._client.send_goal(goal, feedback_cb=self._feedback_callback)

        finished = self._client.wait_for_result(rospy.Duration.from_sec(max(1.0, self._timeout + 5.0)))
        if not finished:
            self._client.cancel_goal()
            rospy.logerr("[usb_c_insertion] event=photo_pose_workflow_failed reason=move_goal_wait_timeout name=%s", pose_name)
            return False

        result = self._client.get_result()
        state = self._client.get_state()
        if result is None or not bool(result.success):
            message = result.message if result is not None else "no_result"
            error_code = result.error_code if result is not None else "move_goal_no_result"
            rospy.logerr(
                "[usb_c_insertion] event=photo_pose_workflow_failed reason=move_goal_failed name=%s state=%d error_code=%s message=%s",
                pose_name,
                int(state),
                error_code,
                message,
            )
            return False

        rospy.loginfo(
            "[usb_c_insertion] event=move_pose_reached name=%s final_x=%.4f final_y=%.4f final_z=%.4f",
            pose_name,
            result.final_pose.pose.position.x,
            result.final_pose.pose.position.y,
            result.final_pose.pose.position.z,
        )
        return True

    def _run_vision(self, label: str) -> PoseStamped | None:
        rospy.wait_for_service(self._vision_service_name, timeout=5.0)
        service = rospy.ServiceProxy(self._vision_service_name, RunVision)
        request = RunVisionRequest(require_fresh_result=self._require_fresh_vision)
        response = service(request)

        if not response.success:
            rospy.logerr(
                "[usb_c_insertion] event=photo_pose_workflow_failed reason=vision_failed message=%s",
                response.message,
            )
            return None

        rospy.loginfo(
            "[usb_c_insertion] event=vision_result_received label=%s path=%s x=%.4f y=%.4f z=%.4f",
            label,
            response.json_path,
            response.port_pose.pose.position.x,
            response.port_pose.pose.position.y,
            response.port_pose.pose.position.z,
        )
        return response.port_pose

    def _run_refined_vision(self, coarse_port_pose: PoseStamped) -> PoseStamped | None:
        if not self._refine_vision_enabled:
            return coarse_port_pose

        camera_target = self._compute_refined_camera_target(coarse_port_pose)
        if camera_target is None:
            if self._refine_vision_required:
                return None
            rospy.logwarn("[usb_c_insertion] event=refined_vision_skipped reason=camera_target_unavailable")
            return coarse_port_pose

        if not self._move_to_named_pose(camera_target, "refined_camera_pose"):
            if self._refine_vision_required:
                return None
            rospy.logwarn("[usb_c_insertion] event=refined_vision_skipped reason=refined_camera_move_failed")
            return coarse_port_pose

        refined_port_pose = self._run_vision("refined")
        if refined_port_pose is None:
            if self._refine_vision_required:
                return None
            rospy.logwarn("[usb_c_insertion] event=refined_vision_skipped reason=refined_vision_failed")
            return coarse_port_pose

        return refined_port_pose

    def _compute_refined_camera_target(self, port_pose: PoseStamped) -> PoseStamped | None:
        frame_id = port_pose.header.frame_id.strip() or self._base_frame
        if frame_id != self._base_frame:
            rospy.logerr(
                "[usb_c_insertion] event=photo_pose_workflow_failed reason=refined_camera_unsupported_port_frame frame=%s",
                frame_id,
            )
            return None

        tool_to_camera = self._tf.lookup_transform(self._tool_frame, self._refine_camera_frame)
        if tool_to_camera is None:
            rospy.logerr(
                "[usb_c_insertion] event=photo_pose_workflow_failed reason=refined_camera_tf_unavailable tool_frame=%s camera_frame=%s",
                self._tool_frame,
                self._refine_camera_frame,
            )
            return None

        current_tool_pose = self._tf.get_tool_pose_in_base()
        if current_tool_pose is None:
            rospy.logerr(
                "[usb_c_insertion] event=photo_pose_workflow_failed reason=refined_camera_tool_pose_unavailable"
            )
            return None

        tool_quaternion = normalize_quaternion(
            (
                current_tool_pose.pose.orientation.x,
                current_tool_pose.pose.orientation.y,
                current_tool_pose.pose.orientation.z,
                current_tool_pose.pose.orientation.w,
            )
        )

        try:
            target_camera_xyz, plane_normal_yaw = self._compute_desired_camera_position_and_yaw(port_pose)
            tool_quaternion = self._keep_current_roll_pitch_with_target_tool_z_yaw(
                tool_quaternion,
                plane_normal_yaw,
            )
            tool_xyz = self._camera_target_to_tool_position(
                target_camera_xyz,
                tool_quaternion,
                tool_to_camera,
            )
        except ValueError as exc:
            rospy.logerr(
                "[usb_c_insertion] event=photo_pose_workflow_failed reason=refined_camera_pose_failed message=%s",
                exc,
            )
            return None

        target_pose = PoseStamped()
        target_pose.header.stamp = rospy.Time.now()
        target_pose.header.frame_id = self._base_frame
        target_pose.pose.position.x = tool_xyz[0]
        target_pose.pose.position.y = tool_xyz[1]
        target_pose.pose.position.z = tool_xyz[2]
        target_pose.pose.orientation.x = tool_quaternion[0]
        target_pose.pose.orientation.y = tool_quaternion[1]
        target_pose.pose.orientation.z = tool_quaternion[2]
        target_pose.pose.orientation.w = tool_quaternion[3]

        rospy.loginfo(
            "[usb_c_insertion] event=refined_camera_target_computed camera_frame=%s distance=%.4f tool_x=%.4f tool_y=%.4f tool_z=%.4f orientation_source=current_roll_pitch_target_yaw",
            self._refine_camera_frame,
            self._refine_camera_distance,
            tool_xyz[0],
            tool_xyz[1],
            tool_xyz[2],
        )
        return target_pose

    def _compute_desired_camera_position_and_yaw(
        self,
        port_pose: PoseStamped,
    ) -> tuple[Tuple[float, float, float], float]:
        port_quaternion = (
            port_pose.pose.orientation.x,
            port_pose.pose.orientation.y,
            port_pose.pose.orientation.z,
            port_pose.pose.orientation.w,
        )
        port_x_axis = rotate_vector_by_quaternion(1.0, 0.0, 0.0, *port_quaternion)
        port_y_axis = rotate_vector_by_quaternion(0.0, 1.0, 0.0, *port_quaternion)
        camera_z_axis = normalize_vector((-port_x_axis[0], -port_x_axis[1], -port_x_axis[2]))
        camera_xyz = (
            port_pose.pose.position.x - camera_z_axis[0] * self._refine_camera_distance,
            port_pose.pose.position.y - camera_z_axis[1] * self._refine_camera_distance,
            port_pose.pose.position.z - camera_z_axis[2] * self._refine_camera_distance,
        )
        plane_normal_yaw = math.atan2(port_x_axis[1], port_x_axis[0])
        desired_tool_z_yaw = normalize_angle(plane_normal_yaw + math.pi)
        rospy.loginfo(
            "[usb_c_insertion] event=refined_camera_plane_orientation_z frame=%s plane_normal_yaw_deg=%.2f desired_tool_z_yaw_deg=%.2f plane_tangent_yaw_deg=%.2f port_x_axis=(%.4f,%.4f,%.4f) port_y_axis=(%.4f,%.4f,%.4f) port_q=(%.4f,%.4f,%.4f,%.4f)",
            self._base_frame,
            math.degrees(plane_normal_yaw),
            math.degrees(desired_tool_z_yaw),
            math.degrees(self._yaw_of_projected_vector(port_y_axis)),
            port_x_axis[0],
            port_x_axis[1],
            port_x_axis[2],
            port_y_axis[0],
            port_y_axis[1],
            port_y_axis[2],
            port_quaternion[0],
            port_quaternion[1],
            port_quaternion[2],
            port_quaternion[3],
        )
        return camera_xyz, plane_normal_yaw

    def _keep_current_roll_pitch_with_target_tool_z_yaw(
        self,
        current_tool_quaternion,
        plane_normal_yaw: float,
    ) -> Tuple[float, float, float, float]:
        current_roll, current_pitch, current_yaw = euler_from_quaternion(current_tool_quaternion)
        tool_x_axis = rotate_vector_by_quaternion(1.0, 0.0, 0.0, *current_tool_quaternion)
        tool_y_axis = rotate_vector_by_quaternion(0.0, 1.0, 0.0, *current_tool_quaternion)
        tool_z_axis = rotate_vector_by_quaternion(0.0, 0.0, 1.0, *current_tool_quaternion)
        current_tool_z_yaw = self._yaw_of_projected_vector(tool_z_axis)
        desired_tool_z_yaw = normalize_angle(plane_normal_yaw + math.pi)
        rospy.loginfo(
            "[usb_c_insertion] event=refined_camera_tcp_orientation_z frame=%s tcp_euler_yaw_deg=%.2f tool_x_yaw_deg=%.2f tool_y_yaw_deg=%.2f tool_z_yaw_deg=%.2f tool_x_axis=(%.4f,%.4f,%.4f) tool_y_axis=(%.4f,%.4f,%.4f) tool_z_axis=(%.4f,%.4f,%.4f) tool_q=(%.4f,%.4f,%.4f,%.4f)",
            self._base_frame,
            math.degrees(current_yaw),
            math.degrees(self._yaw_of_projected_vector(tool_x_axis)),
            math.degrees(self._yaw_of_projected_vector(tool_y_axis)),
            math.degrees(current_tool_z_yaw),
            tool_x_axis[0],
            tool_x_axis[1],
            tool_x_axis[2],
            tool_y_axis[0],
            tool_y_axis[1],
            tool_y_axis[2],
            tool_z_axis[0],
            tool_z_axis[1],
            tool_z_axis[2],
            current_tool_quaternion[0],
            current_tool_quaternion[1],
            current_tool_quaternion[2],
            current_tool_quaternion[3],
        )
        raw_yaw_delta = normalize_angle(desired_tool_z_yaw - current_tool_z_yaw)
        applied_yaw_delta = normalize_angle(self._refine_yaw_delta_sign * raw_yaw_delta)
        applied_yaw_delta_deg = math.degrees(applied_yaw_delta)
        corrected_yaw = normalize_angle(current_yaw + applied_yaw_delta)
        rospy.loginfo(
            "[usb_c_insertion] event=refined_camera_yaw_delta current_tcp_yaw_deg=%.2f current_tool_z_yaw_deg=%.2f plane_normal_yaw_deg=%.2f desired_tool_z_yaw_deg=%.2f raw_delta_deg=%.2f applied_delta_deg=%.2f sign=%.1f max_delta_deg=%.2f",
            math.degrees(current_yaw),
            math.degrees(current_tool_z_yaw),
            math.degrees(plane_normal_yaw),
            math.degrees(desired_tool_z_yaw),
            math.degrees(raw_yaw_delta),
            applied_yaw_delta_deg,
            self._refine_yaw_delta_sign,
            self._refine_yaw_max_delta_deg,
        )
        if abs(applied_yaw_delta_deg) > self._refine_yaw_max_delta_deg:
            raise ValueError(
                "refined_camera_yaw_delta_exceeds_limit: applied_delta_deg=%.2f max_delta_deg=%.2f"
                % (applied_yaw_delta_deg, self._refine_yaw_max_delta_deg)
        )
        return quaternion_from_euler(current_roll, current_pitch, corrected_yaw)

    @staticmethod
    def _yaw_of_projected_vector(vector_xyz) -> float:
        if math.hypot(vector_xyz[0], vector_xyz[1]) <= 1e-9:
            return float("nan")
        return math.atan2(vector_xyz[1], vector_xyz[0])

    def _camera_target_to_tool_position(
        self,
        camera_xyz,
        tool_quaternion,
        tool_to_camera,
    ) -> Tuple[float, float, float]:
        tool_camera_translation = (
            tool_to_camera.transform.translation.x,
            tool_to_camera.transform.translation.y,
            tool_to_camera.transform.translation.z,
        )
        tool_camera_translation_in_base = rotate_vector_by_quaternion(
            tool_camera_translation[0],
            tool_camera_translation[1],
            tool_camera_translation[2],
            *tool_quaternion,
        )
        tool_xyz = (
            camera_xyz[0] - tool_camera_translation_in_base[0],
            camera_xyz[1] - tool_camera_translation_in_base[1],
            camera_xyz[2] - tool_camera_translation_in_base[2],
        )
        return tool_xyz

    def _compute_prepose(self, port_pose: PoseStamped) -> PoseStamped | None:
        rospy.wait_for_service(self._prepose_service_name, timeout=5.0)
        service = rospy.ServiceProxy(self._prepose_service_name, ComputePrePose)
        request = ComputePrePoseRequest(port_pose=port_pose)
        response = service(request)

        if not response.success:
            rospy.logerr(
                "[usb_c_insertion] event=photo_pose_workflow_failed reason=compute_prepose_failed message=%s",
                response.message,
            )
            return None

        rospy.loginfo(
            "[usb_c_insertion] event=prepose_received x=%.4f y=%.4f z=%.4f",
            response.pre_pose.pose.position.x,
            response.pre_pose.pose.position.y,
            response.pre_pose.pose.position.z,
        )
        return response.pre_pose

    def _feedback_callback(self, feedback) -> None:
        rospy.loginfo_throttle(
            1.0,
            "[usb_c_insertion] event=photo_pose_feedback pos_err=%.4f ori_err=%.4f reached_position=%s reached_orientation=%s",
            float(feedback.position_error),
            float(feedback.orientation_error),
            str(bool(feedback.reached_position)).lower(),
            str(bool(feedback.reached_orientation)).lower(),
        )

    def _build_yaw_correction_reference_pose(self, pre_pose: PoseStamped) -> PoseStamped | None:
        reference_pose = PoseStamped()
        reference_pose.header.stamp = rospy.Time.now()
        reference_pose.header.frame_id = self._base_frame
        reference_pose.pose.orientation = pre_pose.pose.orientation

        if not self._yaw_correction_use_current_position:
            reference_pose.pose.position = pre_pose.pose.position
            return reference_pose

        current_tool_pose = self._tf.get_tool_pose_in_base()
        if current_tool_pose is None:
            rospy.logerr(
                "[usb_c_insertion] event=photo_pose_workflow_failed reason=yaw_correction_reference_pose_unavailable"
            )
            return None

        reference_pose.pose.position = current_tool_pose.pose.position
        rospy.loginfo(
            "[usb_c_insertion] event=yaw_correction_reference_pose_selected source=current_tcp x=%.4f y=%.4f z=%.4f",
            reference_pose.pose.position.x,
            reference_pose.pose.position.y,
            reference_pose.pose.position.z,
        )
        return reference_pose

    def _probe_surface(self, port_pose: PoseStamped) -> float | None:
        if not self._probe_client.wait_for_server(rospy.Duration.from_sec(5.0)):
            rospy.logerr("[usb_c_insertion] event=photo_pose_workflow_failed reason=probe_action_unavailable")
            return None

        goal = ProbeSurfaceGoal(port_pose=port_pose)
        rospy.loginfo("[usb_c_insertion] event=probe_surface_goal_sent")
        self._probe_client.send_goal(goal)
        finished = self._probe_client.wait_for_result(rospy.Duration.from_sec(45.0))
        if not finished:
            self._probe_client.cancel_goal()
            rospy.logerr("[usb_c_insertion] event=photo_pose_workflow_failed reason=probe_action_timeout")
            return None

        result = self._probe_client.get_result()
        if result is None or not result.success:
            message = result.message if result is not None else "no_result"
            error_code = result.error_code if result is not None else "probe_action_no_result"
            failure_reason = result.failure_reason if result is not None else ""
            rospy.logerr(
                "[usb_c_insertion] event=photo_pose_workflow_failed reason=probe_action_failed error_code=%s failure_reason=%s message=%s",
                error_code,
                failure_reason,
                message,
            )
            return None

        rospy.loginfo(
            "[usb_c_insertion] event=probe_surface_complete surface_found=%s yaw_correction_rad=%.4f yaw_correction_deg=%.2f",
            str(bool(result.surface_found)).lower(),
            float(result.yaw_correction_rad),
            math.degrees(float(result.yaw_correction_rad)),
        )
        return float(result.yaw_correction_rad)

    def _apply_yaw_correction(
        self,
        reference_pose: PoseStamped,
        yaw_correction_rad: float,
    ) -> PoseStamped | None:
        if not self._apply_yaw_client.wait_for_server(rospy.Duration.from_sec(5.0)):
            rospy.logerr("[usb_c_insertion] event=photo_pose_workflow_failed reason=apply_yaw_action_unavailable")
            return None

        goal = ApplyYawCorrectionGoal()
        goal.reference_pose = reference_pose
        goal.yaw_correction_rad = float(yaw_correction_rad)
        goal.settle_time = self._settle_time
        goal.timeout = self._timeout

        rospy.loginfo(
            "[usb_c_insertion] event=apply_yaw_correction_goal_sent yaw_correction_rad=%.4f yaw_correction_deg=%.2f",
            goal.yaw_correction_rad,
            math.degrees(goal.yaw_correction_rad),
        )
        self._apply_yaw_client.send_goal(goal)
        finished = self._apply_yaw_client.wait_for_result(
            rospy.Duration.from_sec(max(1.0, self._timeout + 5.0))
        )
        if not finished:
            self._apply_yaw_client.cancel_goal()
            rospy.logerr("[usb_c_insertion] event=photo_pose_workflow_failed reason=apply_yaw_action_timeout")
            return None

        result = self._apply_yaw_client.get_result()
        if result is None or not bool(result.success):
            message = result.message if result is not None else "no_result"
            error_code = result.error_code if result is not None else "apply_yaw_action_no_result"
            rospy.logerr(
                "[usb_c_insertion] event=photo_pose_workflow_failed reason=apply_yaw_action_failed error_code=%s message=%s",
                error_code,
                message,
            )
            return None

        rospy.loginfo(
            "[usb_c_insertion] event=apply_yaw_correction_complete x=%.4f y=%.4f z=%.4f",
            result.corrected_pose.pose.position.x,
            result.corrected_pose.pose.position.y,
            result.corrected_pose.pose.position.z,
        )
        return result.corrected_pose

    def _search_port(self, port_pose: PoseStamped, corrected_pose: PoseStamped):
        if not self._search_port_client.wait_for_server(rospy.Duration.from_sec(5.0)):
            rospy.logerr("[usb_c_insertion] event=photo_pose_workflow_failed reason=search_port_action_unavailable")
            return None

        goal = SearchPortGoal()
        goal.port_pose = port_pose
        goal.reference_pose = corrected_pose
        goal.timeout = float(
            rospy.get_param(
                "~photo_pose/search_timeout",
                rospy.get_param("~search/search_timeout", 50.0),
            )
        )

        rospy.loginfo("[usb_c_insertion] event=search_port_goal_sent")
        self._search_port_client.send_goal(goal, feedback_cb=self._search_feedback_callback)
        finished = self._search_port_client.wait_for_result(rospy.Duration.from_sec(max(1.0, goal.timeout + 15.0)))
        if not finished:
            self._search_port_client.cancel_goal()
            rospy.logerr("[usb_c_insertion] event=photo_pose_workflow_failed reason=search_port_action_timeout")
            return None

        result = self._search_port_client.get_result()
        if result is None or not bool(result.success):
            message = result.message if result is not None else "no_result"
            error_code = result.error_code if result is not None else "search_port_action_no_result"
            failure_reason = result.failure_reason if result is not None else ""
            rospy.logerr(
                "[usb_c_insertion] event=photo_pose_workflow_failed reason=search_port_action_failed error_code=%s failure_reason=%s message=%s",
                error_code,
                failure_reason,
                message,
            )
            return None

        rospy.loginfo(
            "[usb_c_insertion] event=search_port_complete port_found=%s inserted_depth=%.4f contact_force=%.3f completed_steps=%d",
            str(bool(result.port_found)).lower(),
            float(result.inserted_depth),
            float(result.contact_force),
            int(result.completed_steps),
        )
        return result

    def _insert_cable(self, search_result, reference_pose: PoseStamped):
        if not self._insert_cable_client.wait_for_server(rospy.Duration.from_sec(5.0)):
            rospy.logerr("[usb_c_insertion] event=photo_pose_workflow_failed reason=insert_cable_action_unavailable")
            return None

        goal = InsertCableGoal()
        goal.reference_point = search_result.wall_contact_point
        goal.reference_pose = reference_pose
        goal.timeout = float(
            rospy.get_param(
                "~photo_pose/insert_timeout",
                rospy.get_param("~insert/force_control_timeout", 8.0),
            )
        )
        goal.zero_ft_before_insert = bool(rospy.get_param("~insert/zero_ft_before_insert", False))

        rospy.loginfo("[usb_c_insertion] event=insert_cable_goal_sent")
        self._insert_cable_client.send_goal(goal, feedback_cb=self._insert_feedback_callback)
        finished = self._insert_cable_client.wait_for_result(
            rospy.Duration.from_sec(max(1.0, goal.timeout + 5.0))
        )
        if not finished:
            self._insert_cable_client.cancel_goal()
            rospy.logerr("[usb_c_insertion] event=photo_pose_workflow_failed reason=insert_cable_action_timeout")
            return None

        result = self._insert_cable_client.get_result()
        if result is None or not bool(result.success):
            message = result.message if result is not None else "no_result"
            error_code = result.error_code if result is not None else "insert_cable_action_no_result"
            failure_reason = result.failure_reason if result is not None else ""
            rospy.logerr(
                "[usb_c_insertion] event=photo_pose_workflow_failed reason=insert_cable_action_failed error_code=%s failure_reason=%s message=%s",
                error_code,
                failure_reason,
                message,
            )
            return result

        rospy.loginfo(
            "[usb_c_insertion] event=insert_cable_complete inserted_depth=%.4f contact_force=%.3f message=%s",
            float(result.inserted_depth),
            float(result.contact_force),
            result.message,
        )
        return result

    def _verify_insertion(self) -> bool:
        if not self._verify_insertion_client.wait_for_server(rospy.Duration.from_sec(5.0)):
            rospy.logerr("[usb_c_insertion] event=photo_pose_workflow_failed reason=verify_insertion_action_unavailable")
            return False

        goal = VerifyInsertionGoal()
        goal.timeout = float(
            rospy.get_param(
                "~photo_pose/verify_timeout",
                rospy.get_param("~verify/action_timeout", 2.0),
            )
        )
        goal.zero_ft_before_verify = bool(rospy.get_param("~verify/zero_ft_before_verify", False))

        rospy.loginfo("[usb_c_insertion] event=verify_insertion_goal_sent")
        self._verify_insertion_client.send_goal(goal, feedback_cb=self._verify_feedback_callback)
        finished = self._verify_insertion_client.wait_for_result(
            rospy.Duration.from_sec(max(1.0, goal.timeout * 2.0 + 3.0))
        )
        if not finished:
            self._verify_insertion_client.cancel_goal()
            rospy.logerr("[usb_c_insertion] event=photo_pose_workflow_failed reason=verify_insertion_action_timeout")
            return False

        result = self._verify_insertion_client.get_result()
        if result is None or not bool(result.success) or not bool(result.verified):
            message = result.message if result is not None else "no_result"
            error_code = result.error_code if result is not None else "verify_insertion_action_no_result"
            failure_reason = result.failure_reason if result is not None else ""
            rospy.logerr(
                "[usb_c_insertion] event=photo_pose_workflow_failed reason=verify_insertion_failed error_code=%s failure_reason=%s message=%s",
                error_code,
                failure_reason,
                message,
            )
            return False

        rospy.loginfo(
            "[usb_c_insertion] event=verify_insertion_complete counterforce_y=%.3f counterforce_z=%.3f",
            float(result.counterforce_y),
            float(result.counterforce_z),
        )
        return True

    def _insert_feedback_callback(self, feedback) -> None:
        rospy.loginfo_throttle(
            1.0,
            "[usb_c_insertion] event=insert_cable_feedback stage=%s inserted_depth=%.4f contact_force=%.3f elapsed=%.2f",
            feedback.stage,
            float(feedback.inserted_depth),
            float(feedback.contact_force),
            float(feedback.elapsed),
        )

    def _verify_feedback_callback(self, feedback) -> None:
        rospy.loginfo_throttle(
            1.0,
            "[usb_c_insertion] event=verify_insertion_feedback stage=%s counterforce_y=%.3f counterforce_z=%.3f elapsed=%.2f",
            feedback.stage,
            float(feedback.counterforce_y),
            float(feedback.counterforce_z),
            float(feedback.elapsed),
        )

    def _search_feedback_callback(self, feedback) -> None:
        rospy.loginfo_throttle(
            1.0,
            "[usb_c_insertion] event=search_port_feedback stage=%s step=%d total=%d inserted_depth=%.4f contact_force=%.3f elapsed=%.2f",
            feedback.stage,
            int(feedback.current_step),
            int(feedback.total_steps),
            float(feedback.inserted_depth),
            float(feedback.contact_force),
            float(feedback.elapsed),
        )

    def _load_goal_pose(self) -> PoseStamped:
        pose = PoseStamped()
        pose.header.frame_id = str(rospy.get_param("~photo_pose/frame_id", self._base_frame))
        pose.pose.position.x = float(rospy.get_param("~photo_pose/x", 0.50))
        pose.pose.position.y = float(rospy.get_param("~photo_pose/y", 0.1))
        pose.pose.position.z = float(rospy.get_param("~photo_pose/z", 0.3))

        qx = float(rospy.get_param("~photo_pose/qx", 0.708))
        qy = float(rospy.get_param("~photo_pose/qy", 0.0))
        qz = float(rospy.get_param("~photo_pose/qz", 0.0))
        qw = float(rospy.get_param("~photo_pose/qw", 0.705))
        norm = math.sqrt(qx * qx + qy * qy + qz * qz + qw * qw)
        if norm <= 1e-9:
            qx, qy, qz, qw = 0.0, 0.0, 0.0, 1.0
        else:
            qx /= norm
            qy /= norm
            qz /= norm
            qw /= norm

        pose.pose.orientation.x = qx
        pose.pose.orientation.y = qy
        pose.pose.orientation.z = qz
        pose.pose.orientation.w = qw
        return pose


def main() -> None:
    rospy.init_node("usb_c_insertion_photo_pose_workflow_example")
    success = PhotoPoseWorkflowExample().run()
    if not success:
        sys.exit(1)


if __name__ == "__main__":
    main()
