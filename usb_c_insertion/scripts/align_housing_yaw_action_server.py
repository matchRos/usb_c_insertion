#!/usr/bin/env python3

from __future__ import annotations

import math
import os
import sys
from typing import Optional, Tuple

import actionlib
from geometry_msgs.msg import PoseStamped, Vector3
import rospy

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from param_utils import (
    required_bool_param,
    required_float_param,
    required_int_param,
    required_str_param,
    required_vector_param,
)
from prepose_planner import (
    euler_from_quaternion,
    normalize_quaternion,
    quaternion_from_yaw,
    quaternion_multiply,
    rotate_vector_by_quaternion,
)
from tf_interface import TFInterface
from usb_c_insertion.msg import (
    AlignHousingYawAction,
    AlignHousingYawFeedback,
    AlignHousingYawResult,
    EstimateHousingPlaneAction,
    EstimateHousingPlaneGoal,
    MoveToPoseAction,
    MoveToPoseGoal,
)


class AlignHousingYawActionServer:
    """
    Iteratively align the current tool yaw to the measured housing plane.

    Each iteration estimates the local housing plane, compares the current
    tool approach axis against the opposite plane normal in base XY, applies a
    bounded yaw correction through move_to_pose, and then measures again.
    """

    def __init__(self):
        self._mirror_global_config_to_private_namespace()

        self._action_name = required_str_param("~align_housing_yaw/action_name")
        self._estimate_action_name = required_str_param("~align_housing_yaw/estimate_action_name")
        self._move_action_name = required_str_param("~align_housing_yaw/move_action_name")
        self._base_frame = required_str_param("~frames/base_frame")

        self._default_image_topic = required_str_param("~align_housing_yaw/image_topic")
        self._default_cloud_topic = required_str_param("~align_housing_yaw/cloud_topic")
        self._default_estimate_timeout = required_float_param("~align_housing_yaw/estimate_timeout")
        self._default_yaw_tolerance_rad = required_float_param("~align_housing_yaw/yaw_tolerance_rad")
        self._default_max_iterations = required_int_param("~align_housing_yaw/max_iterations")
        self._default_max_yaw_step_rad = required_float_param("~align_housing_yaw/max_yaw_step_rad")
        self._default_settle_time = required_float_param("~align_housing_yaw/settle_time")
        self._default_move_timeout = required_float_param("~align_housing_yaw/move_timeout")
        self._estimate_wait_timeout = required_float_param("~align_housing_yaw/estimate_wait_timeout")
        self._move_wait_timeout = required_float_param("~align_housing_yaw/move_wait_timeout")
        self._tool_axis = self._read_vector_param("~align_housing_yaw/tool_axis")
        self._target_axis_from_plane_normal_sign = required_float_param(
            "~align_housing_yaw/target_axis_from_plane_normal_sign"
        )

        self._tf = TFInterface()
        self._estimate_client = actionlib.SimpleActionClient(self._estimate_action_name, EstimateHousingPlaneAction)
        self._move_client = actionlib.SimpleActionClient(self._move_action_name, MoveToPoseAction)
        self._server = actionlib.SimpleActionServer(
            self._action_name,
            AlignHousingYawAction,
            execute_cb=self._execute,
            auto_start=False,
        )
        self._server.start()
        rospy.loginfo(
            "[usb_c_insertion] event=align_housing_yaw_action_ready action=%s estimate_action=%s move_action=%s yaw_tolerance_rad=%.4f max_iterations=%d max_yaw_step_rad=%.4f tool_axis=(%.3f,%.3f,%.3f) plane_normal_sign=%.1f",
            self._action_name,
            self._estimate_action_name,
            self._move_action_name,
            self._default_yaw_tolerance_rad,
            self._default_max_iterations,
            self._default_max_yaw_step_rad,
            self._tool_axis[0],
            self._tool_axis[1],
            self._tool_axis[2],
            self._target_axis_from_plane_normal_sign,
        )

    def _mirror_global_config_to_private_namespace(self) -> None:
        """
        Keep manually started action-server params in sync with global YAML.
        """
        namespaces = (
            "frames",
            "topics",
            "motion",
            "housing_plane",
            "align_housing_yaw",
        )
        mirrored = []
        for namespace in namespaces:
            private_name = "~%s" % namespace
            global_name = "/%s" % namespace
            if not rospy.has_param(global_name):
                continue
            global_value = rospy.get_param(global_name)
            if rospy.has_param(private_name) and rospy.get_param(private_name) == global_value:
                continue
            rospy.set_param(private_name, global_value)
            mirrored.append(namespace)
        if mirrored:
            rospy.loginfo(
                "[usb_c_insertion] event=align_housing_yaw_params_mirrored_from_global namespaces=%s",
                ",".join(mirrored),
            )

    def _execute(self, goal) -> None:
        started_at = rospy.Time.now()
        if not self._estimate_client.wait_for_server(rospy.Duration.from_sec(max(0.1, self._estimate_wait_timeout))):
            self._abort("estimate_action_unavailable", started_at)
            return
        if not self._move_client.wait_for_server(rospy.Duration.from_sec(max(0.1, self._move_wait_timeout))):
            self._abort("move_action_unavailable", started_at)
            return

        image_topic = str(goal.image_topic).strip() or self._default_image_topic
        cloud_topic = str(goal.cloud_topic).strip() or self._default_cloud_topic
        estimate_timeout = self._goal_or_default(goal.estimate_timeout, self._default_estimate_timeout)
        yaw_tolerance_rad = self._goal_or_default(goal.yaw_tolerance_rad, self._default_yaw_tolerance_rad)
        max_iterations = self._goal_int_or_default(goal.max_iterations, self._default_max_iterations)
        max_yaw_step_rad = self._goal_or_default(goal.max_yaw_step_rad, self._default_max_yaw_step_rad)
        settle_time = self._goal_or_default(goal.settle_time, self._default_settle_time)
        move_timeout = self._goal_or_default(goal.move_timeout, self._default_move_timeout)
        rospy.loginfo(
            "[usb_c_insertion] event=align_housing_yaw_goal_received image_topic=%s cloud_topic=%s estimate_timeout=%.2f yaw_tolerance_rad=%.4f max_iterations=%d max_yaw_step_rad=%.4f settle_time=%.2f move_timeout=%.2f",
            image_topic,
            cloud_topic,
            estimate_timeout,
            yaw_tolerance_rad,
            max_iterations,
            max_yaw_step_rad,
            settle_time,
            move_timeout,
        )

        initial_error: Optional[float] = None
        final_error = 0.0
        total_yaw_command = 0.0
        correction_count = 0
        final_normal = Vector3()
        final_pose: Optional[PoseStamped] = None

        for measurement_index in range(max(0, max_iterations) + 1):
            if self._server.is_preempt_requested():
                self._cancel_children()
                self._server.set_preempted(
                    self._make_result(
                        False,
                        "preempted",
                        "preempted",
                        correction_count,
                        initial_error,
                        final_error,
                        total_yaw_command,
                        final_normal,
                        final_pose,
                    )
                )
                return

            plane_result = self._estimate_plane(image_topic, cloud_topic, estimate_timeout)
            if plane_result is None or not bool(plane_result.success):
                message = plane_result.message if plane_result is not None else "no_plane_result"
                error_code = plane_result.error_code if plane_result is not None else "estimate_failed"
                self._abort(
                    "estimate_failed: %s" % message,
                    started_at,
                    error_code,
                    correction_count,
                    initial_error,
                    final_error,
                    total_yaw_command,
                    final_normal,
                    final_pose,
                )
                return

            current_pose = self._tf.get_tool_pose_in_base()
            if current_pose is None:
                self._abort(
                    "tool_pose_unavailable",
                    started_at,
                    iterations=correction_count,
                    initial_error=initial_error,
                    final_error=final_error,
                    total_yaw_command=total_yaw_command,
                    final_normal=final_normal,
                    final_pose=final_pose,
                )
                return

            yaw_error = self._compute_yaw_error(current_pose, plane_result.plane_normal_base)
            if yaw_error is None:
                self._abort(
                    "yaw_error_unobservable",
                    started_at,
                    iterations=correction_count,
                    initial_error=initial_error,
                    final_error=final_error,
                    total_yaw_command=total_yaw_command,
                    final_normal=plane_result.plane_normal_base,
                    final_pose=current_pose,
                )
                return

            if initial_error is None:
                initial_error = yaw_error
            final_error = yaw_error
            final_normal = plane_result.plane_normal_base
            final_pose = current_pose

            if abs(yaw_error) <= yaw_tolerance_rad:
                self._publish_feedback(
                    "aligned",
                    correction_count,
                    yaw_error,
                    0.0,
                    total_yaw_command,
                    plane_result,
                )
                result = self._make_result(
                    True,
                    "aligned",
                    "",
                    correction_count,
                    initial_error,
                    final_error,
                    total_yaw_command,
                    final_normal,
                    final_pose,
                )
                self._server.set_succeeded(result)
                rospy.loginfo(
                    "[usb_c_insertion] event=align_housing_yaw_complete success=true iterations=%d initial_yaw_error_deg=%.2f final_yaw_error_deg=%.2f total_yaw_command_deg=%.2f",
                    correction_count,
                    math.degrees(initial_error),
                    math.degrees(final_error),
                    math.degrees(total_yaw_command),
                )
                return

            if correction_count >= max_iterations:
                self._publish_feedback(
                    "max_iterations_reached",
                    correction_count,
                    yaw_error,
                    0.0,
                    total_yaw_command,
                    plane_result,
                )
                self._abort(
                    "max_iterations_reached",
                    started_at,
                    iterations=correction_count,
                    initial_error=initial_error,
                    final_error=final_error,
                    total_yaw_command=total_yaw_command,
                    final_normal=final_normal,
                    final_pose=final_pose,
                )
                return

            yaw_command = self._limit_yaw_step(yaw_error, max_yaw_step_rad)
            target_pose = self._build_yaw_target_pose(current_pose, yaw_command)
            self._publish_feedback(
                "move_to_corrected_yaw",
                correction_count + 1,
                yaw_error,
                yaw_command,
                total_yaw_command + yaw_command,
                plane_result,
            )
            move_result = self._move_to_pose(target_pose, settle_time, move_timeout)
            if move_result is None or not bool(move_result.success):
                message = move_result.message if move_result is not None else "no_move_result"
                error_code = move_result.error_code if move_result is not None else "move_failed"
                self._abort(
                    "move_failed: %s" % message,
                    started_at,
                    error_code,
                    correction_count,
                    initial_error,
                    final_error,
                    total_yaw_command,
                    final_normal,
                    final_pose,
                )
                return

            self._log_yaw_step_motion(
                correction_count + 1,
                current_pose,
                target_pose,
                move_result.final_pose,
                plane_result.plane_normal_base,
                yaw_error,
                yaw_command,
            )
            correction_count += 1
            total_yaw_command += yaw_command
            final_pose = move_result.final_pose

    def _estimate_plane(self, image_topic: str, cloud_topic: str, timeout: float):
        goal = EstimateHousingPlaneGoal()
        goal.image_topic = image_topic
        goal.cloud_topic = cloud_topic
        goal.timeout = float(timeout)
        goal.min_blob_area = required_float_param("~housing_plane/min_blob_area")
        goal.roi_radius_px = required_int_param("~housing_plane/roi_radius_px")
        goal.roi_stride_px = required_int_param("~housing_plane/roi_stride_px")
        goal.depth_window_m = required_float_param("~housing_plane/depth_window_m")
        goal.ransac_iterations = required_int_param("~housing_plane/ransac_iterations")
        goal.ransac_distance_threshold = required_float_param("~housing_plane/ransac_distance_threshold")
        goal.min_inliers = required_int_param("~housing_plane/min_inliers")
        goal.use_svd_refit = required_bool_param("~housing_plane/use_svd_refit")
        goal.use_largest_component = required_bool_param("~housing_plane/use_largest_component")
        self._estimate_client.send_goal(goal)
        finished = self._estimate_client.wait_for_result(rospy.Duration.from_sec(max(0.1, float(timeout) + 2.0)))
        if not finished:
            self._estimate_client.cancel_goal()
            return None
        return self._estimate_client.get_result()

    def _move_to_pose(self, target_pose: PoseStamped, settle_time: float, timeout: float):
        move_goal = MoveToPoseGoal()
        move_goal.target_pose = target_pose
        move_goal.settle_time = float(settle_time)
        move_goal.timeout = float(timeout)
        self._move_client.send_goal(move_goal)
        finished = self._move_client.wait_for_result(rospy.Duration.from_sec(max(0.1, float(timeout) + 5.0)))
        if not finished:
            self._move_client.cancel_goal()
            return None
        return self._move_client.get_result()

    def _build_yaw_target_pose(self, current_pose: PoseStamped, yaw_command: float) -> PoseStamped:
        target_pose = PoseStamped()
        target_pose.header.stamp = rospy.Time.now()
        target_pose.header.frame_id = self._base_frame
        target_pose.pose.position = current_pose.pose.position
        current_quaternion = (
            current_pose.pose.orientation.x,
            current_pose.pose.orientation.y,
            current_pose.pose.orientation.z,
            current_pose.pose.orientation.w,
        )
        yaw_quaternion = quaternion_from_yaw(float(yaw_command))
        target_quaternion = normalize_quaternion(quaternion_multiply(yaw_quaternion, current_quaternion))
        target_pose.pose.orientation.x = target_quaternion[0]
        target_pose.pose.orientation.y = target_quaternion[1]
        target_pose.pose.orientation.z = target_quaternion[2]
        target_pose.pose.orientation.w = target_quaternion[3]
        return target_pose

    def _log_yaw_step_motion(
        self,
        iteration: int,
        before_pose: PoseStamped,
        target_pose: PoseStamped,
        after_pose: PoseStamped,
        plane_normal_base: Vector3,
        yaw_error: float,
        yaw_command: float,
    ) -> None:
        before_error = yaw_error
        target_error = self._compute_yaw_error(target_pose, plane_normal_base)
        after_error = self._compute_yaw_error(after_pose, plane_normal_base)
        before_axis_xy = self._tool_axis_xy(before_pose)
        target_axis_xy = self._tool_axis_xy(target_pose)
        after_axis_xy = self._tool_axis_xy(after_pose)
        target_normal_xy = self._target_axis_xy(plane_normal_base)
        commanded_axis_delta = self._signed_xy_angle(before_axis_xy, target_axis_xy)
        actual_axis_delta = self._signed_xy_angle(before_axis_xy, after_axis_xy)
        before_q = self._pose_quaternion(before_pose)
        target_q = self._pose_quaternion(target_pose)
        after_q = self._pose_quaternion(after_pose)
        before_rpy = euler_from_quaternion(before_q)
        target_rpy = euler_from_quaternion(target_q)
        after_rpy = euler_from_quaternion(after_q)

        rospy.loginfo(
            "[usb_c_insertion] event=align_housing_yaw_step_motion iteration=%d "
            "commanded_yaw_deg=%.3f commanded_axis_delta_deg=%.3f actual_axis_delta_deg=%.3f "
            "before_error_deg=%.3f target_error_deg=%.3f after_error_deg=%.3f "
            "target_normal_xy=(%.4f,%.4f) before_tool_xy=(%.4f,%.4f) "
            "target_tool_xy=(%.4f,%.4f) after_tool_xy=(%.4f,%.4f)",
            int(iteration),
            math.degrees(yaw_command),
            self._degrees_or_nan(commanded_axis_delta),
            self._degrees_or_nan(actual_axis_delta),
            math.degrees(before_error),
            self._degrees_or_nan(target_error),
            self._degrees_or_nan(after_error),
            self._component_or_nan(target_normal_xy, 0),
            self._component_or_nan(target_normal_xy, 1),
            self._component_or_nan(before_axis_xy, 0),
            self._component_or_nan(before_axis_xy, 1),
            self._component_or_nan(target_axis_xy, 0),
            self._component_or_nan(target_axis_xy, 1),
            self._component_or_nan(after_axis_xy, 0),
            self._component_or_nan(after_axis_xy, 1),
        )
        rospy.loginfo(
            "[usb_c_insertion] event=align_housing_yaw_step_orientation iteration=%d "
            "before_rpy_deg=(%.3f,%.3f,%.3f) target_rpy_deg=(%.3f,%.3f,%.3f) "
            "after_rpy_deg=(%.3f,%.3f,%.3f) "
            "before_q=(%.5f,%.5f,%.5f,%.5f) target_q=(%.5f,%.5f,%.5f,%.5f) "
            "after_q=(%.5f,%.5f,%.5f,%.5f)",
            int(iteration),
            math.degrees(before_rpy[0]),
            math.degrees(before_rpy[1]),
            math.degrees(before_rpy[2]),
            math.degrees(target_rpy[0]),
            math.degrees(target_rpy[1]),
            math.degrees(target_rpy[2]),
            math.degrees(after_rpy[0]),
            math.degrees(after_rpy[1]),
            math.degrees(after_rpy[2]),
            before_q[0],
            before_q[1],
            before_q[2],
            before_q[3],
            target_q[0],
            target_q[1],
            target_q[2],
            target_q[3],
            after_q[0],
            after_q[1],
            after_q[2],
            after_q[3],
        )

    def _compute_yaw_error(self, current_pose: PoseStamped, plane_normal_base: Vector3) -> Optional[float]:
        current_quaternion = (
            current_pose.pose.orientation.x,
            current_pose.pose.orientation.y,
            current_pose.pose.orientation.z,
            current_pose.pose.orientation.w,
        )
        current_axis = rotate_vector_by_quaternion(
            self._tool_axis[0],
            self._tool_axis[1],
            self._tool_axis[2],
            current_quaternion[0],
            current_quaternion[1],
            current_quaternion[2],
            current_quaternion[3],
        )
        target_axis = (
            self._target_axis_from_plane_normal_sign * float(plane_normal_base.x),
            self._target_axis_from_plane_normal_sign * float(plane_normal_base.y),
            self._target_axis_from_plane_normal_sign * float(plane_normal_base.z),
        )
        current_xy = self._normalize_xy((current_axis[0], current_axis[1]))
        target_xy = self._normalize_xy((target_axis[0], target_axis[1]))
        if current_xy is None or target_xy is None:
            return None
        cross_z = current_xy[0] * target_xy[1] - current_xy[1] * target_xy[0]
        dot = current_xy[0] * target_xy[0] + current_xy[1] * target_xy[1]
        return math.atan2(cross_z, dot)

    def _tool_axis_xy(self, pose: PoseStamped) -> Optional[Tuple[float, float]]:
        qx, qy, qz, qw = self._pose_quaternion(pose)
        axis = rotate_vector_by_quaternion(
            self._tool_axis[0],
            self._tool_axis[1],
            self._tool_axis[2],
            qx,
            qy,
            qz,
            qw,
        )
        return self._normalize_xy((axis[0], axis[1]))

    def _target_axis_xy(self, plane_normal_base: Vector3) -> Optional[Tuple[float, float]]:
        return self._normalize_xy(
            (
                self._target_axis_from_plane_normal_sign * float(plane_normal_base.x),
                self._target_axis_from_plane_normal_sign * float(plane_normal_base.y),
            )
        )

    def _publish_feedback(
        self,
        stage: str,
        iteration: int,
        yaw_error: float,
        yaw_command: float,
        total_yaw_command: float,
        plane_result,
    ) -> None:
        feedback = AlignHousingYawFeedback()
        feedback.stage = str(stage)
        feedback.iteration = int(iteration)
        feedback.yaw_error_rad = float(yaw_error)
        feedback.yaw_command_rad = float(yaw_command)
        feedback.total_yaw_command_rad = float(total_yaw_command)
        feedback.normal_base = plane_result.plane_normal_base
        feedback.inlier_ratio = float(plane_result.inlier_ratio)
        feedback.rms_error = float(plane_result.rms_error)
        self._server.publish_feedback(feedback)
        rospy.loginfo(
            "[usb_c_insertion] event=align_housing_yaw_feedback stage=%s iteration=%d yaw_error_deg=%.2f yaw_command_deg=%.2f total_yaw_command_deg=%.2f normal_base=(%.4f,%.4f,%.4f) ratio=%.3f rms=%.4f",
            stage,
            int(iteration),
            math.degrees(yaw_error),
            math.degrees(yaw_command),
            math.degrees(total_yaw_command),
            plane_result.plane_normal_base.x,
            plane_result.plane_normal_base.y,
            plane_result.plane_normal_base.z,
            plane_result.inlier_ratio,
            plane_result.rms_error,
        )

    def _abort(
        self,
        message: str,
        started_at: rospy.Time,
        error_code: Optional[str] = None,
        iterations: int = 0,
        initial_error: Optional[float] = None,
        final_error: float = 0.0,
        total_yaw_command: float = 0.0,
        final_normal: Optional[Vector3] = None,
        final_pose: Optional[PoseStamped] = None,
    ) -> None:
        result = self._make_result(
            False,
            message,
            error_code or message,
            iterations,
            initial_error,
            final_error,
            total_yaw_command,
            final_normal,
            final_pose,
        )
        self._server.set_aborted(result)
        rospy.logwarn(
            "[usb_c_insertion] event=align_housing_yaw_complete success=false message=%s error_code=%s iterations=%d elapsed=%.2f",
            message,
            error_code or message,
            int(iterations),
            (rospy.Time.now() - started_at).to_sec(),
        )

    @staticmethod
    def _make_result(
        success: bool,
        message: str,
        error_code: str,
        iterations: int,
        initial_error: Optional[float],
        final_error: float,
        total_yaw_command: float,
        final_normal: Optional[Vector3],
        final_pose: Optional[PoseStamped],
    ) -> AlignHousingYawResult:
        result = AlignHousingYawResult()
        result.success = bool(success)
        result.message = str(message)
        result.error_code = str(error_code)
        result.iterations = int(iterations)
        result.initial_yaw_error_rad = float(initial_error) if initial_error is not None else 0.0
        result.final_yaw_error_rad = float(final_error)
        result.total_yaw_command_rad = float(total_yaw_command)
        if final_normal is not None:
            result.final_normal_base = final_normal
        if final_pose is not None:
            result.final_pose = final_pose
        return result

    def _cancel_children(self) -> None:
        self._estimate_client.cancel_goal()
        self._move_client.cancel_goal()

    @staticmethod
    def _limit_yaw_step(yaw_error: float, max_yaw_step_rad: float) -> float:
        limit = abs(float(max_yaw_step_rad))
        if limit <= 0.0:
            return float(yaw_error)
        return max(-limit, min(limit, float(yaw_error)))

    @staticmethod
    def _normalize_xy(vector_xy: Tuple[float, float]) -> Optional[Tuple[float, float]]:
        norm = math.sqrt(vector_xy[0] * vector_xy[0] + vector_xy[1] * vector_xy[1])
        if norm <= 1e-6:
            return None
        return vector_xy[0] / norm, vector_xy[1] / norm

    @staticmethod
    def _signed_xy_angle(
        from_xy: Optional[Tuple[float, float]],
        to_xy: Optional[Tuple[float, float]],
    ) -> Optional[float]:
        if from_xy is None or to_xy is None:
            return None
        cross_z = from_xy[0] * to_xy[1] - from_xy[1] * to_xy[0]
        dot = from_xy[0] * to_xy[0] + from_xy[1] * to_xy[1]
        return math.atan2(cross_z, dot)

    @staticmethod
    def _pose_quaternion(pose: PoseStamped) -> Tuple[float, float, float, float]:
        return (
            pose.pose.orientation.x,
            pose.pose.orientation.y,
            pose.pose.orientation.z,
            pose.pose.orientation.w,
        )

    @staticmethod
    def _degrees_or_nan(value: Optional[float]) -> float:
        return math.degrees(value) if value is not None else float("nan")

    @staticmethod
    def _component_or_nan(vector, index: int) -> float:
        return float(vector[index]) if vector is not None else float("nan")

    @staticmethod
    def _goal_or_default(value: float, default: float) -> float:
        return float(value) if float(value) > 0.0 else float(default)

    @staticmethod
    def _goal_int_or_default(value: int, default: int) -> int:
        return int(value) if int(value) > 0 else int(default)

    @staticmethod
    def _read_vector_param(param_name: str):
        value = required_vector_param(param_name)
        if not isinstance(value, (list, tuple)) or len(value) != 3:
            rospy.logwarn("[usb_c_insertion] event=align_housing_yaw_invalid_vector_param param=%s", param_name)
            raise ValueError("Invalid vector ROS parameter: %s" % rospy.resolve_name(param_name))
        vector = tuple(float(component) for component in value)
        norm = math.sqrt(sum(component * component for component in vector))
        if norm <= 1e-9:
            rospy.logwarn("[usb_c_insertion] event=align_housing_yaw_zero_vector_param param=%s", param_name)
            raise ValueError("Zero-length vector ROS parameter: %s" % rospy.resolve_name(param_name))
        return tuple(component / norm for component in vector)


def main() -> None:
    rospy.init_node("usb_c_insertion_align_housing_yaw_action_server")
    AlignHousingYawActionServer()
    rospy.spin()


if __name__ == "__main__":
    main()
