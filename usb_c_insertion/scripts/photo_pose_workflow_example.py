#!/usr/bin/env python3

from __future__ import annotations

import math
import os
import sys

import actionlib
from geometry_msgs.msg import PoseStamped
import rospy

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from usb_c_insertion.msg import (
    ApplyYawCorrectionAction,
    ApplyYawCorrectionGoal,
    MoveToPoseAction,
    MoveToPoseGoal,
    ProbeSurfaceAction,
    ProbeSurfaceGoal,
    SearchPortAction,
    SearchPortGoal,
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
        self._base_frame = str(rospy.get_param("~frames/base_frame", "base_link"))

        self._settle_time = float(rospy.get_param("~photo_pose/settle_time", 0.4))
        self._timeout = float(rospy.get_param("~photo_pose/timeout", 20.0))
        self._require_fresh_vision = bool(rospy.get_param("~photo_pose/require_fresh_vision_result", True))

        self._goal = self._load_goal_pose()
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

        port_pose = self._run_vision()
        if port_pose is None:
            return False

        pre_pose = self._compute_prepose(port_pose)
        if pre_pose is None:
            return False

        if not self._move_to_named_pose(pre_pose, "prepose"):
            return False

        yaw_correction = self._probe_surface(port_pose)
        if yaw_correction is None:
            return False

        corrected_pose = self._apply_yaw_correction(pre_pose, yaw_correction)
        if corrected_pose is None:
            return False

        return self._search_port(port_pose, corrected_pose)

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

    def _run_vision(self) -> PoseStamped | None:
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
            "[usb_c_insertion] event=vision_result_received path=%s x=%.4f y=%.4f z=%.4f",
            response.json_path,
            response.port_pose.pose.position.x,
            response.port_pose.pose.position.y,
            response.port_pose.pose.position.z,
        )
        return response.port_pose

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

    def _search_port(self, port_pose: PoseStamped, corrected_pose: PoseStamped) -> bool:
        if not self._search_port_client.wait_for_server(rospy.Duration.from_sec(5.0)):
            rospy.logerr("[usb_c_insertion] event=photo_pose_workflow_failed reason=search_port_action_unavailable")
            return False

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
            return False

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
            return False

        rospy.loginfo(
            "[usb_c_insertion] event=search_port_complete port_found=%s inserted_depth=%.4f contact_force=%.3f completed_steps=%d",
            str(bool(result.port_found)).lower(),
            float(result.inserted_depth),
            float(result.contact_force),
            int(result.completed_steps),
        )
        return bool(result.port_found)

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
