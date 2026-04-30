#!/usr/bin/env python3

from __future__ import annotations

import json
import os
import sys
from typing import Dict, Optional

from geometry_msgs.msg import PoseStamped
import rospy
from std_msgs.msg import String

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from extraction_controller import ExtractionController
from ft_interface import FTInterface
from insertion_workflow import InsertionWorkflow
from param_utils import required_bool_param, required_float_param, required_int_param, required_str_param
from presentation_snapshot_recorder import PresentationSnapshotRecorder
from preinsert_workflow_helpers import PreinsertWorkflowHelpers
from robot_interface import RobotInterface
from tf_interface import TFInterface


class WorkflowStatusPublisher:
    STAGES = (
        ("dependencies", "Dependencies"),
        ("move_overview", "Move To Overview"),
        ("overview_vision", "Overview Vision"),
        ("plan_camera_pose", "Plan Camera Pose"),
        ("move_camera_pose", "Move Camera Pose"),
        ("align_housing_yaw", "Align Housing Yaw"),
        ("center_port", "Center Port"),
        ("orientation_plane", "Orientation Plane"),
        ("verify_looming", "Verify Looming"),
        ("final_plane", "Final Plane"),
        ("updated_port_pose", "Updated Port Pose"),
        ("tcp_precontact", "TCP Precontact"),
        ("insertion", "Insertion"),
        ("wait_after_insertion", "Wait"),
        ("extract", "Extract"),
        ("return_to_start", "Return To Start"),
    )

    def __init__(self, topic: str):
        self._publisher = rospy.Publisher(topic, String, queue_size=20, latch=True)
        self._sequence = 0
        self._labels = dict(self.STAGES)

    def publish_all_pending(self) -> None:
        for stage_id, label in self.STAGES:
            self.publish(stage_id, "pending", label=label)

    def publish(
        self,
        stage_id: str,
        status: str,
        label: Optional[str] = None,
        success: Optional[bool] = None,
        message: str = "",
        values: Optional[Dict] = None,
    ) -> None:
        self._sequence += 1
        payload = {
            "seq": self._sequence,
            "stamp": rospy.Time.now().to_sec(),
            "stage_id": stage_id,
            "label": label or self._labels.get(stage_id, stage_id),
            "status": status,
            "success": success,
            "message": message,
            "values": values or {},
        }
        self._publisher.publish(String(data=json.dumps(payload, sort_keys=True)))


class CombinedInsertionWorkflow:
    """
    Run the complete physical cycle and publish step-level status updates.

    The sequence is:
    pre-insertion alignment, insertion/search, dwell, extraction, return to
    overview pose.
    """

    def __init__(self):
        self._mirror_global_config_to_private_namespace()

        self._status_topic = required_str_param("~combined_workflow/status_topic")
        self._startup_delay = required_float_param("~combined_workflow/startup_delay")
        self._settle_before_insertion = required_float_param("~combined_workflow/settle_before_insertion")
        self._wait_after_insertion = required_float_param("~combined_workflow/wait_after_insertion")
        self._return_to_start = required_bool_param("~combined_workflow/return_to_start")
        self._return_to_start_timeout = required_float_param("~combined_workflow/return_to_start_timeout")
        self._zero_ft_before_extract = required_bool_param("~combined_workflow/zero_ft_before_extract")
        self._extract_zero_ft_settle_time = required_float_param("~combined_workflow/extract_zero_ft_settle_time")
        self._pipeline_wait_timeout = required_float_param("~combined_workflow/pipeline_wait_timeout")

        self._status = WorkflowStatusPublisher(self._status_topic)
        self._helpers: Optional[PreinsertWorkflowHelpers] = None
        self._snapshots: Optional[PresentationSnapshotRecorder] = None
        self._tf = TFInterface()
        self._overview_pose: Optional[PoseStamped] = None
        self._updated_port_pose_publisher = None

    def _mirror_global_config_to_private_namespace(self) -> None:
        namespaces = (
            "frames",
            "topics",
            "motion",
            "micro_motion",
            "contact",
            "probe",
            "search",
            "center_port",
            "looming",
            "housing_plane",
            "align_housing_yaw",
            "insert",
            "verify",
            "extract",
            "gripper",
            "state_machine",
            "workflow",
            "photo_pose",
            "presentation_snapshots",
            "insertion_workflow",
            "combined_workflow",
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
                "[usb_c_insertion] event=combined_workflow_params_mirrored_from_global namespaces=%s",
                ",".join(mirrored),
            )

    def run(self) -> bool:
        rospy.loginfo("[usb_c_insertion] event=combined_workflow_start")
        if self._startup_delay > 0.0:
            rospy.sleep(self._startup_delay)
        self._status.publish_all_pending()

        try:
            if not self._run_preinsert_alignment():
                return False
            if not self._run_insertion():
                return False
            if not self._run_wait_after_insertion():
                return False
            if not self._run_extraction():
                return False
            if self._return_to_start and not self._run_return_to_start():
                return False
        except Exception as exc:
            rospy.logerr("[usb_c_insertion] event=combined_workflow_exception error=%s", exc)
            self._status.publish(
                "return_to_start",
                "failed",
                success=False,
                message="workflow_exception: %s" % exc,
            )
            return False

        rospy.loginfo("[usb_c_insertion] event=combined_workflow_complete success=true")
        return True

    def _run_preinsert_alignment(self) -> bool:
        self._helpers = PreinsertWorkflowHelpers()
        self._snapshots = PresentationSnapshotRecorder()
        updated_topic = self._helpers._required_str_param("~workflow/updated_port_pose_topic")
        self._updated_port_pose_publisher = rospy.Publisher(updated_topic, PoseStamped, queue_size=1, latch=True)

        self._status.publish("dependencies", "running")
        if not self._helpers.wait_for_dependencies():
            return self._fail("dependencies", "dependency_unavailable")
        self._status.publish("dependencies", "success", success=True)

        self._status.publish("move_overview", "running")
        self._overview_pose = self._helpers.load_overview_pose()
        overview_final = self._helpers.move_to_pose(self._overview_pose, "overview_pose")
        if overview_final is None:
            return self._fail("move_overview", "move_failed")
        self._status.publish(
            "move_overview",
            "success",
            success=True,
            values={"target": self._pose_values(self._overview_pose), "final": self._pose_values(overview_final)},
        )

        self._status.publish("overview_vision", "running")
        coarse_port_pose = self._helpers.run_overview_vision()
        if coarse_port_pose is None:
            return self._fail("overview_vision", "vision_failed")
        self._snapshots.capture_port_pose_axes(
            "01_overview_initial_port_estimate.png",
            "01 Overview position",
            coarse_port_pose,
        )
        self._status.publish(
            "overview_vision",
            "success",
            success=True,
            values={"port_pose": self._pose_values(coarse_port_pose)},
        )

        self._status.publish("plan_camera_pose", "running")
        camera_pose = self._helpers.plan_camera_pose_from_port(coarse_port_pose)
        if camera_pose is None:
            return self._fail("plan_camera_pose", "plan_failed")
        self._status.publish(
            "plan_camera_pose",
            "success",
            success=True,
            values={"camera_pose": self._pose_values(camera_pose)},
        )

        self._status.publish("move_camera_pose", "running")
        camera_final = self._helpers.move_to_pose(camera_pose, "camera_to_coarse_port")
        if camera_final is None:
            return self._fail("move_camera_pose", "move_failed")
        self._snapshots.capture_current_view(
            "02_camera_at_initial_estimate.png",
            "02 Camera at initial estimate",
        )
        self._status.publish(
            "move_camera_pose",
            "success",
            success=True,
            values={"final": self._pose_values(camera_final)},
        )

        self._status.publish("align_housing_yaw", "running")
        if not self._helpers.align_housing_yaw():
            return self._fail("align_housing_yaw", "align_failed")
        self._snapshots.capture_marker_alignment(
            "03_after_yaw_alignment_before_centering.png",
            "03 After yaw alignment",
        )
        align_result = self._helpers._align_client.get_result()
        self._status.publish(
            "align_housing_yaw",
            "success",
            success=True,
            values=self._align_values(align_result),
        )

        self._status.publish("center_port", "running")
        centered_pose = self._helpers.center_port_in_image()
        if centered_pose is None:
            return self._fail("center_port", "center_failed")
        center_result = self._helpers._center_client.get_result()
        self._snapshots.capture_marker_alignment(
            "04_centered_over_port.png",
            "04 Camera over circle center",
            fallback_marker_center=PresentationSnapshotRecorder.center_from_center_result(center_result),
        )
        self._status.publish(
            "center_port",
            "success",
            success=True,
            values=self._center_values(center_result, centered_pose),
        )

        self._status.publish("orientation_plane", "running")
        orientation_check = self._helpers.estimate_housing_plane("orientation_check")
        if orientation_check is None:
            return self._fail("orientation_plane", "estimate_failed")
        if not self._helpers.validate_plane_quality(orientation_check, "orientation_check"):
            return self._fail("orientation_plane", "quality_failed", self._plane_values(orientation_check))
        self._status.publish(
            "orientation_plane",
            "success",
            success=True,
            values=self._plane_values(orientation_check),
        )

        self._status.publish("verify_looming", "running")
        looming_result = self._helpers.verify_looming()
        if looming_result is None:
            return self._fail("verify_looming", "looming_failed")
        recenter_ok, recenter_result = self._helpers.recenter_after_looming_if_needed(looming_result)
        if not recenter_ok:
            return self._fail("verify_looming", "looming_recenter_failed")
        marker_center = PresentationSnapshotRecorder.center_from_center_result(recenter_result)
        if marker_center is None:
            marker_center = PresentationSnapshotRecorder.center_from_looming_result(looming_result)
        self._snapshots.capture_marker_alignment(
            "05_after_verify_looming.png",
            "05 After looming verification",
            fallback_marker_center=marker_center,
        )
        self._status.publish(
            "verify_looming",
            "success",
            success=True,
            values=self._looming_values(looming_result),
        )

        self._status.publish("final_plane", "running")
        final_plane = self._helpers.estimate_housing_plane("final_depth_update")
        if final_plane is None:
            return self._fail("final_plane", "estimate_failed")
        if not self._helpers.validate_plane_quality(final_plane, "final_depth_update"):
            return self._fail("final_plane", "quality_failed", self._plane_values(final_plane))
        self._status.publish("final_plane", "success", success=True, values=self._plane_values(final_plane))

        self._status.publish("updated_port_pose", "running")
        updated_port_pose = self._helpers.build_updated_port_pose(final_plane)
        if updated_port_pose is None:
            return self._fail("updated_port_pose", "build_failed")
        self._updated_port_pose_publisher.publish(updated_port_pose)
        self._status.publish(
            "updated_port_pose",
            "success",
            success=True,
            values={"topic": updated_topic, "port_pose": self._pose_values(updated_port_pose)},
        )

        self._status.publish("tcp_precontact", "running")
        tcp_precontact_pose = self._helpers.plan_tcp_precontact_pose(updated_port_pose)
        if tcp_precontact_pose is None:
            return self._fail("tcp_precontact", "plan_failed")
        precontact_final = self._helpers.move_to_pose(tcp_precontact_pose, "tcp_precontact_wait")
        if precontact_final is None:
            return self._fail("tcp_precontact", "move_failed")
        self._status.publish(
            "tcp_precontact",
            "success",
            success=True,
            values={"target": self._pose_values(tcp_precontact_pose), "final": self._pose_values(precontact_final)},
        )
        return True

    def _run_insertion(self) -> bool:
        settle_time = max(0.0, self._settle_before_insertion)
        self._status.publish(
            "insertion",
            "running",
            message="settling_before_insertion",
            values={"settle_s": round(settle_time, 2)},
        )
        RobotInterface().stop_motion()
        if settle_time > 0.0:
            rospy.loginfo(
                "[usb_c_insertion] event=combined_workflow_settle_before_insertion duration=%.3f",
                settle_time,
            )
            rospy.sleep(settle_time)

        self._status.publish("insertion", "running", message="starting_insertion_workflow")
        insertion = InsertionWorkflow()
        success = insertion.run()
        values = {"final_tool_pose": self._current_tool_pose_values()}
        if not success:
            return self._fail("insertion", "insertion_failed_or_unverified", values)
        self._status.publish("insertion", "success", success=True, values=values)
        return True

    def _run_wait_after_insertion(self) -> bool:
        self._status.publish(
            "wait_after_insertion",
            "running",
            values={"remaining_s": round(max(0.0, self._wait_after_insertion), 2)},
        )
        deadline = rospy.Time.now() + rospy.Duration.from_sec(max(0.0, self._wait_after_insertion))
        rate = rospy.Rate(2.0)
        while not rospy.is_shutdown() and rospy.Time.now() < deadline:
            remaining = max(0.0, (deadline - rospy.Time.now()).to_sec())
            self._status.publish(
                "wait_after_insertion",
                "running",
                values={"remaining_s": round(remaining, 2)},
            )
            rate.sleep()
        if rospy.is_shutdown():
            return self._fail("wait_after_insertion", "shutdown")
        self._status.publish("wait_after_insertion", "success", success=True)
        return True

    def _run_extraction(self) -> bool:
        self._status.publish("extract", "running")
        robot = RobotInterface()
        tf_interface = TFInterface()
        ft_interface = FTInterface(
            wrench_topic=required_str_param("~topics/wrench"),
            filter_window_size=required_int_param("~contact/baseline_window"),
            wrench_timeout=required_float_param("~contact/wrench_timeout"),
            zero_service_name=required_str_param("~topics/zero_ft_service"),
        )
        if not robot.wait_for_motion_pipeline(timeout=self._pipeline_wait_timeout, require_pose_servo=False):
            return self._fail("extract", "motion_pipeline_unavailable")
        if self._zero_ft_before_extract:
            robot.stop_motion()
            if not ft_interface.zero_sensor():
                return self._fail("extract", "zero_ft_failed")
            rospy.sleep(max(0.0, self._extract_zero_ft_settle_time))

        result = ExtractionController(robot, tf_interface, ft_interface).extract()
        values = {
            "reason": result.reason,
            "extracted_distance": round(result.extracted_distance, 5),
            "pull_force": round(result.pull_force, 3),
            "lateral_force": round(result.lateral_force, 3),
            "torque_norm": round(result.torque_norm, 3),
            "gripper_opened": bool(result.gripper_opened),
        }
        if not result.success:
            return self._fail("extract", result.reason, values)
        self._status.publish("extract", "success", success=True, values=values)
        return True

    def _run_return_to_start(self) -> bool:
        self._status.publish("return_to_start", "running")
        if self._helpers is None or self._overview_pose is None:
            return self._fail("return_to_start", "missing_overview_pose")
        final_pose = self._helpers.move_to_pose(
            self._overview_pose,
            "return_to_overview_pose",
            timeout=self._return_to_start_timeout,
        )
        if final_pose is None:
            return self._fail("return_to_start", "move_failed")
        self._status.publish(
            "return_to_start",
            "success",
            success=True,
            values={"target": self._pose_values(self._overview_pose), "final": self._pose_values(final_pose)},
        )
        return True

    def _fail(self, stage_id: str, message: str, values: Optional[Dict] = None) -> bool:
        self._status.publish(stage_id, "failed", success=False, message=message, values=values or {})
        rospy.logerr("[usb_c_insertion] event=combined_workflow_failed stage=%s reason=%s", stage_id, message)
        return False

    @staticmethod
    def _pose_values(pose: Optional[PoseStamped]) -> Dict:
        if pose is None:
            return {}
        return {
            "frame": pose.header.frame_id,
            "x": round(pose.pose.position.x, 5),
            "y": round(pose.pose.position.y, 5),
            "z": round(pose.pose.position.z, 5),
            "qx": round(pose.pose.orientation.x, 5),
            "qy": round(pose.pose.orientation.y, 5),
            "qz": round(pose.pose.orientation.z, 5),
            "qw": round(pose.pose.orientation.w, 5),
        }

    def _current_tool_pose_values(self) -> Dict:
        return self._pose_values(self._tf.get_tool_pose_in_base())

    @staticmethod
    def _align_values(result) -> Dict:
        if result is None:
            return {}
        return {
            "iterations": int(result.iterations),
            "final_yaw_error_deg": round(float(result.final_yaw_error_rad) * 57.295779513, 4),
            "total_yaw_command_deg": round(float(result.total_yaw_command_rad) * 57.295779513, 4),
            "error_code": result.error_code,
            "message": result.message,
        }

    def _center_values(self, result, final_pose: PoseStamped) -> Dict:
        values = {"final_pose": self._pose_values(final_pose)}
        if result is not None:
            values.update(
                {
                    "error_norm_px": round(float(result.error_norm), 3),
                    "blob_center_x": round(float(result.blob_center_x), 2),
                    "blob_center_y": round(float(result.blob_center_y), 2),
                    "message": result.message,
                }
            )
        return values

    @staticmethod
    def _plane_values(result) -> Dict:
        if result is None:
            return {}
        return {
            "inliers": int(result.inlier_count),
            "ratio": round(float(result.inlier_ratio), 4),
            "rms": round(float(result.rms_error), 5),
            "marker_x": round(result.marker_plane_point_base.point.x, 5),
            "marker_y": round(result.marker_plane_point_base.point.y, 5),
            "marker_z": round(result.marker_plane_point_base.point.z, 5),
            "normal_x": round(result.plane_normal_base.x, 5),
            "normal_y": round(result.plane_normal_base.y, 5),
            "normal_z": round(result.plane_normal_base.z, 5),
        }

    @staticmethod
    def _looming_values(result) -> Dict:
        if result is None:
            return {}
        return {
            "scale_ratio": round(float(result.scale_ratio), 4),
            "center_shift_px": round(float(result.center_shift_px), 3),
            "traveled_distance": round(float(result.traveled_distance), 5),
            "message": result.message,
        }


def main() -> None:
    rospy.init_node("usb_c_insertion_combined_workflow")
    success = CombinedInsertionWorkflow().run()
    if not success:
        sys.exit(1)


if __name__ == "__main__":
    main()
