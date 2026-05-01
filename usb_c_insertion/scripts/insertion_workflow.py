#!/usr/bin/env python3

from __future__ import annotations

from dataclasses import dataclass
import math
import os
import sys
from typing import Callable, Dict, Optional, Tuple

from geometry_msgs.msg import PointStamped, PoseStamped, Twist
import rospy
from std_msgs.msg import Bool

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from contact_detector import ContactDetector
from ft_interface import FTInterface
from insertion_controller import InsertionController
from param_utils import required_bool_param, required_float_param, required_int_param, required_str_param
from post_insertion_verifier import PostInsertionVerifier
from prepose_planner import rotate_vector_by_quaternion
from robot_interface import RobotInterface
from search_pattern import generate_preferred_square_spiral_pattern
from tf_interface import TFInterface


@dataclass(frozen=True)
class ContactApproachResult:
    success: bool
    reason: str
    contact_pose: Optional[PoseStamped]
    contact_point: Optional[PointStamped]
    contact_force: float
    travel_distance: float


@dataclass(frozen=True)
class PullRetentionResult:
    success: bool
    reason: str
    pull_force: float
    retraction_distance: float


@dataclass(frozen=True)
class InitialApproachCheckResult:
    accepted: bool
    terminal_failure: bool
    reason: str
    method: str
    pull_result: Optional[PullRetentionResult]
    counterforce_y: float
    counterforce_z: float


@dataclass(frozen=True)
class SearchProbeResult:
    success: bool
    reason: str
    candidate_found: bool
    final_pose: Optional[PoseStamped]
    inserted_depth: float
    contact_force: float
    max_contact_force: float
    travel_distance: float


@dataclass(frozen=True)
class SpiralSearchResult:
    success: bool
    reason: str
    method: str
    completed_steps: int
    total_steps: int
    inserted_depth: float
    contact_force: float


class InsertionWorkflow:
    """
    Start the insertion phase from the precontact pose.

    The workflow first touches the housing/port surface along tool z, then
    checks whether the connector is already seated. The first retention check
    gently pulls on the connector; if that is inconclusive, the workflow falls
    back to the existing lateral/vertical counterforce verification.
    """

    def __init__(self, status_callback: Optional[Callable[[str, str, Optional[bool], str, Dict], None]] = None):
        self._status_callback = status_callback
        self._base_frame = required_str_param("~frames/base_frame")
        self._zero_ft_before_contact = required_bool_param("~insertion_workflow/zero_ft_before_contact")
        self._zero_ft_settle_time = required_float_param("~insertion_workflow/zero_ft_settle_time")
        self._pipeline_wait_timeout = required_float_param("~insertion_workflow/pipeline_wait_timeout")

        self._approach_tool_z_sign = self._sign(required_float_param("~insertion_workflow/approach_tool_z_sign"))
        self._approach_speed = required_float_param("~insertion_workflow/approach_speed")
        self._approach_timeout = required_float_param("~insertion_workflow/approach_timeout")
        self._approach_max_travel = required_float_param("~insertion_workflow/approach_max_travel")
        self._approach_contact_force = required_float_param("~insertion_workflow/approach_contact_force")
        self._contact_settle_time = required_float_param("~insertion_workflow/contact_settle_time")

        self._pull_tool_z_sign = self._sign(required_float_param("~insertion_workflow/pull_tool_z_sign"))
        self._pull_force_mode = required_str_param("~insertion_workflow/pull_force_mode").lower()
        self._zero_ft_before_pull = required_bool_param("~insertion_workflow/zero_ft_before_pull")
        self._pull_force_target = required_float_param("~insertion_workflow/pull_force_target")
        self._pull_force_tolerance = required_float_param("~insertion_workflow/pull_force_tolerance")
        self._pull_force_gain = required_float_param("~insertion_workflow/pull_force_gain")
        self._pull_speed_limit = required_float_param("~insertion_workflow/pull_speed_limit")
        self._pull_timeout = required_float_param("~insertion_workflow/pull_timeout")
        self._pull_hold_time = required_float_param("~insertion_workflow/pull_hold_time")
        self._pull_max_retraction = required_float_param("~insertion_workflow/pull_max_retraction")
        self._return_to_contact_after_failed_pull = required_bool_param(
            "~insertion_workflow/return_to_contact_after_failed_pull"
        )
        self._return_timeout = required_float_param("~insertion_workflow/return_timeout")

        self._lateral_verify_after_failed_pull = required_bool_param(
            "~insertion_workflow/lateral_verify_after_failed_pull"
        )
        self._zero_ft_before_lateral_verify = required_bool_param(
            "~insertion_workflow/zero_ft_before_lateral_verify"
        )
        self._lateral_verify_timeout = required_float_param("~insertion_workflow/lateral_verify_timeout")
        self._verification_zero_ft_settle_time = required_float_param(
            "~insertion_workflow/verification_zero_ft_settle_time"
        )

        self._search_enabled = required_bool_param("~insertion_workflow/search_enabled")
        self._search_timeout = required_float_param("~insertion_workflow/search_timeout")
        self._search_step_tool_x = required_float_param("~insertion_workflow/search_step_tool_x")
        self._search_step_base_z = required_float_param("~insertion_workflow/search_step_base_z")
        self._search_width_tool_x = required_float_param("~insertion_workflow/search_width_tool_x")
        self._search_height_base_z = required_float_param("~insertion_workflow/search_height_base_z")
        self._search_preferred_tool_x_sign = required_float_param(
            "~insertion_workflow/search_preferred_tool_x_sign"
        )
        self._search_preferred_base_z_sign = required_float_param(
            "~insertion_workflow/search_preferred_base_z_sign"
        )
        self._search_pre_probe_clearance = max(
            0.0,
            required_float_param("~insertion_workflow/search_pre_probe_clearance"),
        )
        self._search_pre_probe_settle_time = max(
            0.0,
            required_float_param("~insertion_workflow/search_pre_probe_settle_time"),
        )
        self._search_transfer_timeout = required_float_param("~insertion_workflow/search_transfer_timeout")
        self._search_probe_speed_limit = required_float_param("~insertion_workflow/search_probe_speed_limit")
        self._search_probe_min_speed = required_float_param("~insertion_workflow/search_probe_min_speed")
        self._search_probe_force_gain = required_float_param("~insertion_workflow/search_probe_force_gain")
        self._search_probe_timeout = required_float_param("~insertion_workflow/search_probe_timeout")
        self._search_probe_max_travel = required_float_param("~insertion_workflow/search_probe_max_travel")
        self._search_contact_force_target = required_float_param(
            "~insertion_workflow/search_contact_force_target"
        )
        self._search_center_first_probe = required_bool_param(
            "~insertion_workflow/search_center_first_probe"
        )
        self._search_center_first_contact_force_target = required_float_param(
            "~insertion_workflow/search_center_first_contact_force_target"
        )
        self._search_center_first_probe_timeout = required_float_param(
            "~insertion_workflow/search_center_first_probe_timeout"
        )
        self._search_center_first_probe_max_travel = required_float_param(
            "~insertion_workflow/search_center_first_probe_max_travel"
        )
        self._search_center_first_min_probe_time = max(
            0.0,
            required_float_param("~insertion_workflow/search_center_first_min_probe_time"),
        )
        self._search_contact_force_abort_threshold = required_float_param(
            "~insertion_workflow/search_contact_force_abort_threshold"
        )
        self._search_socket_depth_threshold = required_float_param(
            "~insertion_workflow/search_socket_depth_threshold"
        )
        self._search_candidate_pull_verify = required_bool_param(
            "~insertion_workflow/search_candidate_pull_verify"
        )
        self._search_candidate_lateral_verify = required_bool_param(
            "~insertion_workflow/search_candidate_lateral_verify"
        )
        self._search_insert_after_candidate = required_bool_param(
            "~insertion_workflow/search_insert_after_candidate"
        )
        self._search_continue_after_unverified_candidate = required_bool_param(
            "~insertion_workflow/search_continue_after_unverified_candidate"
        )
        self._search_direct_stop_repeat_count = required_int_param(
            "~insertion_workflow/search_direct_stop_repeat_count"
        )
        self._search_direct_stop_interval = max(
            0.0,
            required_float_param("~insertion_workflow/search_direct_stop_interval"),
        )

        self._command_rate = required_float_param("~motion/command_rate")
        self._position_tolerance = required_float_param("~motion/pose_servo_position_tolerance")
        self._orientation_tolerance = required_float_param("~motion/pose_servo_orientation_tolerance")

        self._robot = RobotInterface()
        self._tf = TFInterface()
        self._ft = FTInterface(
            wrench_topic=required_str_param("~topics/wrench"),
            filter_window_size=required_int_param("~contact/baseline_window"),
            wrench_timeout=required_float_param("~contact/wrench_timeout"),
            zero_service_name=required_str_param("~topics/zero_ft_service"),
        )
        self._contact_detector = ContactDetector(
            self._ft,
            hysteresis=required_float_param("~contact/hysteresis"),
        )
        self._insertion_controller = InsertionController(self._robot, self._tf, self._ft)
        self._post_insertion_verifier = PostInsertionVerifier(self._robot, self._tf, self._ft)
        self._direct_probe_control_active = False
        self._direct_twist_publisher = rospy.Publisher(
            required_str_param("~topics/twist_cmd"),
            Twist,
            queue_size=10,
        )
        self._micro_motion_active_publisher = rospy.Publisher(
            required_str_param("~topics/micro_motion_active"),
            Bool,
            queue_size=1,
            latch=True,
        )
        self._force_insertion_attempted = False
        rospy.on_shutdown(self._handle_shutdown)

    def run(self) -> bool:
        rospy.loginfo("[usb_c_insertion] event=insertion_workflow_start")
        self._publish_status("test_insertion_target_pose", "running", message="preparing_insertion")
        if not self._robot.wait_for_motion_pipeline(
            timeout=self._pipeline_wait_timeout,
            require_pose_servo=True,
        ):
            rospy.logerr("[usb_c_insertion] event=insertion_workflow_failed reason=motion_pipeline_unavailable")
            self._publish_status(
                "test_insertion_target_pose",
                "failed",
                success=False,
                message="motion_pipeline_unavailable",
            )
            return False

        if not self._wait_for_wrench():
            rospy.logerr("[usb_c_insertion] event=insertion_workflow_failed reason=wrench_unavailable")
            self._publish_status(
                "test_insertion_target_pose",
                "failed",
                success=False,
                message="wrench_unavailable",
            )
            return False

        if self._zero_ft_before_contact:
            rospy.loginfo("[usb_c_insertion] event=insertion_workflow_zero_ft")
            self._publish_status("test_insertion_target_pose", "running", message="zero_ft_before_contact")
            if not self._ft.zero_sensor():
                rospy.logerr("[usb_c_insertion] event=insertion_workflow_failed reason=zero_ft_failed")
                self._publish_status(
                    "test_insertion_target_pose",
                    "failed",
                    success=False,
                    message="zero_ft_failed",
                )
                return False
            rospy.sleep(max(0.0, self._zero_ft_settle_time))

        self._publish_status("test_insertion_target_pose", "running", message="approaching_until_contact")
        contact = self._approach_until_contact()
        if not contact.success or contact.contact_pose is None:
            rospy.logerr(
                "[usb_c_insertion] event=insertion_workflow_failed reason=contact_approach_%s",
                contact.reason,
            )
            self._publish_status(
                "test_insertion_target_pose",
                "failed",
                success=False,
                message="contact_approach_%s" % contact.reason,
                values=self._contact_values(contact),
            )
            return False

        rospy.sleep(max(0.0, self._contact_settle_time))
        self._publish_status(
            "test_insertion_target_pose",
            "success",
            success=True,
            message=contact.reason,
            values=self._contact_values(contact),
        )

        self._publish_status("initial_approach_check", "running", message="checking_initial_approach")
        initial_check = self._check_initial_approach(contact.contact_pose)
        initial_values = self._initial_check_values(initial_check)
        if initial_check.terminal_failure:
            self._publish_status(
                "initial_approach_check",
                "failed",
                success=False,
                message=initial_check.reason,
                values=initial_values,
            )
            rospy.loginfo(
                "[usb_c_insertion] event=insertion_workflow_complete inserted_verified=false method=initial_approach_check reason=%s",
                initial_check.reason,
            )
            self._publish_status(
                "search_pattern",
                "skipped",
                success=False,
                message="initial_approach_terminal_failure",
            )
            self._skip_search_detail_stages("initial_approach_terminal_failure", success=False)
            self._publish_status(
                "force_controlled_insertion",
                "skipped",
                success=False,
                message="initial_approach_terminal_failure",
            )
            return False

        if initial_check.accepted:
            self._publish_status(
                "initial_approach_check",
                "success",
                success=True,
                message=initial_check.method,
                values=initial_values,
            )
            self._publish_status(
                "search_pattern",
                "skipped",
                success=True,
                message="initial_approach_accepted",
            )
            self._skip_search_detail_stages("initial_approach_accepted", success=True)
            insert_result = self._run_force_controlled_insertion_from_contact(
                contact.contact_pose,
                "initial_approach",
            )
            if insert_result is not None and insert_result.success:
                rospy.loginfo(
                    "[usb_c_insertion] event=insertion_workflow_complete inserted_verified=true method=initial_approach_force_insert inserted_depth=%.4f contact_force=%.3f",
                    insert_result.inserted_depth,
                    insert_result.contact_force,
                )
                return True

            reason = insert_result.reason if insert_result is not None else "insert_geometry_failed"
            inserted_depth = insert_result.inserted_depth if insert_result is not None else 0.0
            contact_force = insert_result.contact_force if insert_result is not None else 0.0
            rospy.loginfo(
                "[usb_c_insertion] event=insertion_workflow_complete inserted_verified=false method=initial_approach_force_insert reason=%s inserted_depth=%.4f contact_force=%.3f",
                reason,
                inserted_depth,
                contact_force,
            )
            return False

        self._publish_status(
            "initial_approach_check",
            "success",
            success=True,
            message="initial_approach_not_accepted",
            values=initial_values,
        )

        if not self._search_enabled:
            rospy.loginfo("[usb_c_insertion] event=insertion_workflow_complete inserted_verified=false method=pre_search_checks")
            self._publish_status(
                "search_pattern",
                "skipped",
                success=False,
                message="search_disabled_after_initial_check",
            )
            self._skip_search_detail_stages("search_disabled_after_initial_check", success=False)
            self._publish_status(
                "force_controlled_insertion",
                "skipped",
                success=False,
                message="no_valid_insertion_candidate",
            )
            return False

        self._publish_status("search_pattern", "running", message="running_spiral_search")
        search_result = self._run_spiral_search(contact.contact_pose)
        if search_result.success:
            self._publish_status(
                "search_pattern",
                "success",
                success=True,
                message=search_result.method,
                values=self._search_values(search_result),
            )
            if not self._force_insertion_attempted:
                self._publish_status(
                    "force_controlled_insertion",
                    "skipped",
                    success=True,
                    message="force_insert_disabled_after_candidate",
                )
            rospy.loginfo(
                "[usb_c_insertion] event=insertion_workflow_complete inserted_verified=true method=%s search_steps=%d/%d inserted_depth=%.4f contact_force=%.3f",
                search_result.method,
                search_result.completed_steps,
                search_result.total_steps,
                search_result.inserted_depth,
                search_result.contact_force,
            )
            return True

        self._publish_status(
            "search_pattern",
            "failed",
            success=False,
            message=search_result.reason,
            values=self._search_values(search_result),
        )
        if not self._force_insertion_attempted:
            self._publish_status(
                "force_controlled_insertion",
                "skipped",
                success=False,
                message="no_valid_insertion_candidate",
            )
        rospy.loginfo(
            "[usb_c_insertion] event=insertion_workflow_complete inserted_verified=false method=spiral_search reason=%s search_steps=%d/%d inserted_depth=%.4f contact_force=%.3f",
            search_result.reason,
            search_result.completed_steps,
            search_result.total_steps,
            search_result.inserted_depth,
            search_result.contact_force,
        )
        return False

    def _check_initial_approach(self, contact_pose: PoseStamped) -> InitialApproachCheckResult:
        pull_result = self._verify_pull_retention(contact_pose)
        if pull_result.success:
            rospy.loginfo(
                "[usb_c_insertion] event=insertion_workflow_initial_approach_accepted method=pull_retention pull_force=%.3f retraction=%.4f",
                pull_result.pull_force,
                pull_result.retraction_distance,
            )
            return InitialApproachCheckResult(
                True,
                False,
                pull_result.reason,
                "pull_retention",
                pull_result,
                0.0,
                0.0,
            )

        rospy.loginfo(
            "[usb_c_insertion] event=insertion_workflow_pull_retention_not_verified reason=%s pull_force=%.3f retraction=%.4f",
            pull_result.reason,
            pull_result.pull_force,
            pull_result.retraction_distance,
        )
        if self._return_to_contact_after_failed_pull:
            if not self._move_to_pose(contact_pose, "return_to_contact_after_failed_pull", self._return_timeout):
                rospy.loginfo(
                    "[usb_c_insertion] event=insertion_workflow_complete inserted_verified=false method=return_to_contact reason=return_to_contact_failed"
                )
                return InitialApproachCheckResult(
                    False,
                    True,
                    "return_to_contact_failed",
                    "return_to_contact",
                    pull_result,
                    0.0,
                    0.0,
                )

        if self._lateral_verify_after_failed_pull:
            if not self._zero_ft_for_verification_stage("lateral_after_failed_pull"):
                return InitialApproachCheckResult(
                    False,
                    True,
                    "lateral_zero_ft_failed",
                    "lateral_counterforce",
                    pull_result,
                    0.0,
                    0.0,
                )
            lateral_result = self._post_insertion_verifier.verify_retention(
                move_timeout=self._lateral_verify_timeout
            )
            if lateral_result.success:
                rospy.loginfo(
                    "[usb_c_insertion] event=insertion_workflow_initial_approach_accepted method=lateral_counterforce counterforce_y=%.3f counterforce_z=%.3f",
                    lateral_result.counterforce_y,
                    lateral_result.counterforce_z,
                )
                return InitialApproachCheckResult(
                    True,
                    False,
                    lateral_result.reason,
                    "lateral_counterforce",
                    pull_result,
                    lateral_result.counterforce_y,
                    lateral_result.counterforce_z,
                )

            rospy.loginfo(
                "[usb_c_insertion] event=insertion_workflow_lateral_not_verified reason=%s counterforce_y=%.3f counterforce_z=%.3f",
                lateral_result.reason,
                lateral_result.counterforce_y,
                lateral_result.counterforce_z,
            )
            return InitialApproachCheckResult(
                False,
                False,
                lateral_result.reason,
                "lateral_counterforce",
                pull_result,
                lateral_result.counterforce_y,
                lateral_result.counterforce_z,
            )

        rospy.loginfo("[usb_c_insertion] event=insertion_workflow_lateral_skipped_after_failed_pull")
        return InitialApproachCheckResult(
            False,
            False,
            pull_result.reason,
            "pull_retention",
            pull_result,
            0.0,
            0.0,
        )

    def _run_force_controlled_insertion_from_contact(self, contact_pose: PoseStamped, source: str):
        try:
            reference_surface_xyz = self._pose_xyz(contact_pose)
            probe_direction = self._tool_z_direction(contact_pose, self._approach_tool_z_sign)
        except ValueError as exc:
            rospy.logerr("[usb_c_insertion] event=insertion_workflow_force_insert_failed reason=%s", exc)
            self._publish_status(
                "force_controlled_insertion",
                "failed",
                success=False,
                message="insert_geometry_failed",
                values={"source": source, "reason": str(exc)},
            )
            return None
        return self._run_force_controlled_insertion(reference_surface_xyz, probe_direction, source)

    def _run_force_controlled_insertion(self, reference_surface_xyz, probe_direction_xyz, source: str):
        self._force_insertion_attempted = True
        direction = self._normalize_vector(probe_direction_xyz)
        self._publish_status(
            "force_controlled_insertion",
            "running",
            message="force_insert_%s" % source,
            values={
                "source": source,
                "reference_x": round(reference_surface_xyz[0], 5),
                "reference_y": round(reference_surface_xyz[1], 5),
                "reference_z": round(reference_surface_xyz[2], 5),
                "direction_x": round(direction[0], 5),
                "direction_y": round(direction[1], 5),
                "direction_z": round(direction[2], 5),
            },
        )
        insert_result = self._insertion_controller.insert_until_depth(
            reference_surface_xyz,
            direction,
        )
        values = self._insert_values(insert_result, source)
        if insert_result.success:
            self._publish_status(
                "force_controlled_insertion",
                "success",
                success=True,
                message=insert_result.reason,
                values=values,
            )
        else:
            self._publish_status(
                "force_controlled_insertion",
                "failed",
                success=False,
                message=insert_result.reason,
                values=values,
            )
        return insert_result

    def _approach_until_contact(self) -> ContactApproachResult:
        start_pose = self._tf.get_tool_pose_in_base()
        if start_pose is None:
            self._robot.stop_motion()
            return ContactApproachResult(False, "missing_initial_tf", None, None, 0.0, 0.0)

        direction = self._tool_z_direction(start_pose, self._approach_tool_z_sign)
        self._contact_detector.update_baseline()
        deadline = rospy.Time.now() + rospy.Duration.from_sec(max(0.1, self._approach_timeout))
        rate = rospy.Rate(max(1.0, self._command_rate))
        max_force = 0.0

        rospy.loginfo(
            "[usb_c_insertion] event=insertion_contact_approach_start direction=(%.4f,%.4f,%.4f) speed=%.4f threshold=%.3f max_travel=%.4f",
            direction[0],
            direction[1],
            direction[2],
            self._approach_speed,
            self._approach_contact_force,
            self._approach_max_travel,
        )
        while not rospy.is_shutdown():
            if rospy.Time.now() > deadline:
                self._robot.stop_motion()
                return ContactApproachResult(False, "timeout", None, None, max_force, 0.0)
            if self._ft.is_wrench_stale():
                self._robot.stop_motion()
                return ContactApproachResult(False, "stale_wrench", None, None, max_force, 0.0)

            current_pose = self._tf.get_tool_pose_in_base()
            if current_pose is None:
                self._robot.stop_motion()
                return ContactApproachResult(False, "missing_tf", None, None, max_force, 0.0)

            travel = self._project_pose_displacement(start_pose, current_pose, direction)
            contact_force = self._contact_detector.get_contact_force_along_direction(direction)
            max_force = max(max_force, contact_force)
            if contact_force >= self._approach_contact_force:
                self._robot.stop_motion()
                contact_point = self._pose_to_point_stamped(current_pose)
                rospy.loginfo(
                    "[usb_c_insertion] event=insertion_contact_approach_complete reason=contact_detected travel=%.4f contact_force=%.3f max_force=%.3f x=%.4f y=%.4f z=%.4f",
                    travel,
                    contact_force,
                    max_force,
                    current_pose.pose.position.x,
                    current_pose.pose.position.y,
                    current_pose.pose.position.z,
                )
                return ContactApproachResult(True, "contact_detected", current_pose, contact_point, contact_force, travel)

            if travel >= self._approach_max_travel:
                self._robot.stop_motion()
                return ContactApproachResult(False, "max_travel_reached", None, None, max_force, travel)

            self._robot.send_twist(
                direction[0] * self._approach_speed,
                direction[1] * self._approach_speed,
                direction[2] * self._approach_speed,
                0.0,
                0.0,
                0.0,
            )
            rospy.loginfo_throttle(
                0.5,
                "[usb_c_insertion] event=insertion_contact_approach_progress travel=%.4f contact_force=%.3f threshold=%.3f",
                travel,
                contact_force,
                self._approach_contact_force,
            )
            rate.sleep()

        self._robot.stop_motion()
        return ContactApproachResult(False, "shutdown", None, None, max_force, 0.0)

    def _verify_pull_retention(self, contact_pose: PoseStamped) -> PullRetentionResult:
        if self._zero_ft_before_pull:
            if not self._zero_ft_for_verification_stage("pull_retention"):
                return PullRetentionResult(False, "zero_ft_failed", 0.0, 0.0)

        start_pose = self._tf.get_tool_pose_in_base()
        if start_pose is None:
            self._robot.stop_motion()
            return PullRetentionResult(False, "missing_initial_tf", 0.0, 0.0)

        pull_direction = self._tool_z_direction(contact_pose, self._pull_tool_z_sign)
        deadline = rospy.Time.now() + rospy.Duration.from_sec(max(0.1, self._pull_timeout))
        rate = rospy.Rate(max(1.0, self._command_rate))
        held_since: Optional[rospy.Time] = None
        max_force = 0.0
        max_retraction = 0.0
        required_force = max(0.0, self._pull_force_target - self._pull_force_tolerance)

        rospy.loginfo(
            "[usb_c_insertion] event=insertion_pull_retention_start direction=(%.4f,%.4f,%.4f) force_mode=%s target_force=%.3f required_force=%.3f max_retraction=%.4f",
            pull_direction[0],
            pull_direction[1],
            pull_direction[2],
            self._pull_force_mode,
            self._pull_force_target,
            required_force,
            self._pull_max_retraction,
        )
        while not rospy.is_shutdown():
            if rospy.Time.now() > deadline:
                self._robot.stop_motion()
                return PullRetentionResult(False, "timeout", max_force, max_retraction)
            if self._ft.is_wrench_stale():
                self._robot.stop_motion()
                return PullRetentionResult(False, "stale_wrench", max_force, max_retraction)

            pose = self._tf.get_tool_pose_in_base()
            if pose is None:
                self._robot.stop_motion()
                return PullRetentionResult(False, "missing_tf", max_force, max_retraction)

            wrench = self._ft.get_filtered_wrench()
            pull_force = self._get_pull_retention_force(pull_direction, wrench)
            retraction = max(0.0, self._project_pose_displacement(start_pose, pose, pull_direction))
            max_force = max(max_force, pull_force)
            max_retraction = max(max_retraction, retraction)

            if retraction > self._pull_max_retraction:
                self._robot.stop_motion()
                return PullRetentionResult(False, "max_retraction_reached", max_force, max_retraction)

            if pull_force >= required_force:
                if held_since is None:
                    held_since = rospy.Time.now()
                elif (rospy.Time.now() - held_since).to_sec() >= self._pull_hold_time:
                    self._robot.stop_motion()
                    return PullRetentionResult(True, "force_held_without_retraction", pull_force, retraction)
                speed = 0.0
            else:
                held_since = None
                force_error = self._pull_force_target - pull_force
                if abs(force_error) <= self._pull_force_tolerance:
                    speed = 0.0
                else:
                    speed = max(0.0, min(self._pull_speed_limit, self._pull_force_gain * force_error))

            self._robot.send_twist(
                pull_direction[0] * speed,
                pull_direction[1] * speed,
                pull_direction[2] * speed,
                0.0,
                0.0,
                0.0,
            )
            rospy.loginfo_throttle(
                0.5,
                "[usb_c_insertion] event=insertion_pull_retention_progress pull_force=%.3f tool_force_z=%.3f target_force=%.3f required_force=%.3f retraction=%.4f max_retraction=%.4f speed=%.4f",
                pull_force,
                wrench.force_z,
                self._pull_force_target,
                required_force,
                retraction,
                self._pull_max_retraction,
                speed,
            )
            rate.sleep()

        self._robot.stop_motion()
        return PullRetentionResult(False, "shutdown", max_force, max_retraction)

    def _run_spiral_search(self, contact_pose: PoseStamped) -> SpiralSearchResult:
        self._publish_status("search_setup", "running", message="building_search_pattern")
        try:
            pattern = generate_preferred_square_spiral_pattern(
                step_x=self._search_step_tool_x,
                step_y=self._search_step_base_z,
                width=self._search_width_tool_x,
                height=self._search_height_base_z,
                preferred_x_sign=self._search_preferred_tool_x_sign,
                preferred_y_sign=self._search_preferred_base_z_sign,
            )
            probe_direction = self._tool_z_direction(contact_pose, self._approach_tool_z_sign)
            wall_tangent = self._horizontal_tool_x_direction(contact_pose)
        except ValueError as exc:
            values = self._search_setup_values(error=str(exc))
            self._publish_status(
                "search_setup",
                "failed",
                success=False,
                message="search_geometry_failed",
                values=values,
            )
            self._publish_status("search_center_probe", "skipped", success=False, message="search_setup_failed")
            self._publish_status("search_spiral_probe", "skipped", success=False, message="search_setup_failed")
            self._publish_status("search_candidate_check", "skipped", success=False, message="search_setup_failed")
            return SpiralSearchResult(False, "search_geometry_failed: %s" % exc, "spiral_search", 0, 0, 0.0, 0.0)

        pattern_steps = len(pattern)
        total_steps = pattern_steps + (1 if self._search_center_first_probe else 0)
        setup_values = self._search_setup_values(
            pattern_steps=pattern_steps,
            total_steps=total_steps,
            probe_direction=probe_direction,
            wall_tangent=wall_tangent,
        )
        if total_steps <= 0:
            self._publish_status(
                "search_setup",
                "failed",
                success=False,
                message="empty_search_pattern",
                values=setup_values,
            )
            self._publish_status("search_center_probe", "skipped", success=False, message="empty_search_pattern")
            self._publish_status("search_spiral_probe", "skipped", success=False, message="empty_search_pattern")
            self._publish_status("search_candidate_check", "skipped", success=False, message="empty_search_pattern")
            return SpiralSearchResult(False, "empty_search_pattern", "spiral_search", 0, 0, 0.0, 0.0)

        deadline = rospy.Time.now() + rospy.Duration.from_sec(max(0.1, self._search_timeout))
        surface_xyz = self._pose_xyz(contact_pose)
        completed_steps = 0
        last_inserted_depth = 0.0
        last_contact_force = 0.0

        rospy.loginfo(
            "[usb_c_insertion] event=insertion_spiral_search_start steps=%d center_first=%s step_tool_x=%.4f step_base_z=%.4f width_tool_x=%.4f height_base_z=%.4f contact_force_target=%.3f center_first_contact_force_target=%.3f center_first_min_probe_time=%.2f probe_speed_limit=%.4f socket_depth_threshold=%.4f",
            total_steps,
            str(self._search_center_first_probe).lower(),
            self._search_step_tool_x,
            self._search_step_base_z,
            self._search_width_tool_x,
            self._search_height_base_z,
            self._search_contact_force_target,
            self._search_center_first_contact_force_target,
            self._search_center_first_min_probe_time,
            self._search_probe_speed_limit,
            self._search_socket_depth_threshold,
        )
        self._publish_status(
            "search_setup",
            "success",
            success=True,
            message="search_pattern_ready",
            values=setup_values,
        )

        if self._search_center_first_probe:
            completed_steps = 1
            pre_probe_xyz = self._offset_xyz(surface_xyz, probe_direction, -self._search_pre_probe_clearance)
            pre_probe_pose = self._make_pose_like(contact_pose, pre_probe_xyz)
            center_values = self._search_probe_values(
                step_index=completed_steps,
                total_steps=total_steps,
                center_first=True,
                surface_xyz=surface_xyz,
                pre_probe_xyz=pre_probe_xyz,
                probe_direction_xyz=probe_direction,
                contact_force_target=max(
                    self._search_contact_force_target,
                    self._search_center_first_contact_force_target,
                ),
                probe_timeout=max(self._search_probe_timeout, self._search_center_first_probe_timeout),
                probe_max_travel=max(self._search_probe_max_travel, self._search_center_first_probe_max_travel),
            )
            self._publish_status(
                "search_center_probe",
                "running",
                message="moving_to_center_pre_probe",
                values=center_values,
            )
            if not self._move_to_pose(
                pre_probe_pose,
                "spiral_search_center_pre_probe",
                self._search_transfer_timeout,
            ):
                center_values["reason"] = "search_transfer_to_center_pre_probe_failed"
                self._publish_status(
                    "search_center_probe",
                    "failed",
                    success=False,
                    message="search_transfer_to_center_pre_probe_failed",
                    values=center_values,
                )
                self._publish_status("search_spiral_probe", "skipped", success=False, message="center_probe_failed")
                self._publish_status("search_candidate_check", "skipped", success=False, message="center_probe_failed")
                return SpiralSearchResult(
                    False,
                    "search_transfer_to_center_pre_probe_failed",
                    "spiral_search",
                    completed_steps,
                    total_steps,
                    last_inserted_depth,
                    last_contact_force,
                )
            rospy.sleep(self._search_pre_probe_settle_time)

            self._publish_status(
                "search_center_probe",
                "running",
                message="probing_center",
                values=center_values,
            )
            probe = self._probe_search_position(
                surface_xyz,
                probe_direction,
                completed_steps,
                total_steps,
                center_first=True,
            )
            last_inserted_depth = probe.inserted_depth
            last_contact_force = probe.max_contact_force
            center_values = self._search_probe_values(
                probe,
                step_index=completed_steps,
                total_steps=total_steps,
                center_first=True,
                surface_xyz=surface_xyz,
                pre_probe_xyz=pre_probe_xyz,
                probe_direction_xyz=probe_direction,
                contact_force_target=max(
                    self._search_contact_force_target,
                    self._search_center_first_contact_force_target,
                ),
                probe_timeout=max(self._search_probe_timeout, self._search_center_first_probe_timeout),
                probe_max_travel=max(self._search_probe_max_travel, self._search_center_first_probe_max_travel),
            )
            if not probe.success:
                self._publish_status(
                    "search_center_probe",
                    "failed",
                    success=False,
                    message=probe.reason,
                    values=center_values,
                )
                self._publish_status("search_spiral_probe", "skipped", success=False, message="center_probe_failed")
                self._publish_status("search_candidate_check", "skipped", success=False, message="center_probe_failed")
                return SpiralSearchResult(
                    False,
                    probe.reason,
                    "spiral_search",
                    completed_steps,
                    total_steps,
                    probe.inserted_depth,
                    probe.max_contact_force,
                )

            self._publish_status(
                "search_center_probe",
                "success",
                success=True,
                message="candidate_found" if probe.candidate_found else probe.reason,
                values=center_values,
            )
            candidate_result = self._candidate_search_result(
                surface_xyz,
                probe_direction,
                probe,
                completed_steps,
                total_steps,
            )
            if candidate_result is not None:
                self._publish_status(
                    "search_spiral_probe",
                    "skipped",
                    success=candidate_result.success,
                    message="candidate_result_before_spiral",
                )
                return candidate_result

            self._publish_status(
                "search_center_probe",
                "running",
                message="retracting_after_center_probe",
                values=center_values,
            )
            if not self._move_to_pose(
                pre_probe_pose,
                "spiral_search_center_retract",
                self._search_transfer_timeout,
            ):
                center_values["reason"] = "search_retract_after_center_probe_failed"
                self._publish_status(
                    "search_center_probe",
                    "failed",
                    success=False,
                    message="search_retract_after_center_probe_failed",
                    values=center_values,
                )
                self._publish_status("search_spiral_probe", "skipped", success=False, message="center_retract_failed")
                self._publish_status("search_candidate_check", "skipped", success=False, message="center_retract_failed")
                return SpiralSearchResult(
                    False,
                    "search_retract_after_center_probe_failed",
                    "spiral_search",
                    completed_steps,
                    total_steps,
                    last_inserted_depth,
                    last_contact_force,
                )
            self._publish_status(
                "search_center_probe",
                "success",
                success=True,
                message="center_probe_complete",
                values=center_values,
            )
        else:
            self._publish_status("search_center_probe", "skipped", success=True, message="center_first_disabled")

        if pattern_steps <= 0:
            values = {
                "completed_steps": int(completed_steps),
                "total_steps": int(total_steps),
                "pattern_steps": int(pattern_steps),
                "inserted_depth": round(last_inserted_depth, 5),
                "contact_force": round(last_contact_force, 3),
            }
            self._publish_status(
                "search_spiral_probe",
                "skipped",
                success=False,
                message="no_spiral_offsets",
                values=values,
            )
            self._publish_status(
                "search_candidate_check",
                "skipped",
                success=False,
                message="no_candidate_found",
                values=values,
            )
            return SpiralSearchResult(
                False,
                "search_pattern_exhausted",
                "spiral_search",
                completed_steps,
                total_steps,
                last_inserted_depth,
                last_contact_force,
            )

        step_offset = 1 if self._search_center_first_probe else 0
        self._publish_status(
            "search_spiral_probe",
            "running",
            message="starting_spiral_probes",
            values={
                "pattern_steps": int(pattern_steps),
                "step_offset": int(step_offset),
                "total_steps": int(total_steps),
            },
        )
        for pattern_index, offset in enumerate(pattern, start=1):
            completed_steps = pattern_index + step_offset
            if rospy.is_shutdown():
                values = {
                    "completed_steps": int(completed_steps),
                    "total_steps": int(total_steps),
                    "inserted_depth": round(last_inserted_depth, 5),
                    "contact_force": round(last_contact_force, 3),
                }
                self._publish_status("search_spiral_probe", "failed", success=False, message="shutdown", values=values)
                self._publish_status("search_candidate_check", "skipped", success=False, message="search_stopped")
                return SpiralSearchResult(
                    False,
                    "shutdown",
                    "spiral_search",
                    completed_steps,
                    total_steps,
                    last_inserted_depth,
                    last_contact_force,
                )
            if rospy.Time.now() > deadline:
                values = {
                    "completed_steps": int(completed_steps),
                    "total_steps": int(total_steps),
                    "inserted_depth": round(last_inserted_depth, 5),
                    "contact_force": round(last_contact_force, 3),
                }
                self._publish_status(
                    "search_spiral_probe",
                    "failed",
                    success=False,
                    message="search_timeout",
                    values=values,
                )
                self._publish_status("search_candidate_check", "skipped", success=False, message="search_timeout")
                return SpiralSearchResult(
                    False,
                    "search_timeout",
                    "spiral_search",
                    completed_steps,
                    total_steps,
                    last_inserted_depth,
                    last_contact_force,
                )

            surface_xyz = (
                surface_xyz[0] + offset.dx * wall_tangent[0],
                surface_xyz[1] + offset.dx * wall_tangent[1],
                surface_xyz[2] + offset.dy,
            )
            pre_probe_xyz = self._offset_xyz(surface_xyz, probe_direction, -self._search_pre_probe_clearance)
            pre_probe_pose = self._make_pose_like(contact_pose, pre_probe_xyz)
            probe_values = self._search_probe_values(
                step_index=completed_steps,
                total_steps=total_steps,
                center_first=False,
                surface_xyz=surface_xyz,
                pre_probe_xyz=pre_probe_xyz,
                probe_direction_xyz=probe_direction,
                contact_force_target=self._search_contact_force_target,
                probe_timeout=self._search_probe_timeout,
                probe_max_travel=self._search_probe_max_travel,
                offset=offset,
            )

            self._publish_status(
                "search_spiral_probe",
                "running",
                message="moving_to_spiral_pre_probe",
                values=probe_values,
            )
            if not self._move_to_pose(
                pre_probe_pose,
                "spiral_search_pre_probe_%d" % completed_steps,
                self._search_transfer_timeout,
            ):
                probe_values["reason"] = "search_transfer_to_pre_probe_failed"
                self._publish_status(
                    "search_spiral_probe",
                    "failed",
                    success=False,
                    message="search_transfer_to_pre_probe_failed",
                    values=probe_values,
                )
                self._publish_status("search_candidate_check", "skipped", success=False, message="spiral_probe_failed")
                return SpiralSearchResult(
                    False,
                    "search_transfer_to_pre_probe_failed",
                    "spiral_search",
                    completed_steps,
                    total_steps,
                    last_inserted_depth,
                    last_contact_force,
                )
            rospy.sleep(self._search_pre_probe_settle_time)

            self._publish_status(
                "search_spiral_probe",
                "running",
                message="probing_spiral_step",
                values=probe_values,
            )
            probe = self._probe_search_position(surface_xyz, probe_direction, completed_steps, total_steps)
            last_inserted_depth = probe.inserted_depth
            last_contact_force = probe.max_contact_force
            probe_values = self._search_probe_values(
                probe,
                step_index=completed_steps,
                total_steps=total_steps,
                center_first=False,
                surface_xyz=surface_xyz,
                pre_probe_xyz=pre_probe_xyz,
                probe_direction_xyz=probe_direction,
                contact_force_target=self._search_contact_force_target,
                probe_timeout=self._search_probe_timeout,
                probe_max_travel=self._search_probe_max_travel,
                offset=offset,
            )
            if not probe.success:
                self._publish_status(
                    "search_spiral_probe",
                    "failed",
                    success=False,
                    message=probe.reason,
                    values=probe_values,
                )
                self._publish_status("search_candidate_check", "skipped", success=False, message="spiral_probe_failed")
                return SpiralSearchResult(
                    False,
                    probe.reason,
                    "spiral_search",
                    completed_steps,
                    total_steps,
                    probe.inserted_depth,
                    probe.max_contact_force,
                )

            self._publish_status(
                "search_spiral_probe",
                "success" if probe.candidate_found else "running",
                success=True if probe.candidate_found else None,
                message="candidate_found" if probe.candidate_found else probe.reason,
                values=probe_values,
            )
            candidate_result = self._candidate_search_result(
                surface_xyz,
                probe_direction,
                probe,
                completed_steps,
                total_steps,
            )
            if candidate_result is not None:
                return candidate_result

            self._publish_status(
                "search_spiral_probe",
                "running",
                message="retracting_after_spiral_probe",
                values=probe_values,
            )
            if not self._move_to_pose(
                pre_probe_pose,
                "spiral_search_retract_%d" % completed_steps,
                self._search_transfer_timeout,
            ):
                probe_values["reason"] = "search_retract_after_probe_failed"
                self._publish_status(
                    "search_spiral_probe",
                    "failed",
                    success=False,
                    message="search_retract_after_probe_failed",
                    values=probe_values,
                )
                self._publish_status("search_candidate_check", "skipped", success=False, message="spiral_retract_failed")
                return SpiralSearchResult(
                    False,
                    "search_retract_after_probe_failed",
                    "spiral_search",
                    completed_steps,
                    total_steps,
                    last_inserted_depth,
                    last_contact_force,
                )

        values = {
            "completed_steps": int(completed_steps),
            "total_steps": int(total_steps),
            "pattern_steps": int(pattern_steps),
            "inserted_depth": round(last_inserted_depth, 5),
            "contact_force": round(last_contact_force, 3),
            "socket_depth_threshold": round(self._search_socket_depth_threshold, 5),
        }
        self._publish_status(
            "search_spiral_probe",
            "failed",
            success=False,
            message="search_pattern_exhausted",
            values=values,
        )
        self._publish_status("search_candidate_check", "skipped", success=False, message="no_candidate_found")
        return SpiralSearchResult(
            False,
            "search_pattern_exhausted",
            "spiral_search",
            completed_steps,
            total_steps,
            last_inserted_depth,
            last_contact_force,
        )

    def _probe_search_position(
        self,
        surface_xyz,
        probe_direction_xyz,
        step_index: int,
        total_steps: int,
        center_first: bool = False,
    ) -> SearchProbeResult:
        direction = self._normalize_vector(probe_direction_xyz)
        contact_force_target = self._search_contact_force_target
        probe_timeout = self._search_probe_timeout
        probe_max_travel = self._search_probe_max_travel
        if center_first:
            contact_force_target = max(contact_force_target, self._search_center_first_contact_force_target)
            probe_timeout = max(probe_timeout, self._search_center_first_probe_timeout)
            probe_max_travel = max(probe_max_travel, self._search_center_first_probe_max_travel)
        min_probe_time = self._search_center_first_min_probe_time if center_first else 0.0

        def finish(
            success: bool,
            reason: str,
            candidate_found: bool,
            final_pose: Optional[PoseStamped],
            inserted_depth: float,
            contact_force: float,
            max_contact_force: float,
            travel_distance: float,
        ) -> SearchProbeResult:
            result = SearchProbeResult(
                success,
                reason,
                candidate_found,
                final_pose,
                inserted_depth,
                contact_force,
                max_contact_force,
                travel_distance,
            )
            self._log_search_probe_result(
                step_index,
                total_steps,
                center_first,
                surface_xyz,
                direction,
                contact_force_target,
                probe_timeout,
                probe_max_travel,
                result,
            )
            return result

        start_pose = self._tf.get_tool_pose_in_base()
        if start_pose is None:
            self._robot.stop_motion()
            return finish(False, "missing_initial_tf", False, None, 0.0, 0.0, 0.0, 0.0)
        if self._search_probe_speed_limit <= 0.0:
            self._robot.stop_motion()
            return finish(False, "invalid_search_probe_speed", False, start_pose, 0.0, 0.0, 0.0, 0.0)

        start_xyz = self._pose_xyz(start_pose)

        deadline = rospy.Time.now() + rospy.Duration.from_sec(max(0.1, probe_timeout))
        probe_start_time = rospy.Time.now()
        rate = rospy.Rate(max(1.0, self._command_rate))
        max_contact_force = 0.0
        last_inserted_depth = 0.0
        last_travel = 0.0
        self._contact_detector.update_baseline()

        rospy.loginfo(
            "[usb_c_insertion] event=insertion_spiral_probe_start step=%d total=%d center_first=%s surface=(%.4f,%.4f,%.4f) direction=(%.4f,%.4f,%.4f) contact_force_target=%.3f timeout=%.2f max_travel=%.4f",
            step_index,
            total_steps,
            str(center_first).lower(),
            surface_xyz[0],
            surface_xyz[1],
            surface_xyz[2],
            direction[0],
            direction[1],
            direction[2],
            contact_force_target,
            probe_timeout,
            probe_max_travel,
        )
        self._begin_direct_probe_control()
        try:
            while not rospy.is_shutdown():
                if rospy.Time.now() > deadline:
                    return finish(
                        False,
                        "search_probe_timeout",
                        False,
                        self._tf.get_tool_pose_in_base(),
                        last_inserted_depth,
                        max_contact_force,
                        max_contact_force,
                        last_travel,
                    )
                if self._ft.is_wrench_stale():
                    return finish(
                        False,
                        "stale_wrench",
                        False,
                        self._tf.get_tool_pose_in_base(),
                        last_inserted_depth,
                        max_contact_force,
                        max_contact_force,
                        last_travel,
                    )

                pose = self._tf.get_tool_pose_in_base()
                if pose is None:
                    return finish(
                        False,
                        "missing_tf",
                        False,
                        None,
                        last_inserted_depth,
                        max_contact_force,
                        max_contact_force,
                        last_travel,
                    )

                current_xyz = self._pose_xyz(pose)
                elapsed = (rospy.Time.now() - probe_start_time).to_sec()
                contact_force = self._contact_detector.get_contact_force_along_direction(direction)
                inserted_depth = self._project_xyz_displacement(surface_xyz, current_xyz, direction)
                travel = self._project_xyz_displacement(start_xyz, current_xyz, direction)
                max_contact_force = max(max_contact_force, contact_force)
                last_inserted_depth = inserted_depth
                last_travel = travel

                if (
                    self._search_contact_force_abort_threshold > 0.0
                    and contact_force >= self._search_contact_force_abort_threshold
                ):
                    rospy.logwarn(
                        "[usb_c_insertion] event=insertion_spiral_probe_force_abort step=%d total=%d contact_force=%.3f abort_threshold=%.3f inserted_depth=%.4f travel=%.4f",
                        step_index,
                        total_steps,
                        contact_force,
                        self._search_contact_force_abort_threshold,
                        inserted_depth,
                        travel,
                    )
                    return finish(
                        False,
                        "search_probe_contact_force_abort",
                        False,
                        pose,
                        inserted_depth,
                        contact_force,
                        max_contact_force,
                        travel,
                    )

                if inserted_depth >= self._search_socket_depth_threshold:
                    rospy.loginfo(
                        "[usb_c_insertion] event=insertion_spiral_candidate_found step=%d total=%d inserted_depth=%.4f threshold=%.4f contact_force=%.3f max_contact_force=%.3f travel=%.4f",
                        step_index,
                        total_steps,
                        inserted_depth,
                        self._search_socket_depth_threshold,
                        contact_force,
                        max_contact_force,
                        travel,
                    )
                    return finish(
                        True,
                        "socket_depth_reached",
                        True,
                        pose,
                        inserted_depth,
                        contact_force,
                        max_contact_force,
                        travel,
                    )

                if contact_force >= contact_force_target:
                    if elapsed < min_probe_time:
                        rospy.loginfo_throttle(
                            0.5,
                            "[usb_c_insertion] event=insertion_spiral_center_probe_hold step=%d total=%d elapsed=%.2f min_time=%.2f contact_force=%.3f target=%.3f inserted_depth=%.4f travel=%.4f",
                            step_index,
                            total_steps,
                            elapsed,
                            min_probe_time,
                            contact_force,
                            contact_force_target,
                            inserted_depth,
                            travel,
                        )
                    else:
                        rospy.loginfo(
                            "[usb_c_insertion] event=insertion_spiral_wall_contact step=%d total=%d inserted_depth=%.4f contact_force=%.3f target=%.3f max_contact_force=%.3f travel=%.4f",
                            step_index,
                            total_steps,
                            inserted_depth,
                            contact_force,
                            contact_force_target,
                            max_contact_force,
                            travel,
                        )
                        return finish(
                            True,
                            "wall_contact_reached",
                            False,
                            pose,
                            inserted_depth,
                            contact_force,
                            max_contact_force,
                            travel,
                        )

                if travel >= probe_max_travel:
                    return finish(
                        False,
                        "search_probe_max_travel_reached",
                        False,
                        pose,
                        inserted_depth,
                        contact_force,
                        max_contact_force,
                        travel,
                    )

                speed = self._search_probe_speed_for_force(contact_force, contact_force_target)
                self._publish_direct_twist(
                    direction[0] * speed,
                    direction[1] * speed,
                    direction[2] * speed,
                )
                rospy.loginfo_throttle(
                    0.5,
                    "[usb_c_insertion] event=insertion_spiral_probe_progress step=%d total=%d inserted_depth=%.4f socket_depth_threshold=%.4f contact_force=%.3f target=%.3f max_contact_force=%.3f travel=%.4f speed=%.4f",
                    step_index,
                    total_steps,
                    inserted_depth,
                    self._search_socket_depth_threshold,
                    contact_force,
                    contact_force_target,
                    max_contact_force,
                    travel,
                    speed,
                )
                rate.sleep()
        finally:
            self._end_direct_probe_control()

        return finish(
            False,
            "shutdown",
            False,
            None,
            last_inserted_depth,
            max_contact_force,
            max_contact_force,
            last_travel,
        )

    def _candidate_search_result(
        self,
        reference_surface_xyz,
        probe_direction_xyz,
        probe: SearchProbeResult,
        completed_steps: int,
        total_steps: int,
    ) -> Optional[SpiralSearchResult]:
        if not probe.candidate_found:
            return None
        center_candidate = bool(self._search_center_first_probe and completed_steps == 1)
        contact_force_target = self._search_contact_force_target
        probe_timeout = self._search_probe_timeout
        probe_max_travel = self._search_probe_max_travel
        if center_candidate:
            contact_force_target = max(contact_force_target, self._search_center_first_contact_force_target)
            probe_timeout = max(probe_timeout, self._search_center_first_probe_timeout)
            probe_max_travel = max(probe_max_travel, self._search_center_first_probe_max_travel)
        values = self._search_probe_values(
            probe,
            step_index=completed_steps,
            total_steps=total_steps,
            center_first=center_candidate,
            surface_xyz=reference_surface_xyz,
            probe_direction_xyz=probe_direction_xyz,
            contact_force_target=contact_force_target,
            probe_timeout=probe_timeout,
            probe_max_travel=probe_max_travel,
        )
        values.update(self._xyz_values("reference_surface", reference_surface_xyz))
        self._publish_status(
            "search_candidate_check",
            "running",
            message="candidate_found",
            values=values,
        )
        if probe.final_pose is None:
            values["reason"] = "candidate_missing_final_pose"
            self._publish_status(
                "search_candidate_check",
                "failed",
                success=False,
                message="candidate_missing_final_pose",
                values=values,
            )
            return SpiralSearchResult(
                False,
                "candidate_missing_final_pose",
                "spiral_search",
                completed_steps,
                total_steps,
                probe.inserted_depth,
                probe.max_contact_force,
            )

        verified, method_or_reason, verification_values = self._verify_search_candidate(
            reference_surface_xyz,
            probe_direction_xyz,
            probe.final_pose,
        )
        values["verification_result"] = method_or_reason
        values.update(verification_values)
        if verified:
            self._publish_status(
                "search_candidate_check",
                "success",
                success=True,
                message=method_or_reason,
                values=values,
            )
            return SpiralSearchResult(
                True,
                "candidate_verified",
                method_or_reason,
                completed_steps,
                total_steps,
                probe.inserted_depth,
                probe.max_contact_force,
            )
        if not self._search_continue_after_unverified_candidate:
            self._publish_status(
                "search_candidate_check",
                "failed",
                success=False,
                message=method_or_reason,
                values=values,
            )
            return SpiralSearchResult(
                False,
                method_or_reason,
                "spiral_search",
                completed_steps,
                total_steps,
                probe.inserted_depth,
                probe.max_contact_force,
            )
        self._publish_status(
            "search_candidate_check",
            "running",
            message="candidate_rejected_continuing_search",
            values=values,
        )
        return None

    def _verify_search_candidate(
        self,
        reference_surface_xyz,
        probe_direction_xyz,
        candidate_pose: PoseStamped,
    ) -> Tuple[bool, str, Dict]:
        verification_pose = candidate_pose
        verification_values: Dict = {
            "candidate_pose": self._pose_values(candidate_pose),
            "insert_after_candidate": bool(self._search_insert_after_candidate),
            "pull_verify": bool(self._search_candidate_pull_verify),
            "lateral_verify": bool(self._search_candidate_lateral_verify),
        }
        if self._search_insert_after_candidate:
            self._publish_status(
                "search_candidate_check",
                "running",
                message="force_inserting_candidate",
                values={
                    "candidate_pose": self._pose_values(candidate_pose),
                    "verification": "force_insert",
                },
            )
            insert_result = self._run_force_controlled_insertion(
                reference_surface_xyz,
                probe_direction_xyz,
                "search_candidate",
            )
            rospy.loginfo(
                "[usb_c_insertion] event=insertion_spiral_candidate_insert_complete success=%s reason=%s inserted_depth=%.4f contact_force=%.3f",
                str(insert_result.success).lower(),
                insert_result.reason,
                insert_result.inserted_depth,
                insert_result.contact_force,
            )
            verification_values.update(
                {
                    "insert_reason": insert_result.reason,
                    "inserted_depth_after_candidate": round(insert_result.inserted_depth, 5),
                    "insert_contact_force": round(insert_result.contact_force, 3),
                }
            )
            if not insert_result.success:
                return False, "spiral_search_insert_failed", verification_values
            current_pose = self._tf.get_tool_pose_in_base()
            if current_pose is not None:
                verification_pose = current_pose
                verification_values["verification_pose"] = self._pose_values(current_pose)

        if not self._search_candidate_pull_verify and not self._search_candidate_lateral_verify:
            method = "spiral_search_insert" if self._search_insert_after_candidate else "spiral_search_socket_depth"
            return True, method, verification_values

        if self._search_candidate_pull_verify:
            self._publish_status(
                "search_candidate_check",
                "running",
                message="pull_retention_check",
                values={"verification_pose": self._pose_values(verification_pose)},
            )
            pull_result = self._verify_pull_retention(verification_pose)
            if pull_result.success:
                rospy.loginfo(
                    "[usb_c_insertion] event=insertion_spiral_candidate_verified method=pull_retention pull_force=%.3f retraction=%.4f",
                    pull_result.pull_force,
                    pull_result.retraction_distance,
                )
                verification_values.update(
                    {
                        "pull_reason": pull_result.reason,
                        "pull_force": round(pull_result.pull_force, 3),
                        "pull_retraction": round(pull_result.retraction_distance, 5),
                    }
                )
                return True, "spiral_search_pull_retention", verification_values
            rospy.loginfo(
                "[usb_c_insertion] event=insertion_spiral_candidate_pull_not_verified reason=%s pull_force=%.3f retraction=%.4f",
                pull_result.reason,
                pull_result.pull_force,
                pull_result.retraction_distance,
            )
            verification_values.update(
                {
                    "pull_reason": pull_result.reason,
                    "pull_force": round(pull_result.pull_force, 3),
                    "pull_retraction": round(pull_result.retraction_distance, 5),
                }
            )
            self._publish_status(
                "search_candidate_check",
                "running",
                message="pull_retention_not_verified",
                values={
                    "pull_reason": pull_result.reason,
                    "pull_force": round(pull_result.pull_force, 3),
                    "pull_retraction": round(pull_result.retraction_distance, 5),
                },
            )

        if self._search_candidate_lateral_verify:
            if self._search_candidate_pull_verify:
                if not self._move_to_pose(
                    verification_pose,
                    "return_to_spiral_candidate_after_pull",
                    self._return_timeout,
                ):
                    return False, "spiral_search_candidate_return_after_pull_failed", verification_values
            if not self._zero_ft_for_verification_stage("spiral_candidate_lateral"):
                return False, "spiral_search_candidate_lateral_zero_ft_failed", verification_values
            self._publish_status(
                "search_candidate_check",
                "running",
                message="lateral_counterforce_check",
                values={"verification_pose": self._pose_values(verification_pose)},
            )
            lateral_result = self._post_insertion_verifier.verify_retention(
                move_timeout=self._lateral_verify_timeout
            )
            if lateral_result.success:
                rospy.loginfo(
                    "[usb_c_insertion] event=insertion_spiral_candidate_verified method=lateral_counterforce counterforce_y=%.3f counterforce_z=%.3f",
                    lateral_result.counterforce_y,
                    lateral_result.counterforce_z,
                )
                verification_values.update(
                    {
                        "lateral_reason": lateral_result.reason,
                        "counterforce_y": round(lateral_result.counterforce_y, 3),
                        "counterforce_z": round(lateral_result.counterforce_z, 3),
                    }
                )
                return True, "spiral_search_lateral_counterforce", verification_values
            rospy.loginfo(
                "[usb_c_insertion] event=insertion_spiral_candidate_lateral_not_verified reason=%s counterforce_y=%.3f counterforce_z=%.3f",
                lateral_result.reason,
                lateral_result.counterforce_y,
                lateral_result.counterforce_z,
            )
            verification_values.update(
                {
                    "lateral_reason": lateral_result.reason,
                    "counterforce_y": round(lateral_result.counterforce_y, 3),
                    "counterforce_z": round(lateral_result.counterforce_z, 3),
                }
            )

        return False, "spiral_search_candidate_not_verified", verification_values

    def _skip_search_detail_stages(self, message: str, success: Optional[bool] = None) -> None:
        for stage_id in ("search_setup", "search_center_probe", "search_spiral_probe", "search_candidate_check"):
            self._publish_status(stage_id, "skipped", success=success, message=message)

    def _log_search_probe_result(
        self,
        step_index: int,
        total_steps: int,
        center_first: bool,
        surface_xyz,
        direction_xyz,
        contact_force_target: float,
        probe_timeout: float,
        probe_max_travel: float,
        result: SearchProbeResult,
    ) -> None:
        classification = self._classify_search_probe_result(
            result.reason,
            result.inserted_depth,
            result.contact_force,
            contact_force_target,
        )
        final_pose_available = result.final_pose is not None
        if result.final_pose is not None:
            final_xyz = self._pose_xyz(result.final_pose)
        else:
            final_xyz = (float("nan"), float("nan"), float("nan"))
        rospy.loginfo(
            "[usb_c_insertion] event=insertion_spiral_probe_result step=%d total=%d center_first=%s classification=%s success=%s candidate_found=%s reason=%s inserted_depth=%.4f socket_depth_threshold=%.4f travel_distance=%.4f contact_force=%.3f contact_force_target=%.3f max_contact_force=%.3f max_travel=%.4f timeout=%.2f surface=(%.4f,%.4f,%.4f) direction=(%.4f,%.4f,%.4f) final_pose_available=%s final=(%.4f,%.4f,%.4f)",
            step_index,
            total_steps,
            str(center_first).lower(),
            classification,
            str(result.success).lower(),
            str(result.candidate_found).lower(),
            result.reason,
            result.inserted_depth,
            self._search_socket_depth_threshold,
            result.travel_distance,
            result.contact_force,
            contact_force_target,
            result.max_contact_force,
            probe_max_travel,
            probe_timeout,
            surface_xyz[0],
            surface_xyz[1],
            surface_xyz[2],
            direction_xyz[0],
            direction_xyz[1],
            direction_xyz[2],
            str(final_pose_available).lower(),
            final_xyz[0],
            final_xyz[1],
            final_xyz[2],
        )

    def _classify_search_probe_result(
        self,
        reason: str,
        inserted_depth: float,
        contact_force: float,
        contact_force_target: float,
    ) -> str:
        if reason == "socket_depth_reached" or inserted_depth >= self._search_socket_depth_threshold:
            return "socket_candidate"
        if reason == "wall_contact_reached" or contact_force >= contact_force_target:
            return "wall_contact"
        if reason in ("search_probe_timeout", "search_timeout"):
            return "timeout"
        if reason == "search_probe_max_travel_reached":
            return "max_travel"
        if reason == "search_probe_contact_force_abort":
            return "force_abort"
        if reason in ("missing_initial_tf", "missing_tf"):
            return "tf_error"
        if reason == "stale_wrench":
            return "wrench_error"
        return "failed" if reason else "unknown"

    def _search_setup_values(
        self,
        pattern_steps: int = 0,
        total_steps: int = 0,
        probe_direction=None,
        wall_tangent=None,
        error: str = "",
    ) -> Dict:
        values = {
            "pattern_steps": int(pattern_steps),
            "total_steps": int(total_steps),
            "center_first": bool(self._search_center_first_probe),
            "timeout": round(self._search_timeout, 2),
            "step_tool_x": round(self._search_step_tool_x, 5),
            "step_base_z": round(self._search_step_base_z, 5),
            "width_tool_x": round(self._search_width_tool_x, 5),
            "height_base_z": round(self._search_height_base_z, 5),
            "preferred_tool_x_sign": round(self._search_preferred_tool_x_sign, 3),
            "preferred_base_z_sign": round(self._search_preferred_base_z_sign, 3),
            "pre_probe_clearance": round(self._search_pre_probe_clearance, 5),
            "contact_force_target": round(self._search_contact_force_target, 3),
            "center_first_contact_force_target": round(self._search_center_first_contact_force_target, 3),
            "probe_speed_limit": round(self._search_probe_speed_limit, 5),
            "probe_min_speed": round(self._search_probe_min_speed, 5),
            "probe_timeout": round(self._search_probe_timeout, 2),
            "probe_max_travel": round(self._search_probe_max_travel, 5),
            "center_first_probe_timeout": round(self._search_center_first_probe_timeout, 2),
            "center_first_probe_max_travel": round(self._search_center_first_probe_max_travel, 5),
            "center_first_min_probe_time": round(self._search_center_first_min_probe_time, 2),
            "socket_depth_threshold": round(self._search_socket_depth_threshold, 5),
        }
        if probe_direction is not None:
            values.update(self._xyz_values("probe_direction", probe_direction))
        if wall_tangent is not None:
            values.update(self._xyz_values("wall_tangent", wall_tangent))
        if error:
            values["error"] = error
        return values

    def _search_probe_values(
        self,
        probe: Optional[SearchProbeResult] = None,
        step_index: Optional[int] = None,
        total_steps: Optional[int] = None,
        center_first: Optional[bool] = None,
        surface_xyz=None,
        pre_probe_xyz=None,
        probe_direction_xyz=None,
        contact_force_target: Optional[float] = None,
        probe_timeout: Optional[float] = None,
        probe_max_travel: Optional[float] = None,
        offset=None,
    ) -> Dict:
        values: Dict = {
            "socket_depth_threshold": round(self._search_socket_depth_threshold, 5),
            "force_abort_threshold": round(self._search_contact_force_abort_threshold, 3),
        }
        if step_index is not None:
            values["step"] = int(step_index)
        if total_steps is not None:
            values["total_steps"] = int(total_steps)
        if center_first is not None:
            values["center_first"] = bool(center_first)
        if surface_xyz is not None:
            values.update(self._xyz_values("surface", surface_xyz))
        if pre_probe_xyz is not None:
            values.update(self._xyz_values("pre_probe", pre_probe_xyz))
        if probe_direction_xyz is not None:
            values.update(self._xyz_values("probe_direction", probe_direction_xyz))
        if contact_force_target is not None:
            values["contact_force_target"] = round(contact_force_target, 3)
        if probe_timeout is not None:
            values["probe_timeout"] = round(probe_timeout, 2)
        if probe_max_travel is not None:
            values["probe_max_travel"] = round(probe_max_travel, 5)
        if offset is not None:
            values["offset_tool_x"] = round(float(getattr(offset, "dx", 0.0)), 5)
            values["offset_base_z"] = round(float(getattr(offset, "dy", 0.0)), 5)
        if probe is not None:
            values.update(
                {
                    "reason": probe.reason,
                    "candidate_found": bool(probe.candidate_found),
                    "inserted_depth": round(probe.inserted_depth, 5),
                    "contact_force": round(probe.contact_force, 3),
                    "max_contact_force": round(probe.max_contact_force, 3),
                    "travel_distance": round(probe.travel_distance, 5),
                }
            )
            if contact_force_target is not None:
                values["classification"] = self._classify_search_probe_result(
                    probe.reason,
                    probe.inserted_depth,
                    probe.contact_force,
                    contact_force_target,
                )
            if probe.final_pose is not None:
                values["final_pose"] = self._pose_values(probe.final_pose)
        return values

    def _publish_status(
        self,
        stage_id: str,
        status: str,
        success: Optional[bool] = None,
        message: str = "",
        values: Optional[Dict] = None,
    ) -> None:
        if self._status_callback is None:
            return
        try:
            self._status_callback(stage_id, status, success, message, values or {})
        except Exception as exc:
            rospy.logwarn(
                "[usb_c_insertion] event=insertion_status_publish_failed stage=%s error=%s",
                stage_id,
                exc,
            )

    def _contact_values(self, result: ContactApproachResult) -> Dict:
        values = {
            "reason": result.reason,
            "contact_force": round(result.contact_force, 3),
            "travel_distance": round(result.travel_distance, 5),
        }
        if result.contact_pose is not None:
            values["contact_pose"] = self._pose_values(result.contact_pose)
        if result.contact_point is not None:
            values.update(
                {
                    "contact_x": round(result.contact_point.point.x, 5),
                    "contact_y": round(result.contact_point.point.y, 5),
                    "contact_z": round(result.contact_point.point.z, 5),
                }
            )
        return values

    def _initial_check_values(self, result: InitialApproachCheckResult) -> Dict:
        values = {
            "accepted": bool(result.accepted),
            "terminal_failure": bool(result.terminal_failure),
            "reason": result.reason,
            "method": result.method,
            "counterforce_y": round(result.counterforce_y, 3),
            "counterforce_z": round(result.counterforce_z, 3),
        }
        if result.pull_result is not None:
            values.update(
                {
                    "pull_reason": result.pull_result.reason,
                    "pull_force": round(result.pull_result.pull_force, 3),
                    "pull_retraction": round(result.pull_result.retraction_distance, 5),
                }
            )
        return values

    def _search_values(self, result: SpiralSearchResult) -> Dict:
        progress = 0.0
        if result.total_steps > 0:
            progress = float(result.completed_steps) / float(result.total_steps)
        return {
            "reason": result.reason,
            "method": result.method,
            "completed_steps": int(result.completed_steps),
            "total_steps": int(result.total_steps),
            "progress": round(progress, 3),
            "inserted_depth": round(result.inserted_depth, 5),
            "contact_force": round(result.contact_force, 3),
            "force_insert_attempted": bool(self._force_insertion_attempted),
        }

    def _insert_values(self, result, source: str) -> Dict:
        return {
            "source": source,
            "reason": result.reason,
            "inserted_depth": round(result.inserted_depth, 5),
            "contact_force": round(result.contact_force, 3),
            "final_tool_pose": self._pose_values(self._tf.get_tool_pose_in_base()),
        }

    @staticmethod
    def _xyz_values(prefix: str, xyz) -> Dict:
        return {
            "%s_x" % prefix: round(float(xyz[0]), 5),
            "%s_y" % prefix: round(float(xyz[1]), 5),
            "%s_z" % prefix: round(float(xyz[2]), 5),
        }

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

    def _search_probe_speed_for_force(self, contact_force: float, contact_force_target: float) -> float:
        force_error = max(0.0, contact_force_target - contact_force)
        proportional_speed = self._search_probe_force_gain * force_error
        requested_speed = min(self._search_probe_speed_limit, proportional_speed)
        if self._search_probe_min_speed > 0.0 and force_error > 0.0:
            requested_speed = max(
                min(self._search_probe_min_speed, self._search_probe_speed_limit),
                requested_speed,
            )
        return max(0.0, min(self._search_probe_speed_limit, requested_speed))

    def _begin_direct_probe_control(self) -> None:
        self._robot.enable_pose_servo(False)
        self._micro_motion_active_publisher.publish(Bool(data=True))
        self._direct_probe_control_active = True
        rospy.sleep(0.01)

    def _end_direct_probe_control(self) -> None:
        self._publish_direct_zero_twist()
        self._micro_motion_active_publisher.publish(Bool(data=False))
        self._direct_probe_control_active = False

    def _publish_direct_twist(self, vx: float, vy: float, vz: float) -> None:
        twist = Twist()
        twist.linear.x = float(vx)
        twist.linear.y = float(vy)
        twist.linear.z = float(vz)
        self._direct_twist_publisher.publish(self._to_controller_frame(twist))

    def _publish_direct_zero_twist(self) -> None:
        zero_twist = self._to_controller_frame(Twist())
        for index in range(max(1, self._search_direct_stop_repeat_count)):
            self._direct_twist_publisher.publish(zero_twist)
            if self._search_direct_stop_interval > 0.0 and index + 1 < self._search_direct_stop_repeat_count:
                rospy.sleep(self._search_direct_stop_interval)

    def _move_to_pose(self, target_pose: PoseStamped, name: str, timeout: float) -> bool:
        self._robot.send_pose_target(
            target_pose.pose.position.x,
            target_pose.pose.position.y,
            target_pose.pose.position.z,
            qx=target_pose.pose.orientation.x,
            qy=target_pose.pose.orientation.y,
            qz=target_pose.pose.orientation.z,
            qw=target_pose.pose.orientation.w,
            frame_id=target_pose.header.frame_id or self._base_frame,
        )
        self._robot.enable_pose_servo(True)
        deadline = rospy.Time.now() + rospy.Duration.from_sec(max(0.1, timeout))
        rate = rospy.Rate(max(1.0, self._command_rate))
        while not rospy.is_shutdown():
            pose = self._tf.get_tool_pose_in_base()
            if pose is None:
                self._robot.stop_motion()
                rospy.logwarn("[usb_c_insertion] event=insertion_workflow_pose_move_failed name=%s reason=missing_tf", name)
                return False
            position_error = self._pose_distance(pose, target_pose)
            orientation_error = self._orientation_error(pose, target_pose)
            if position_error <= self._position_tolerance and orientation_error <= self._orientation_tolerance:
                self._robot.enable_pose_servo(False)
                rospy.loginfo(
                    "[usb_c_insertion] event=insertion_workflow_pose_move_complete name=%s position_error=%.5f orientation_error=%.5f",
                    name,
                    position_error,
                    orientation_error,
                )
                return True
            if rospy.Time.now() > deadline:
                self._robot.stop_motion()
                rospy.logwarn(
                    "[usb_c_insertion] event=insertion_workflow_pose_move_failed name=%s reason=timeout position_error=%.5f orientation_error=%.5f",
                    name,
                    position_error,
                    orientation_error,
                )
                return False
            rate.sleep()
        self._robot.stop_motion()
        return False

    def _wait_for_wrench(self) -> bool:
        deadline = rospy.Time.now() + rospy.Duration.from_sec(max(0.5, self._pipeline_wait_timeout))
        rate = rospy.Rate(20.0)
        while not rospy.is_shutdown():
            if not self._ft.is_wrench_stale():
                return True
            if rospy.Time.now() > deadline:
                return False
            rate.sleep()
        return False

    def _zero_ft_for_verification_stage(self, stage: str) -> bool:
        if stage.endswith("lateral") or "lateral" in stage:
            if not self._zero_ft_before_lateral_verify:
                return True
        elif not self._zero_ft_before_pull:
            return True

        self._robot.stop_motion()
        rospy.loginfo("[usb_c_insertion] event=insertion_workflow_zero_ft stage=%s", stage)
        if not self._ft.zero_sensor():
            rospy.logerr("[usb_c_insertion] event=insertion_workflow_zero_ft_failed stage=%s", stage)
            return False
        rospy.sleep(max(0.0, self._verification_zero_ft_settle_time))
        if not self._wait_for_wrench():
            rospy.logerr("[usb_c_insertion] event=insertion_workflow_zero_ft_failed stage=%s reason=wrench_unavailable", stage)
            return False
        return True

    @staticmethod
    def _tool_z_direction(pose: PoseStamped, sign: float) -> Tuple[float, float, float]:
        qx = pose.pose.orientation.x
        qy = pose.pose.orientation.y
        qz = pose.pose.orientation.z
        qw = pose.pose.orientation.w
        direction = rotate_vector_by_quaternion(0.0, 0.0, sign, qx, qy, qz, qw)
        return InsertionWorkflow._normalize_vector(direction)

    @staticmethod
    def _horizontal_tool_x_direction(pose: PoseStamped) -> Tuple[float, float, float]:
        qx = pose.pose.orientation.x
        qy = pose.pose.orientation.y
        qz = pose.pose.orientation.z
        qw = pose.pose.orientation.w
        tool_x = rotate_vector_by_quaternion(1.0, 0.0, 0.0, qx, qy, qz, qw)
        projected = (tool_x[0], tool_x[1], 0.0)
        try:
            return InsertionWorkflow._normalize_vector(projected)
        except ValueError:
            return InsertionWorkflow._normalize_vector(tool_x)

    def _get_pull_retention_force(self, pull_direction_xyz, wrench=None) -> float:
        wrench = wrench or self._ft.get_filtered_wrench()
        if self._pull_force_mode in ("tool_z", "tool", "sensor_z"):
            return max(0.0, -self._pull_tool_z_sign * wrench.force_z)
        if self._pull_force_mode in ("base_projection", "base", "projected"):
            return self._get_opposing_force_along_direction(pull_direction_xyz)
        rospy.logwarn_throttle(
            2.0,
            "[usb_c_insertion] event=insertion_pull_force_mode_unknown mode=%s fallback=tool_z",
            self._pull_force_mode,
        )
        wrench = self._ft.get_filtered_wrench()
        return max(0.0, -self._pull_tool_z_sign * wrench.force_z)

    def _get_opposing_force_along_direction(self, direction_xyz) -> float:
        direction = self._normalize_vector(direction_xyz)
        wrench = self._ft.get_filtered_wrench()
        projected_force = (
            wrench.force_x * direction[0]
            + wrench.force_y * direction[1]
            + wrench.force_z * direction[2]
        )
        return max(0.0, -projected_force)

    @staticmethod
    def _pose_xyz(pose: PoseStamped) -> Tuple[float, float, float]:
        return (
            pose.pose.position.x,
            pose.pose.position.y,
            pose.pose.position.z,
        )

    @staticmethod
    def _offset_xyz(xyz, direction_xyz, distance: float) -> Tuple[float, float, float]:
        return (
            xyz[0] + direction_xyz[0] * distance,
            xyz[1] + direction_xyz[1] * distance,
            xyz[2] + direction_xyz[2] * distance,
        )

    @staticmethod
    def _make_pose_like(reference_pose: PoseStamped, xyz) -> PoseStamped:
        pose = PoseStamped()
        pose.header.stamp = rospy.Time.now()
        pose.header.frame_id = reference_pose.header.frame_id
        pose.pose.position.x = float(xyz[0])
        pose.pose.position.y = float(xyz[1])
        pose.pose.position.z = float(xyz[2])
        pose.pose.orientation = reference_pose.pose.orientation
        return pose

    @staticmethod
    def _project_xyz_displacement(start_xyz, current_xyz, direction_xyz) -> float:
        delta = (
            current_xyz[0] - start_xyz[0],
            current_xyz[1] - start_xyz[1],
            current_xyz[2] - start_xyz[2],
        )
        return sum(delta[index] * direction_xyz[index] for index in range(3))

    @staticmethod
    def _project_pose_displacement(start_pose: PoseStamped, current_pose: PoseStamped, direction_xyz) -> float:
        delta = (
            current_pose.pose.position.x - start_pose.pose.position.x,
            current_pose.pose.position.y - start_pose.pose.position.y,
            current_pose.pose.position.z - start_pose.pose.position.z,
        )
        return sum(delta[index] * direction_xyz[index] for index in range(3))

    @staticmethod
    def _pose_distance(first: PoseStamped, second: PoseStamped) -> float:
        dx = second.pose.position.x - first.pose.position.x
        dy = second.pose.position.y - first.pose.position.y
        dz = second.pose.position.z - first.pose.position.z
        return math.sqrt(dx * dx + dy * dy + dz * dz)

    @staticmethod
    def _orientation_error(current_pose: PoseStamped, target_pose: PoseStamped) -> float:
        current = InsertionWorkflow._normalize_quaternion(
            (
                current_pose.pose.orientation.x,
                current_pose.pose.orientation.y,
                current_pose.pose.orientation.z,
                current_pose.pose.orientation.w,
            )
        )
        target = InsertionWorkflow._normalize_quaternion(
            (
                target_pose.pose.orientation.x,
                target_pose.pose.orientation.y,
                target_pose.pose.orientation.z,
                target_pose.pose.orientation.w,
            )
        )
        error = InsertionWorkflow._quaternion_multiply(target, InsertionWorkflow._quaternion_conjugate(current))
        if error[3] < 0.0:
            error = (-error[0], -error[1], -error[2], -error[3])
        vector_norm = math.sqrt(error[0] * error[0] + error[1] * error[1] + error[2] * error[2])
        if vector_norm <= 1e-9:
            return 0.0
        return 2.0 * math.atan2(vector_norm, error[3])

    @staticmethod
    def _pose_to_point_stamped(pose: PoseStamped) -> PointStamped:
        point = PointStamped()
        point.header = pose.header
        point.point = pose.pose.position
        return point

    @staticmethod
    def _normalize_vector(vector_xyz) -> Tuple[float, float, float]:
        norm = math.sqrt(sum(component * component for component in vector_xyz))
        if norm <= 1e-9:
            raise ValueError("direction must be non-zero")
        return tuple(component / norm for component in vector_xyz)

    @staticmethod
    def _normalize_quaternion(quaternion_xyzw):
        norm = math.sqrt(sum(component * component for component in quaternion_xyzw))
        if norm <= 1e-9:
            return (0.0, 0.0, 0.0, 1.0)
        return tuple(component / norm for component in quaternion_xyzw)

    @staticmethod
    def _quaternion_conjugate(quaternion_xyzw):
        return (-quaternion_xyzw[0], -quaternion_xyzw[1], -quaternion_xyzw[2], quaternion_xyzw[3])

    @staticmethod
    def _quaternion_multiply(first_xyzw, second_xyzw):
        x1, y1, z1, w1 = first_xyzw
        x2, y2, z2, w2 = second_xyzw
        return (
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        )

    @staticmethod
    def _sign(value: float) -> float:
        return 1.0 if float(value) >= 0.0 else -1.0

    @staticmethod
    def _to_controller_frame(twist: Twist) -> Twist:
        converted = Twist()
        converted.linear.x = -twist.linear.x
        converted.linear.y = -twist.linear.y
        converted.linear.z = twist.linear.z
        converted.angular.x = -twist.angular.x
        converted.angular.y = -twist.angular.y
        converted.angular.z = twist.angular.z
        return converted

    def _handle_shutdown(self) -> None:
        if self._direct_probe_control_active:
            self._end_direct_probe_control()
            return
        self._publish_direct_zero_twist()


def main() -> None:
    rospy.init_node("usb_c_insertion_insertion_workflow")
    success = InsertionWorkflow().run()
    if not success:
        sys.exit(1)


if __name__ == "__main__":
    main()
