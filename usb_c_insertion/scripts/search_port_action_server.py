#!/usr/bin/env python3

from __future__ import annotations

import math
import os
import sys
from typing import Optional, Tuple

import actionlib
from geometry_msgs.msg import PointStamped, PoseStamped, Twist
import rospy
from std_msgs.msg import Bool

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from contact_detector import ContactDetector
from ft_interface import FTInterface
from param_utils import required_bool_param, required_float_param, required_int_param, required_str_param
from prepose_planner import rotate_vector_by_quaternion
from robot_interface import RobotInterface
from search_pattern import generate_centered_raster_pattern, generate_preferred_square_spiral_pattern
from tf_interface import TFInterface
from usb_c_insertion.msg import (
    MicroMoveAction,
    MicroMoveGoal,
    MoveToPoseAction,
    MoveToPoseGoal,
    SearchPortAction,
    SearchPortFeedback,
    SearchPortResult,
    VerifyInsertionAction,
    VerifyInsertionGoal,
)


class SearchPortActionServer:
    """
    Search for the USB-C port around the estimated port pose.

    The action assumes the TCP orientation has already been corrected. It first
    touches the wall once to establish the local surface reference, then moves
    quickly between contact-free pre-probe positions and slowly probes along the
    tool-z direction at each search point.
    """

    def __init__(self):
        self._action_name = "search_port"
        self._move_action_name = "move_to_pose"
        self._micro_move_action_name = "micro_move"
        self._verify_insertion_action_name = required_str_param("~verify/action_name")
        self._base_frame = required_str_param("~frames/base_frame")
        self._twist_topic = required_str_param("~topics/twist_cmd")
        self._micro_motion_active_topic = required_str_param("~topics/micro_motion_active")

        self._search_pattern = required_str_param("~search/pattern").lower()
        self._search_step_x = required_float_param("~search/step_x")
        self._search_step_y = required_float_param("~search/step_y")
        self._search_width = required_float_param("~search/max_search_width")
        self._search_height = required_float_param("~search/max_search_height")
        self._search_timeout = required_float_param("~search/search_timeout")
        self._search_preferred_tool_x_sign = required_float_param("~search/preferred_tool_x_sign")
        self._search_preferred_z_sign = required_float_param("~search/preferred_z_sign")
        self._search_diagonal_first = required_bool_param("~search/diagonal_first")

        self._search_transfer_timeout = required_float_param("~search/transfer_timeout")
        self._search_transfer_max_velocity = required_float_param("~search/transfer_max_velocity")
        self._search_transfer_max_acceleration = required_float_param("~search/transfer_max_acceleration")
        self._search_transfer_max_jerk = required_float_param("~search/transfer_max_jerk")
        self._search_transfer_arc_enabled = required_bool_param("~search/transfer_arc_enabled")
        self._search_transfer_arc_height = max(0.0, required_float_param("~search/transfer_arc_height"))
        self._search_pre_probe_clearance = max(0.0, required_float_param("~search/pre_probe_clearance"))
        self._search_pre_probe_settle_time = max(0.0, required_float_param("~search/pre_probe_settle_time"))

        self._search_contact_force_target = required_float_param("~search/contact_force_target")
        self._search_initial_probe_speed = required_float_param("~search/initial_probe_speed")
        self._search_initial_probe_max_travel = required_float_param("~search/initial_probe_max_travel")
        self._search_initial_probe_timeout = required_float_param("~search/initial_probe_timeout")
        self._search_probe_speed = required_float_param("~search/probe_speed")
        self._search_probe_timeout = required_float_param("~search/probe_timeout")
        self._search_probe_max_travel = required_float_param("~search/probe_max_travel")
        self._search_socket_depth_threshold = required_float_param("~search/socket_depth_threshold")
        self._search_verify_initial_contact = required_bool_param("~search/verify_initial_contact")
        self._search_verification_required = required_bool_param("~search/verification_required")
        self._search_verify_timeout = required_float_param("~search/verify_timeout")
        self._search_direct_stop_repeat_count = required_int_param("~search/direct_stop_repeat_count")
        self._search_direct_stop_interval = max(0.0, required_float_param("~search/direct_stop_interval"))
        self._search_zero_ft_before_search = required_bool_param("~search/zero_ft_before_search")

        self._precontact_offset_tool_x = required_float_param("~state_machine/precontact_offset_tool_x")
        self._precontact_offset_tool_y = required_float_param("~state_machine/precontact_offset_tool_y")
        self._precontact_offset_tool_z = required_float_param("~state_machine/precontact_offset_tool_z")
        self._target_offset_tool_x = required_float_param("~state_machine/target_offset_tool_x")
        self._target_offset_tool_y = required_float_param("~state_machine/target_offset_tool_y")
        self._move_settle_time = required_float_param("~motion/action_settle_time")
        self._command_rate = required_float_param("~motion/command_rate")
        self._looming_tool_z_direction_sign = self._sign(required_float_param("~looming/tool_z_direction_sign"))
        self._enforce_precontact_standoff = required_bool_param("~state_machine/enforce_precontact_standoff")
        self._min_precontact_standoff = abs(required_float_param("~state_machine/min_precontact_standoff"))

        self._robot = RobotInterface()
        self._tf = TFInterface()
        self._ft = FTInterface(
            wrench_topic=required_str_param("~topics/wrench"),
            filter_window_size=required_int_param("~contact/baseline_window"),
            wrench_timeout=required_float_param("~contact/wrench_timeout"),
        )
        self._contact_detector = ContactDetector(
            self._ft,
            hysteresis=required_float_param("~contact/hysteresis"),
        )
        self._direct_twist_publisher = rospy.Publisher(self._twist_topic, Twist, queue_size=10)
        self._micro_motion_active_publisher = rospy.Publisher(
            self._micro_motion_active_topic,
            Bool,
            queue_size=1,
            latch=True,
        )
        self._direct_probe_control_active = False
        self._verify_insertion_available = False
        self._current_search_offset_x = 0.0
        self._current_search_offset_z = 0.0
        self._move_client = actionlib.SimpleActionClient(self._move_action_name, MoveToPoseAction)
        self._micro_move_client = actionlib.SimpleActionClient(self._micro_move_action_name, MicroMoveAction)
        self._verify_insertion_client = actionlib.SimpleActionClient(
            self._verify_insertion_action_name,
            VerifyInsertionAction,
        )
        self._server = actionlib.SimpleActionServer(
            self._action_name,
            SearchPortAction,
            execute_cb=self._execute,
            auto_start=False,
        )
        rospy.on_shutdown(self._handle_shutdown)
        self._server.start()
        rospy.loginfo("[usb_c_insertion] event=search_port_action_ready action=%s", self._action_name)

    def _execute(self, goal) -> None:
        timeout = self._goal_or_default(goal.timeout, self._search_timeout)
        started_at = rospy.Time.now()
        self._current_search_offset_x = 0.0
        self._current_search_offset_z = 0.0

        if not self._move_client.wait_for_server(rospy.Duration.from_sec(5.0)):
            self._abort("move_action_unavailable")
            return
        if not self._micro_move_client.wait_for_server(rospy.Duration.from_sec(5.0)):
            self._abort("micro_move_action_unavailable")
            return
        if not self._verify_insertion_ready():
            return

        if not self._validate_pose_frame(goal.port_pose, "unsupported_port_pose_frame"):
            return
        if not self._validate_pose_frame(goal.reference_pose, "unsupported_reference_pose_frame"):
            return

        reference_quaternion = (
            goal.reference_pose.pose.orientation.x,
            goal.reference_pose.pose.orientation.y,
            goal.reference_pose.pose.orientation.z,
            goal.reference_pose.pose.orientation.w,
        )

        try:
            probe_direction = self._tool_direction(reference_quaternion, (0.0, 0.0, 1.0))
            wall_tangent = self._horizontal_tool_x(reference_quaternion)
            standoff = self._precontact_standoff_distance()
            if not self._is_precontact_standoff_safe(standoff):
                return
            precontact_xyz = self._compute_precontact_xyz(goal.port_pose, reference_quaternion)
        except ValueError as exc:
            self._abort("search_geometry_failed: %s" % exc, "search_geometry_failed")
            return

        self._publish_feedback("move_to_port_precontact", started_at, 0, 0, None, probe_direction)
        rospy.loginfo(
            "[usb_c_insertion] event=search_precontact_reference "
            "precontact_offset_tool_z=%.4f standoff=%.4f port_xyz=(%.4f,%.4f,%.4f) "
            "precontact_xyz=(%.4f,%.4f,%.4f) probe_direction=(%.4f,%.4f,%.4f)",
            self._precontact_offset_tool_z,
            standoff,
            goal.port_pose.pose.position.x,
            goal.port_pose.pose.position.y,
            goal.port_pose.pose.position.z,
            precontact_xyz[0],
            precontact_xyz[1],
            precontact_xyz[2],
            probe_direction[0],
            probe_direction[1],
            probe_direction[2],
        )
        move_success, move_error_code = self._move_to_pose(
            precontact_xyz,
            reference_quaternion,
            min(timeout, self._search_timeout),
        )
        if not move_success:
            self._abort("move_to_port_precontact_failed", move_error_code)
            return

        if self._search_zero_ft_before_search:
            self._publish_feedback("zero_ft_before_search", started_at, 0, 0, None, probe_direction)
            if not self._ft.zero_sensor():
                self._abort("zero_ft_before_search_failed")
                return
            if not self._sleep_interruptible(0.2):
                return

        self._publish_feedback("initial_surface_probe", started_at, 0, 0, None, probe_direction)
        initial_success, initial_reason, reference_point, initial_force = self._probe_until_contact(
            probe_direction,
            speed=self._search_initial_probe_speed,
            max_travel=max(self._search_initial_probe_max_travel, self._search_pre_probe_clearance),
            timeout=min(self._search_initial_probe_timeout, timeout),
            probe_label="initial",
            step_index=0,
            total_steps=0,
        )
        if not initial_success or reference_point is None:
            self._abort("initial_surface_probe_failed", initial_reason, initial_reason)
            return

        wall_contact_point = self._make_point_stamped(reference_point)
        self._publish_feedback("initial_surface_reference_ready", started_at, 0, 0, reference_point, probe_direction)
        initial_verification_ok, initial_verified, initial_verify_result = self._verify_current_insertion(
            stage="initial_contact_verification",
            started_at=started_at,
            step_index=0,
            total_steps=0,
            reference_point=reference_point,
            probe_direction=probe_direction,
            enabled=self._search_verify_initial_contact,
            accept_if_disabled=False,
        )
        if not initial_verification_ok:
            return
        if initial_verified:
            rospy.loginfo(
                "[usb_c_insertion] event=initial_contact_verified_port counterforce_y=%.3f counterforce_z=%.3f",
                float(initial_verify_result.counterforce_y) if initial_verify_result is not None else 0.0,
                float(initial_verify_result.counterforce_z) if initial_verify_result is not None else 0.0,
            )
            result = self._make_result(True, True, "initial_contact_verified_port", "", "")
            result.wall_contact_point = wall_contact_point
            result.found_pose = self._current_pose_or_empty()
            if initial_verify_result is not None:
                result.contact_force = float(
                    max(initial_verify_result.counterforce_y, initial_verify_result.counterforce_z)
                )
            else:
                result.contact_force = float(initial_force)
            result.completed_steps = 0
            self._server.set_succeeded(result)
            return
        rospy.loginfo("[usb_c_insertion] event=initial_contact_not_verified continuing_search=true")

        try:
            pattern = self._generate_search_pattern()
        except ValueError as exc:
            self._abort("search_pattern_invalid: %s" % exc, "search_pattern_invalid")
            return
        if pattern:
            rospy.loginfo(
                "[usb_c_insertion] event=search_pattern_ready pattern=%s steps=%d first_dx=%.4f first_dz=%.4f preferred_tool_x_sign=%.1f preferred_z_sign=%.1f diagonal_first=%s",
                self._search_pattern,
                len(pattern),
                pattern[0].dx,
                pattern[0].dy,
                self._search_preferred_tool_x_sign,
                self._search_preferred_z_sign,
                str(self._search_diagonal_first).lower(),
            )

        current_pose = self._tf.get_tool_pose_in_base()
        if current_pose is None:
            self._abort("missing_tf_before_search")
            return

        total_steps = len(pattern)
        surface_point = reference_point
        search_offset_x = 0.0
        search_offset_z = 0.0

        for step_index, offset in enumerate(pattern, start=1):
            if self._server.is_preempt_requested():
                self._robot.stop_motion()
                self._micro_move_client.cancel_goal()
                self._server.set_preempted(
                    self._make_result(False, False, "preempted", "preempted", "preempted")
                )
                return
            if rospy.is_shutdown():
                self._abort("shutdown_during_search")
                return
            if (rospy.Time.now() - started_at).to_sec() > timeout:
                self._abort("search_timeout")
                return

            search_offset_x += offset.dx
            search_offset_z += offset.dy
            self._current_search_offset_x = search_offset_x
            self._current_search_offset_z = search_offset_z
            surface_point = (
                surface_point[0] + offset.dx * wall_tangent[0],
                surface_point[1] + offset.dx * wall_tangent[1],
                surface_point[2] + offset.dy,
            )
            pre_probe_xyz = self._build_pre_probe_xyz(surface_point, probe_direction)

            self._publish_feedback(
                "search_transfer_to_pre_probe",
                started_at,
                step_index,
                total_steps,
                surface_point,
                probe_direction,
            )
            move_success, move_error_code = self._move_micro_to_xyz(pre_probe_xyz, probe_direction)
            if not move_success:
                self._abort("search_transfer_to_pre_probe_failed", move_error_code)
                return

            self._publish_feedback(
                "search_pre_probe_settle",
                started_at,
                step_index,
                total_steps,
                surface_point,
                probe_direction,
            )
            if not self._sleep_interruptible(self._search_pre_probe_settle_time):
                return

            self._publish_feedback(
                "search_probe",
                started_at,
                step_index,
                total_steps,
                surface_point,
                probe_direction,
            )
            probe_success, probe_reason, inserted_depth, contact_force = self._probe_search_position(
                surface_point,
                probe_direction,
                "search",
                step_index,
                total_steps,
            )
            if not probe_success:
                self._abort("search_probe_failed", probe_reason, probe_reason)
                return

            self._publish_feedback(
                "search_probe_complete",
                started_at,
                step_index,
                total_steps,
                surface_point,
                probe_direction,
            )
            if probe_reason == "socket_depth_reached":
                self._publish_feedback(
                    "search_candidate_found",
                    started_at,
                    step_index,
                    total_steps,
                    surface_point,
                    probe_direction,
                )
                rospy.loginfo(
                    "[usb_c_insertion] event=search_candidate_found step=%d total=%d inserted_depth=%.4f threshold=%.4f contact_force=%.3f lateral_verification=deferred_until_after_insert",
                    int(step_index),
                    int(total_steps),
                    float(inserted_depth),
                    self._search_socket_depth_threshold,
                    float(contact_force),
                )
                result = self._make_result(True, True, "potential_port_found", "", "")
                result.wall_contact_point = wall_contact_point
                result.found_pose = self._current_pose_or_empty()
                result.inserted_depth = float(inserted_depth)
                result.contact_force = float(contact_force)
                result.completed_steps = step_index
                self._server.set_succeeded(result)
                return

        result = self._make_result(False, False, "search_pattern_exhausted", "search_pattern_exhausted", "")
        result.wall_contact_point = wall_contact_point
        result.found_pose = self._current_pose_or_empty()
        result.inserted_depth = float(self._compute_inserted_depth(surface_point, probe_direction))
        result.contact_force = float(self._get_search_contact_force(probe_direction))
        result.completed_steps = total_steps
        self._robot.stop_motion()
        self._server.set_aborted(result)

    def _verify_insertion_ready(self) -> bool:
        if not self._search_verify_initial_contact:
            return True

        self._verify_insertion_available = self._verify_insertion_client.wait_for_server(
            rospy.Duration.from_sec(5.0)
        )
        if self._verify_insertion_available:
            return True

        message = "verify_insertion_action_unavailable"
        if self._search_verification_required:
            self._abort(message)
            return False

        rospy.logwarn("[usb_c_insertion] event=%s verification_required=false", message)
        return True

    def _verify_current_insertion(
        self,
        stage: str,
        started_at: rospy.Time,
        step_index: int,
        total_steps: int,
        reference_point,
        probe_direction,
        enabled: bool,
        accept_if_disabled: bool,
    ):
        if not enabled:
            return True, bool(accept_if_disabled), None
        if not self._verify_insertion_available:
            if self._search_verification_required:
                self._abort("verify_insertion_action_unavailable")
                return False, False, None
            rospy.logwarn("[usb_c_insertion] event=verify_insertion_skipped reason=action_unavailable")
            return True, bool(accept_if_disabled), None

        self._publish_feedback(stage, started_at, step_index, total_steps, reference_point, probe_direction)

        goal = VerifyInsertionGoal()
        goal.timeout = float(self._search_verify_timeout)
        goal.zero_ft_before_verify = False
        self._verify_insertion_client.send_goal(goal)
        finished = self._verify_insertion_client.wait_for_result(
            rospy.Duration.from_sec(max(1.0, self._search_verify_timeout * 2.0 + 1.0))
        )
        if not finished:
            self._verify_insertion_client.cancel_goal()
            if self._search_verification_required:
                self._abort("verify_insertion_action_timeout")
                return False, False, None
            rospy.logwarn("[usb_c_insertion] event=verify_insertion_skipped reason=action_timeout")
            return True, bool(accept_if_disabled), None

        result = self._verify_insertion_client.get_result()
        if result is None or not bool(result.success):
            reason = result.failure_reason if result is not None else "verify_insertion_no_result"
            if self._search_verification_required:
                self._abort(reason)
                return False, False, None
            rospy.loginfo(
                "[usb_c_insertion] event=verify_insertion_result stage=%s verified=false reason=%s",
                stage,
                reason,
            )
            return True, False, result

        rospy.loginfo(
            "[usb_c_insertion] event=verify_insertion_result stage=%s verified=%s counterforce_y=%.3f counterforce_z=%.3f reason=%s",
            stage,
            str(bool(result.verified)).lower(),
            float(result.counterforce_y),
            float(result.counterforce_z),
            result.failure_reason or result.message,
        )
        return True, bool(result.verified), result

    def _validate_pose_frame(self, pose: PoseStamped, error_code: str) -> bool:
        frame_id = pose.header.frame_id.strip() or self._base_frame
        if frame_id == self._base_frame:
            return True
        self._abort(error_code)
        return False

    def _compute_precontact_xyz(self, port_pose: PoseStamped, tool_quaternion) -> Tuple[float, float, float]:
        offset = rotate_vector_by_quaternion(
            self._target_offset_tool_x + self._precontact_offset_tool_x,
            self._target_offset_tool_y + self._precontact_offset_tool_y,
            self._precontact_offset_tool_z,
            tool_quaternion[0],
            tool_quaternion[1],
            tool_quaternion[2],
            tool_quaternion[3],
        )
        return (
            port_pose.pose.position.x + offset[0],
            port_pose.pose.position.y + offset[1],
            port_pose.pose.position.z + offset[2],
        )

    def _precontact_standoff_distance(self) -> float:
        return -self._precontact_offset_tool_z * self._looming_tool_z_direction_sign

    def _is_precontact_standoff_safe(self, standoff: float) -> bool:
        if not self._enforce_precontact_standoff:
            return True
        if standoff >= self._min_precontact_standoff:
            return True
        self._abort(
            "unsafe_precontact_offset: precontact_offset_tool_z=%.4f standoff=%.4f min_standoff=%.4f"
            % (self._precontact_offset_tool_z, standoff, self._min_precontact_standoff),
            "unsafe_precontact_offset",
        )
        return False

    def _generate_search_pattern(self):
        if self._search_pattern == "spiral":
            return generate_preferred_square_spiral_pattern(
                step_x=self._search_step_x,
                step_y=self._search_step_y,
                width=self._search_width,
                height=self._search_height,
                preferred_x_sign=self._search_preferred_tool_x_sign,
                preferred_y_sign=self._search_preferred_z_sign,
            )
        if self._search_pattern == "raster":
            return generate_centered_raster_pattern(
                step_x=self._search_step_x,
                step_y=self._search_step_y,
                width=self._search_width,
                height=self._search_height,
                preferred_x_sign=self._search_preferred_tool_x_sign,
                preferred_y_sign=self._search_preferred_z_sign,
                diagonal_first=self._search_diagonal_first,
            )
        raise ValueError("unsupported search pattern '%s'" % self._search_pattern)

    def _move_to_pose(self, target_xyz, target_quaternion, timeout: float) -> tuple[bool, str]:
        goal = MoveToPoseGoal()
        goal.target_pose = PoseStamped()
        goal.target_pose.header.stamp = rospy.Time.now()
        goal.target_pose.header.frame_id = self._base_frame
        goal.target_pose.pose.position.x = target_xyz[0]
        goal.target_pose.pose.position.y = target_xyz[1]
        goal.target_pose.pose.position.z = target_xyz[2]
        goal.target_pose.pose.orientation.x = target_quaternion[0]
        goal.target_pose.pose.orientation.y = target_quaternion[1]
        goal.target_pose.pose.orientation.z = target_quaternion[2]
        goal.target_pose.pose.orientation.w = target_quaternion[3]
        goal.settle_time = self._move_settle_time
        goal.timeout = timeout

        self._move_client.send_goal(goal)
        finished = self._move_client.wait_for_result(rospy.Duration.from_sec(max(1.0, timeout + 2.0)))
        if not finished:
            self._move_client.cancel_goal()
            return False, "move_to_pose_wait_timeout"

        result = self._move_client.get_result()
        if result is None:
            return False, "move_to_pose_no_result"
        if not result.success:
            return False, result.error_code or result.message
        return True, ""

    def _build_pre_probe_xyz(self, surface_xyz, probe_direction) -> Tuple[float, float, float]:
        direction = self._normalize_vector(probe_direction)
        return (
            surface_xyz[0] - direction[0] * self._search_pre_probe_clearance,
            surface_xyz[1] - direction[1] * self._search_pre_probe_clearance,
            surface_xyz[2] - direction[2] * self._search_pre_probe_clearance,
        )

    def _move_micro_to_xyz(self, target_xyz, probe_direction) -> tuple[bool, str]:
        current_pose = self._tf.get_tool_pose_in_base()
        if current_pose is None:
            return False, "missing_tf"

        current_xyz = (
            current_pose.pose.position.x,
            current_pose.pose.position.y,
            current_pose.pose.position.z,
        )
        displacement = (
            target_xyz[0] - current_xyz[0],
            target_xyz[1] - current_xyz[1],
            target_xyz[2] - current_xyz[2],
        )
        arc_direction = None
        arc_height = 0.0
        if self._search_transfer_arc_enabled and self._search_transfer_arc_height > 0.0:
            direction = self._normalize_vector(probe_direction)
            arc_direction = (-direction[0], -direction[1], -direction[2])
            arc_height = self._search_transfer_arc_height
        return self._move_micro(
            displacement,
            self._search_transfer_timeout,
            arc_direction=arc_direction,
            arc_height=arc_height,
        )

    def _move_micro(self, displacement_xyz, timeout: float, arc_direction=None, arc_height: float = 0.0) -> tuple[bool, str]:
        goal = MicroMoveGoal()
        goal.displacement.x = float(displacement_xyz[0])
        goal.displacement.y = float(displacement_xyz[1])
        goal.displacement.z = float(displacement_xyz[2])
        goal.max_velocity = float(self._search_transfer_max_velocity)
        goal.max_acceleration = float(self._search_transfer_max_acceleration)
        goal.max_jerk = float(self._search_transfer_max_jerk)
        if arc_direction is not None:
            goal.arc_direction.x = float(arc_direction[0])
            goal.arc_direction.y = float(arc_direction[1])
            goal.arc_direction.z = float(arc_direction[2])
        goal.arc_height = float(arc_height)
        goal.timeout = float(timeout)
        goal.monitor_tf = True

        self._micro_move_client.send_goal(goal)
        wait_timeout = max(1.0, float(timeout) + 0.5)
        finished = self._micro_move_client.wait_for_result(rospy.Duration.from_sec(wait_timeout))
        if not finished:
            self._micro_move_client.cancel_goal()
            return False, "micro_move_wait_timeout"

        result = self._micro_move_client.get_result()
        if result is None:
            return False, "micro_move_no_result"
        if not result.success:
            return False, result.error_code or result.message
        return True, ""

    def _sleep_interruptible(self, duration: float) -> bool:
        deadline = rospy.Time.now() + rospy.Duration.from_sec(max(0.0, float(duration)))
        rate = rospy.Rate(100.0)
        while not rospy.is_shutdown() and rospy.Time.now() < deadline:
            if self._server.is_preempt_requested():
                self._robot.stop_motion()
                self._micro_move_client.cancel_goal()
                self._server.set_preempted(
                    self._make_result(False, False, "preempted", "preempted", "preempted")
                )
                return False
            rate.sleep()
        if rospy.is_shutdown():
            self._abort("shutdown_during_search_wait")
            return False
        return True

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

    def _probe_until_contact(
        self,
        direction_xyz,
        speed: float,
        max_travel: float,
        timeout: float,
        probe_label: str,
        step_index: int,
        total_steps: int,
    ) -> tuple[bool, str, Optional[Tuple[float, float, float]], float]:
        deadline = rospy.Time.now() + rospy.Duration.from_sec(max(0.1, timeout))
        rate = rospy.Rate(max(1.0, self._command_rate))
        direction = self._normalize_vector(direction_xyz)
        start_pose = self._tf.get_tool_pose_in_base()
        commanded_speed = max(0.0, float(speed))
        if start_pose is None:
            self._robot.stop_motion()
            return False, "missing_initial_tf", None, self._get_search_contact_force(direction)
        if commanded_speed <= 0.0:
            self._robot.stop_motion()
            return False, "invalid_probe_speed", None, self._get_search_contact_force(direction)

        start_xyz = (
            start_pose.pose.position.x,
            start_pose.pose.position.y,
            start_pose.pose.position.z,
        )
        max_contact_force = self._get_search_contact_force(direction)

        self._begin_direct_probe_control()
        try:
            while not rospy.is_shutdown():
                if self._server.is_preempt_requested():
                    self._publish_direct_zero_twist()
                    return False, "preempted", None, max_contact_force
                if rospy.Time.now() > deadline:
                    self._publish_direct_zero_twist()
                    return False, "initial_probe_timeout", None, max_contact_force
                if self._ft.is_wrench_stale():
                    self._publish_direct_zero_twist()
                    return False, "stale_wrench", None, max_contact_force

                pose = self._tf.get_tool_pose_in_base()
                if pose is None:
                    self._publish_direct_zero_twist()
                    return False, "missing_tf", None, max_contact_force

                current_xyz = (
                    pose.pose.position.x,
                    pose.pose.position.y,
                    pose.pose.position.z,
                )
                contact_force = self._get_search_contact_force(direction)
                max_contact_force = max(max_contact_force, contact_force)
                traveled = self._project_displacement(start_xyz, direction, current_xyz)

                if contact_force >= self._search_contact_force_target:
                    self._publish_direct_zero_twist()
                    self._log_search_probe_result(
                        probe_label,
                        step_index,
                        total_steps,
                        True,
                        "wall_contact_reached",
                        0.0,
                        traveled,
                        contact_force,
                        max_contact_force,
                        current_xyz,
                    )
                    return True, "wall_contact_reached", current_xyz, contact_force

                if traveled >= max_travel:
                    self._publish_direct_zero_twist()
                    self._log_search_probe_result(
                        probe_label,
                        step_index,
                        total_steps,
                        False,
                        "initial_probe_max_travel_reached",
                        0.0,
                        traveled,
                        contact_force,
                        max_contact_force,
                        current_xyz,
                    )
                    return False, "initial_probe_max_travel_reached", current_xyz, contact_force

                self._publish_direct_twist(
                    direction[0] * commanded_speed,
                    direction[1] * commanded_speed,
                    direction[2] * commanded_speed,
                )
                rate.sleep()

            self._publish_direct_zero_twist()
            return False, "shutdown", None, max_contact_force
        finally:
            self._end_direct_probe_control()

    def _probe_search_position(
        self,
        reference_point,
        direction_xyz,
        probe_label: str,
        step_index: int,
        total_steps: int,
    ) -> tuple[bool, str, float, float]:
        deadline = rospy.Time.now() + rospy.Duration.from_sec(max(0.1, self._search_probe_timeout))
        rate = rospy.Rate(max(1.0, self._command_rate))
        direction = self._normalize_vector(direction_xyz)
        start_pose = self._tf.get_tool_pose_in_base()
        if start_pose is None:
            self._robot.stop_motion()
            self._log_search_probe_result(
                probe_label,
                step_index,
                total_steps,
                False,
                "missing_initial_tf",
                0.0,
                0.0,
                self._get_search_contact_force(direction_xyz),
                self._get_search_contact_force(direction_xyz),
                None,
            )
            return False, "missing_initial_tf", 0.0, self._get_search_contact_force(direction_xyz)

        start_xyz = (
            start_pose.pose.position.x,
            start_pose.pose.position.y,
            start_pose.pose.position.z,
        )
        max_contact_force = self._get_search_contact_force(direction)

        self._begin_direct_probe_control()
        try:
            while not rospy.is_shutdown():
                if self._server.is_preempt_requested():
                    self._publish_direct_zero_twist()
                    return self._finish_search_probe(probe_label, step_index, total_steps, False, "preempted", reference_point, direction, start_xyz, max_contact_force)
                if rospy.Time.now() > deadline:
                    self._publish_direct_zero_twist()
                    return self._finish_search_probe(probe_label, step_index, total_steps, False, "search_probe_timeout", reference_point, direction, start_xyz, max_contact_force)
                if self._ft.is_wrench_stale():
                    self._publish_direct_zero_twist()
                    return self._finish_search_probe(probe_label, step_index, total_steps, False, "stale_wrench", reference_point, direction, start_xyz, max_contact_force)
                pose = self._tf.get_tool_pose_in_base()
                if pose is None:
                    self._publish_direct_zero_twist()
                    return self._finish_search_probe(probe_label, step_index, total_steps, False, "missing_tf", reference_point, direction, start_xyz, max_contact_force)

                contact_force = self._get_search_contact_force(direction)
                max_contact_force = max(max_contact_force, contact_force)
                inserted_depth = self._compute_inserted_depth(reference_point, direction)
                if contact_force >= self._search_contact_force_target:
                    self._publish_direct_zero_twist()
                    return self._finish_search_probe(probe_label, step_index, total_steps, True, "wall_contact_reached", reference_point, direction, start_xyz, max_contact_force)

                if inserted_depth >= self._search_socket_depth_threshold:
                    self._publish_direct_zero_twist()
                    return self._finish_search_probe(probe_label, step_index, total_steps, True, "socket_depth_reached", reference_point, direction, start_xyz, max_contact_force)

                probe_travel = self._project_displacement(
                    start_xyz,
                    direction,
                    (
                        pose.pose.position.x,
                        pose.pose.position.y,
                        pose.pose.position.z,
                    ),
                )
                if probe_travel >= self._search_probe_max_travel:
                    self._publish_direct_zero_twist()
                    return self._finish_search_probe(
                        probe_label,
                        step_index,
                        total_steps,
                        False,
                        "search_probe_max_travel_reached",
                        reference_point,
                        direction,
                        start_xyz,
                        max_contact_force,
                    )

                self._publish_direct_twist(
                    direction[0] * self._search_probe_speed,
                    direction[1] * self._search_probe_speed,
                    direction[2] * self._search_probe_speed,
                )
                rate.sleep()

            self._publish_direct_zero_twist()
            return self._finish_search_probe(probe_label, step_index, total_steps, False, "shutdown", reference_point, direction, start_xyz, max_contact_force)
        finally:
            self._end_direct_probe_control()

    def _finish_search_probe(
        self,
        probe_label: str,
        step_index: int,
        total_steps: int,
        success: bool,
        reason: str,
        reference_point,
        direction,
        start_xyz,
        max_contact_force: float,
    ) -> tuple[bool, str, float, float]:
        final_pose = self._tf.get_tool_pose_in_base()
        contact_force = self._get_search_contact_force(direction)
        inserted_depth = self._compute_inserted_depth(reference_point, direction)
        travel_distance = 0.0
        final_xyz = None
        if final_pose is not None:
            final_xyz = (
                final_pose.pose.position.x,
                final_pose.pose.position.y,
                final_pose.pose.position.z,
            )
            travel_distance = self._distance(start_xyz, final_xyz)

        self._log_search_probe_result(
            probe_label,
            step_index,
            total_steps,
            success,
            reason,
            inserted_depth,
            travel_distance,
            contact_force,
            max(max_contact_force, contact_force),
            final_xyz,
        )
        return success, reason, inserted_depth, contact_force

    def _log_search_probe_result(
        self,
        probe_label: str,
        step_index: int,
        total_steps: int,
        success: bool,
        reason: str,
        inserted_depth: float,
        travel_distance: float,
        contact_force: float,
        max_contact_force: float,
        final_xyz,
    ) -> None:
        classification = self._classify_search_probe_result(reason, inserted_depth, contact_force)
        if final_xyz is None:
            rospy.loginfo(
                "[usb_c_insertion] event=search_probe_result label=%s step=%d total=%d classification=%s success=%s reason=%s search_offset_x=%.4f search_offset_z=%.4f inserted_depth=%.4f socket_depth_threshold=%.4f travel_distance=%.4f contact_force=%.3f contact_force_target=%.3f max_contact_force=%.3f final_pose=missing_tf",
                probe_label,
                int(step_index),
                int(total_steps),
                classification,
                str(bool(success)).lower(),
                reason,
                self._current_search_offset_x,
                self._current_search_offset_z,
                inserted_depth,
                self._search_socket_depth_threshold,
                travel_distance,
                contact_force,
                self._search_contact_force_target,
                max_contact_force,
            )
            return

        rospy.loginfo(
            "[usb_c_insertion] event=search_probe_result label=%s step=%d total=%d classification=%s success=%s reason=%s search_offset_x=%.4f search_offset_z=%.4f inserted_depth=%.4f socket_depth_threshold=%.4f travel_distance=%.4f contact_force=%.3f contact_force_target=%.3f max_contact_force=%.3f final_x=%.4f final_y=%.4f final_z=%.4f",
            probe_label,
            int(step_index),
            int(total_steps),
            classification,
            str(bool(success)).lower(),
            reason,
            self._current_search_offset_x,
            self._current_search_offset_z,
            inserted_depth,
            self._search_socket_depth_threshold,
            travel_distance,
            contact_force,
            self._search_contact_force_target,
            max_contact_force,
            final_xyz[0],
            final_xyz[1],
            final_xyz[2],
        )

    def _classify_search_probe_result(self, reason: str, inserted_depth: float, contact_force: float) -> str:
        if reason == "socket_depth_reached" or inserted_depth >= self._search_socket_depth_threshold:
            return "candidate"
        if reason == "wall_contact_reached" or contact_force >= self._search_contact_force_target:
            return "wall_contact"
        if reason == "search_probe_max_travel_reached":
            return "no_contact"
        return "incomplete"

    def _publish_feedback(
        self,
        stage: str,
        started_at: rospy.Time,
        current_step: int,
        total_steps: int,
        reference_point,
        probe_direction,
    ) -> None:
        feedback = SearchPortFeedback()
        feedback.stage = stage
        feedback.current_step = int(current_step)
        feedback.total_steps = int(total_steps)
        feedback.elapsed = float((rospy.Time.now() - started_at).to_sec())
        if reference_point is not None and probe_direction is not None:
            feedback.contact_force = float(self._get_search_contact_force(probe_direction))
            feedback.inserted_depth = float(self._compute_inserted_depth(reference_point, probe_direction))
        else:
            feedback.contact_force = float(self._get_search_contact_force())
        self._server.publish_feedback(feedback)

    def _abort(
        self,
        message: str,
        error_code: Optional[str] = None,
        failure_reason: Optional[str] = None,
    ) -> None:
        if self._direct_probe_control_active:
            self._end_direct_probe_control()
        self._robot.stop_motion()
        self._server.set_aborted(
            self._make_result(False, False, message, error_code or message, failure_reason or "")
        )

    def _handle_shutdown(self) -> None:
        if self._direct_probe_control_active:
            self._end_direct_probe_control()
        else:
            self._publish_direct_zero_twist()

    def _current_pose_or_empty(self) -> PoseStamped:
        pose = self._tf.get_tool_pose_in_base()
        if pose is not None:
            return pose
        empty = PoseStamped()
        empty.header.stamp = rospy.Time.now()
        empty.header.frame_id = self._base_frame
        return empty

    def _make_point_stamped(self, xyz) -> PointStamped:
        point = PointStamped()
        point.header.stamp = rospy.Time.now()
        point.header.frame_id = self._base_frame
        point.point.x = float(xyz[0])
        point.point.y = float(xyz[1])
        point.point.z = float(xyz[2])
        return point

    def _compute_inserted_depth(self, reference_point, direction_xyz) -> float:
        pose = self._tf.get_tool_pose_in_base()
        if pose is None:
            return 0.0
        direction = self._normalize_vector(direction_xyz)
        displacement = (
            pose.pose.position.x - reference_point[0],
            pose.pose.position.y - reference_point[1],
            pose.pose.position.z - reference_point[2],
        )
        return sum(displacement[index] * direction[index] for index in range(3))

    def _get_search_contact_force(self, direction_xyz=None) -> float:
        if direction_xyz is not None:
            try:
                return self._contact_detector.get_contact_force_along_direction(direction_xyz)
            except ValueError:
                pass
        wrench = self._ft.get_filtered_wrench()
        return max(0.0, -wrench.force_z)

    def _tool_direction(self, tool_quaternion, tool_vector) -> Tuple[float, float, float]:
        direction = rotate_vector_by_quaternion(
            tool_vector[0],
            tool_vector[1],
            tool_vector[2],
            tool_quaternion[0],
            tool_quaternion[1],
            tool_quaternion[2],
            tool_quaternion[3],
        )
        return self._normalize_vector(direction)

    def _horizontal_tool_x(self, tool_quaternion) -> Tuple[float, float, float]:
        tool_x = self._tool_direction(tool_quaternion, (1.0, 0.0, 0.0))
        return self._normalize_vector((tool_x[0], tool_x[1], 0.0))

    @staticmethod
    def _make_result(
        success: bool,
        port_found: bool,
        message: str,
        error_code: str,
        failure_reason: str,
    ) -> SearchPortResult:
        result = SearchPortResult()
        result.success = bool(success)
        result.port_found = bool(port_found)
        result.message = str(message)
        result.error_code = str(error_code)
        result.failure_reason = str(failure_reason)
        result.inserted_depth = 0.0
        result.contact_force = 0.0
        result.completed_steps = 0
        return result

    @staticmethod
    def _goal_or_default(value: float, default: float) -> float:
        return float(value) if float(value) > 0.0 else float(default)

    @staticmethod
    def _sign(value) -> float:
        return 1.0 if float(value) >= 0.0 else -1.0

    @staticmethod
    def _normalize_vector(direction_xyz) -> Tuple[float, float, float]:
        magnitude = math.sqrt(sum(component * component for component in direction_xyz))
        if magnitude <= 1e-9:
            raise ValueError("direction_xyz must be non-zero")
        return tuple(component / magnitude for component in direction_xyz)

    @staticmethod
    def _distance(point_a, point_b) -> float:
        return math.sqrt(
            (point_b[0] - point_a[0]) ** 2
            + (point_b[1] - point_a[1]) ** 2
            + (point_b[2] - point_a[2]) ** 2
        )

    @staticmethod
    def _project_displacement(start_xyz, direction_xyz, current_xyz) -> float:
        delta = (
            current_xyz[0] - start_xyz[0],
            current_xyz[1] - start_xyz[1],
            current_xyz[2] - start_xyz[2],
        )
        return sum(delta[index] * direction_xyz[index] for index in range(3))

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


def main() -> None:
    rospy.init_node("usb_c_insertion_search_port_action_server")
    SearchPortActionServer()
    rospy.spin()


if __name__ == "__main__":
    main()
