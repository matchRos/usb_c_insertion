#!/usr/bin/env python3

from __future__ import annotations

import math
import os
import sys
from typing import Optional, Tuple

import actionlib
from geometry_msgs.msg import PointStamped, PoseStamped
import rospy

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from contact_detector import ContactDetector
from ft_interface import FTInterface
from prepose_planner import rotate_vector_by_quaternion
from robot_interface import RobotInterface
from search_pattern import generate_centered_raster_pattern
from tf_interface import TFInterface
from usb_c_insertion.msg import (
    MoveToPoseAction,
    MoveToPoseGoal,
    SearchPortAction,
    SearchPortFeedback,
    SearchPortResult,
)
from wall_probe import WallProbe


class SearchPortActionServer:
    """
    Search for the USB-C port around the estimated port pose.

    The action assumes the TCP orientation has already been corrected. It uses
    the supplied reference pose orientation, moves to a precontact pose around
    the estimated port position, probes the wall, then executes a conservative
    raster in the wall plane while maintaining light contact.
    """

    def __init__(self):
        self._action_name = str(rospy.get_param("~action_name", "search_port")).strip()
        self._move_action_name = str(rospy.get_param("~move_action_name", "move_to_pose")).strip()
        self._base_frame = str(rospy.get_param("~frames/base_frame", "base_link")).strip()

        self._search_step_y = float(rospy.get_param("~search/step_y", 0.001))
        self._search_step_z = float(rospy.get_param("~search/step_z", 0.001))
        self._search_width = float(rospy.get_param("~search/max_search_width", 0.02))
        self._search_height = float(rospy.get_param("~search/max_search_height", 0.02))
        self._search_timeout = float(rospy.get_param("~search/search_timeout", 50.0))
        self._search_traverse_timeout = float(rospy.get_param("~search/traverse_timeout", 4.0))
        self._search_traverse_speed = float(
            rospy.get_param(
                "~search/traverse_speed",
                rospy.get_param("~motion/search_traverse_speed", 0.012),
            )
        )
        self._search_traverse_tolerance = float(rospy.get_param("~search/traverse_tolerance", 0.0003))
        self._search_lift_off_distance = float(rospy.get_param("~search/lift_off_distance", 0.0001))
        self._search_pre_probe_approach_fraction = self._clamp(
            float(rospy.get_param("~search/pre_probe_approach_fraction", 0.9)),
            0.0,
            1.0,
        )
        self._search_diagonal_pre_probe_approach = bool(
            rospy.get_param("~search/diagonal_pre_probe_approach", True)
        )
        self._search_preferred_tool_x_sign = float(rospy.get_param("~search/preferred_tool_x_sign", -1.0))
        self._search_preferred_z_sign = float(rospy.get_param("~search/preferred_z_sign", 1.0))
        self._search_diagonal_first = bool(rospy.get_param("~search/diagonal_first", True))
        self._search_lift_off_speed = float(rospy.get_param("~search/lift_off_speed", 0.001))
        self._search_pre_probe_approach_speed = float(rospy.get_param("~search/pre_probe_approach_speed", 0.001))
        self._search_contact_force_target = float(rospy.get_param("~search/contact_force_target", 2.0))
        self._search_probe_speed = float(
            rospy.get_param(
                "~search/probe_speed",
                rospy.get_param("~search/force_control_speed_limit", 0.003),
            )
        )
        self._search_probe_timeout = float(
            rospy.get_param(
                "~search/probe_timeout",
                rospy.get_param("~search/force_control_timeout", 2.0),
            )
        )
        self._search_socket_depth_threshold = float(rospy.get_param("~search/socket_depth_threshold", 0.002))
        self._search_zero_ft_before_search = bool(
            rospy.get_param(
                "~search/zero_ft_before_search",
                rospy.get_param("~state_machine/auto_zero_ft", True),
            )
        )

        self._precontact_offset_tool_x = float(rospy.get_param("~state_machine/precontact_offset_tool_x", 0.0))
        self._precontact_offset_tool_y = float(rospy.get_param("~state_machine/precontact_offset_tool_y", 0.0))
        self._precontact_offset_tool_z = float(rospy.get_param("~state_machine/precontact_offset_tool_z", 0.010))
        self._target_offset_tool_x = float(
            rospy.get_param(
                "~state_machine/target_offset_tool_x",
                rospy.get_param("~state_machine/probe_offset_tool_x", 0.0),
            )
        )
        self._target_offset_tool_y = float(
            rospy.get_param(
                "~state_machine/target_offset_tool_y",
                rospy.get_param("~state_machine/probe_offset_tool_y", 0.0),
            )
        )
        self._max_probe_distance = float(rospy.get_param("~probe/max_probe_distance", 0.2))
        self._probe_timeout = float(rospy.get_param("~probe/probe_timeout", 20.0))
        self._force_threshold_x = float(rospy.get_param("~contact/force_threshold_x", 2.0))
        self._move_settle_time = float(rospy.get_param("~motion/action_settle_time", 0.4))

        self._robot = RobotInterface()
        self._tf = TFInterface()
        self._ft = FTInterface(
            wrench_topic=rospy.get_param("~topics/wrench", "/wrench"),
            filter_window_size=rospy.get_param("~contact/baseline_window", 20),
            wrench_timeout=rospy.get_param("~contact/wrench_timeout", 0.2),
        )
        self._contact_detector = ContactDetector(
            self._ft,
            hysteresis=rospy.get_param("~contact/hysteresis", 0.5),
        )
        self._wall_probe = WallProbe(self._robot, self._tf, self._contact_detector)
        self._move_client = actionlib.SimpleActionClient(self._move_action_name, MoveToPoseAction)
        self._server = actionlib.SimpleActionServer(
            self._action_name,
            SearchPortAction,
            execute_cb=self._execute,
            auto_start=False,
        )
        self._server.start()
        rospy.loginfo("[usb_c_insertion] event=search_port_action_ready action=%s", self._action_name)

    def _execute(self, goal) -> None:
        timeout = self._goal_or_default(goal.timeout, self._search_timeout)
        started_at = rospy.Time.now()

        if not self._move_client.wait_for_server(rospy.Duration.from_sec(5.0)):
            self._abort("move_action_unavailable")
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
            precontact_xyz = self._compute_precontact_xyz(goal.port_pose, reference_quaternion)
        except ValueError as exc:
            self._abort("search_geometry_failed: %s" % exc, "search_geometry_failed")
            return

        if self._search_zero_ft_before_search:
            self._publish_feedback("zero_ft_before_search", started_at, 0, 0, None, probe_direction)
            if not self._ft.zero_sensor():
                self._abort("zero_ft_before_search_failed")
                return
            rospy.sleep(0.5)

        self._publish_feedback("move_to_port_precontact", started_at, 0, 0, None, probe_direction)
        move_success, move_error_code = self._move_to_pose(
            precontact_xyz,
            reference_quaternion,
            min(timeout, self._search_timeout),
        )
        if not move_success:
            self._abort("move_to_port_precontact_failed", move_error_code)
            return

        self._publish_feedback("approach_wall_near_port", started_at, 0, 0, None, probe_direction)
        contact_axis = self._dominant_axis_name(probe_direction)
        probe_result = self._wall_probe.probe_until_contact(
            direction_xyz=probe_direction,
            axis_name=contact_axis,
            threshold=self._force_threshold_x,
            max_travel_distance=self._max_probe_distance,
            timeout=min(self._probe_timeout, timeout),
            retract_distance=0.0,
        )
        if not probe_result.success or probe_result.contact_point is None:
            self._abort(
                "approach_wall_near_port_failed",
                "approach_wall_%s" % probe_result.reason,
                probe_result.reason,
            )
            return

        reference_point = (
            probe_result.contact_point.point.x,
            probe_result.contact_point.point.y,
            probe_result.contact_point.point.z,
        )

        self._publish_feedback("search_initial_probe", started_at, 0, 0, reference_point, probe_direction)
        probe_success, probe_reason, inserted_depth, contact_force = self._probe_search_position(
            reference_point,
            probe_direction,
            "initial",
            0,
            0,
        )
        if not probe_success:
            self._abort("search_initial_probe_failed", probe_reason, probe_reason)
            return

        self._publish_feedback("search_initial_probe_complete", started_at, 0, 0, reference_point, probe_direction)
        if probe_reason == "socket_depth_reached":
            result = self._make_result(True, True, "potential_port_found", "", "")
            result.wall_contact_point = probe_result.contact_point
            result.found_pose = self._current_pose_or_empty()
            result.inserted_depth = float(inserted_depth)
            result.contact_force = float(contact_force)
            result.completed_steps = 0
            self._server.set_succeeded(result)
            return

        try:
            pattern = generate_centered_raster_pattern(
                step_x=self._search_step_y,
                step_y=self._search_step_z,
                width=self._search_width,
                height=self._search_height,
                preferred_x_sign=self._search_preferred_tool_x_sign,
                preferred_y_sign=self._search_preferred_z_sign,
                diagonal_first=self._search_diagonal_first,
            )
        except ValueError as exc:
            self._abort("search_pattern_invalid: %s" % exc, "search_pattern_invalid")
            return
        if pattern:
            rospy.loginfo(
                "[usb_c_insertion] event=search_pattern_ready steps=%d first_dx=%.4f first_dz=%.4f preferred_tool_x_sign=%.1f preferred_z_sign=%.1f diagonal_first=%s",
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

        current_xyz = (
            current_pose.pose.position.x,
            current_pose.pose.position.y,
            current_pose.pose.position.z,
        )
        total_steps = len(pattern)

        for step_index, offset in enumerate(pattern, start=1):
            if self._server.is_preempt_requested():
                self._robot.stop_motion()
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

            self._publish_feedback(
                "search_lift_off",
                started_at,
                step_index,
                total_steps,
                reference_point,
                probe_direction,
            )
            move_success, move_error_code = self._move_along_direction(
                (-probe_direction[0], -probe_direction[1], -probe_direction[2]),
                self._search_lift_off_distance,
                self._search_lift_off_speed,
                self._search_traverse_timeout,
            )
            if not move_success:
                self._abort("search_lift_off_failed", move_error_code)
                return

            current_xyz = (
                current_xyz[0] + offset.dx * wall_tangent[0],
                current_xyz[1] + offset.dx * wall_tangent[1],
                current_xyz[2] + offset.dy,
            )
            target_lift_distance = self._search_lift_off_distance
            if self._search_diagonal_pre_probe_approach:
                target_lift_distance = self._search_lift_off_distance * (1.0 - self._search_pre_probe_approach_fraction)

            self._publish_feedback(
                "search_move",
                started_at,
                step_index,
                total_steps,
                reference_point,
                probe_direction,
            )
            move_success, move_error_code = self._move_to_xyz(
                self._build_offset_from_surface(current_xyz, probe_direction, target_lift_distance),
                self._search_traverse_speed,
                self._search_traverse_tolerance,
                self._search_traverse_timeout,
            )
            if not move_success:
                self._abort("search_motion_failed", move_error_code)
                return

            if not self._search_diagonal_pre_probe_approach:
                self._publish_feedback(
                    "search_pre_probe_approach",
                    started_at,
                    step_index,
                    total_steps,
                    reference_point,
                    probe_direction,
                )
                move_success, move_error_code = self._move_along_direction(
                    probe_direction,
                    self._search_lift_off_distance * self._search_pre_probe_approach_fraction,
                    self._search_pre_probe_approach_speed,
                    self._search_traverse_timeout,
                )
                if not move_success:
                    self._abort("search_pre_probe_approach_failed", move_error_code)
                    return

            self._publish_feedback(
                "search_probe",
                started_at,
                step_index,
                total_steps,
                reference_point,
                probe_direction,
            )
            probe_success, probe_reason, inserted_depth, contact_force = self._probe_search_position(
                reference_point,
                probe_direction,
                "raster",
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
                reference_point,
                probe_direction,
            )
            if probe_reason == "socket_depth_reached":
                result = self._make_result(True, True, "potential_port_found", "", "")
                result.wall_contact_point = probe_result.contact_point
                result.found_pose = self._current_pose_or_empty()
                result.inserted_depth = float(inserted_depth)
                result.contact_force = float(contact_force)
                result.completed_steps = step_index
                self._server.set_succeeded(result)
                return

        result = self._make_result(False, False, "search_pattern_exhausted", "search_pattern_exhausted", "")
        result.wall_contact_point = probe_result.contact_point
        result.found_pose = self._current_pose_or_empty()
        result.inserted_depth = float(self._compute_inserted_depth(reference_point, probe_direction))
        result.contact_force = float(self._get_search_contact_force())
        result.completed_steps = total_steps
        self._robot.stop_motion()
        self._server.set_aborted(result)

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

    def _build_offset_from_surface(self, surface_xyz, probe_direction, lift_distance: float) -> Tuple[float, float, float]:
        if lift_distance <= 0.0:
            return surface_xyz
        direction = self._normalize_vector(probe_direction)
        return (
            surface_xyz[0] - direction[0] * lift_distance,
            surface_xyz[1] - direction[1] * lift_distance,
            surface_xyz[2] - direction[2] * lift_distance,
        )

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

    def _move_to_xyz(
        self,
        target_xyz,
        speed: float,
        tolerance: float,
        timeout: float,
    ) -> tuple[bool, str]:
        commanded_speed = max(0.0, float(speed))
        if commanded_speed <= 0.0:
            self._robot.stop_motion()
            return False, "invalid_speed"

        deadline = rospy.Time.now() + rospy.Duration.from_sec(max(0.1, timeout))
        allowed_error = max(0.00005, float(tolerance))
        rate = rospy.Rate(max(1.0, float(rospy.get_param("~motion/command_rate", 500.0))))

        while not rospy.is_shutdown():
            if self._server.is_preempt_requested():
                self._robot.stop_motion()
                return False, "preempted"
            if rospy.Time.now() > deadline:
                self._robot.stop_motion()
                return False, "move_to_xyz_timeout"

            pose = self._tf.get_tool_pose_in_base()
            if pose is None:
                self._robot.stop_motion()
                return False, "missing_tf"

            current_xyz = (
                pose.pose.position.x,
                pose.pose.position.y,
                pose.pose.position.z,
            )
            delta = (
                target_xyz[0] - current_xyz[0],
                target_xyz[1] - current_xyz[1],
                target_xyz[2] - current_xyz[2],
            )
            distance = self._distance(current_xyz, target_xyz)
            if distance <= allowed_error:
                self._robot.stop_motion()
                return True, ""

            direction = tuple(component / distance for component in delta)
            self._robot.send_twist(
                direction[0] * commanded_speed,
                direction[1] * commanded_speed,
                direction[2] * commanded_speed,
                0.0,
                0.0,
                0.0,
            )
            rate.sleep()

        self._robot.stop_motion()
        return False, "shutdown"

    def _move_along_direction(
        self,
        direction_xyz,
        distance: float,
        speed: float,
        timeout: float,
    ) -> tuple[bool, str]:
        if distance <= 0.0:
            return True, ""

        start_pose = self._tf.get_tool_pose_in_base()
        if start_pose is None:
            self._robot.stop_motion()
            return False, "missing_initial_tf"

        direction = self._normalize_vector(direction_xyz)
        commanded_speed = max(0.0, float(speed))
        if commanded_speed <= 0.0:
            self._robot.stop_motion()
            return False, "invalid_speed"

        start_xyz = (
            start_pose.pose.position.x,
            start_pose.pose.position.y,
            start_pose.pose.position.z,
        )
        deadline = rospy.Time.now() + rospy.Duration.from_sec(max(0.1, timeout))
        rate = rospy.Rate(max(1.0, float(rospy.get_param("~motion/command_rate", 500.0))))

        while not rospy.is_shutdown():
            if self._server.is_preempt_requested():
                self._robot.stop_motion()
                return False, "preempted"
            if rospy.Time.now() > deadline:
                self._robot.stop_motion()
                return False, "move_along_direction_timeout"

            pose = self._tf.get_tool_pose_in_base()
            if pose is None:
                self._robot.stop_motion()
                return False, "missing_tf"

            traveled = self._project_displacement(
                start_xyz,
                direction,
                (
                    pose.pose.position.x,
                    pose.pose.position.y,
                    pose.pose.position.z,
                ),
            )
            if traveled >= distance:
                self._robot.stop_motion()
                return True, ""

            self._robot.send_twist(
                direction[0] * commanded_speed,
                direction[1] * commanded_speed,
                direction[2] * commanded_speed,
                0.0,
                0.0,
                0.0,
            )
            rate.sleep()

        self._robot.stop_motion()
        return False, "shutdown"

    def _probe_search_position(
        self,
        reference_point,
        direction_xyz,
        probe_label: str,
        step_index: int,
        total_steps: int,
    ) -> tuple[bool, str, float, float]:
        deadline = rospy.Time.now() + rospy.Duration.from_sec(max(0.1, self._search_probe_timeout))
        rate = rospy.Rate(max(1.0, float(rospy.get_param("~motion/command_rate", 500.0))))
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
                self._get_search_contact_force(),
                self._get_search_contact_force(),
                None,
            )
            return False, "missing_initial_tf", 0.0, self._get_search_contact_force()

        start_xyz = (
            start_pose.pose.position.x,
            start_pose.pose.position.y,
            start_pose.pose.position.z,
        )
        max_contact_force = self._get_search_contact_force()

        while not rospy.is_shutdown():
            if self._server.is_preempt_requested():
                self._robot.stop_motion()
                return self._finish_search_probe(probe_label, step_index, total_steps, False, "preempted", reference_point, direction, start_xyz, max_contact_force)
            if rospy.Time.now() > deadline:
                self._robot.stop_motion()
                return self._finish_search_probe(probe_label, step_index, total_steps, False, "search_probe_timeout", reference_point, direction, start_xyz, max_contact_force)
            if self._ft.is_wrench_stale():
                self._robot.stop_motion()
                return self._finish_search_probe(probe_label, step_index, total_steps, False, "stale_wrench", reference_point, direction, start_xyz, max_contact_force)
            if self._tf.get_tool_pose_in_base() is None:
                self._robot.stop_motion()
                return self._finish_search_probe(probe_label, step_index, total_steps, False, "missing_tf", reference_point, direction, start_xyz, max_contact_force)

            contact_force = self._get_search_contact_force()
            max_contact_force = max(max_contact_force, contact_force)
            inserted_depth = self._compute_inserted_depth(reference_point, direction)
            if contact_force >= self._search_contact_force_target:
                self._robot.stop_motion()
                return self._finish_search_probe(probe_label, step_index, total_steps, True, "wall_contact_reached", reference_point, direction, start_xyz, max_contact_force)

            if inserted_depth >= self._search_socket_depth_threshold:
                self._robot.stop_motion()
                return self._finish_search_probe(probe_label, step_index, total_steps, True, "socket_depth_reached", reference_point, direction, start_xyz, max_contact_force)

            self._robot.send_twist(
                direction[0] * self._search_probe_speed,
                direction[1] * self._search_probe_speed,
                direction[2] * self._search_probe_speed,
                0.0,
                0.0,
                0.0,
            )
            rate.sleep()

        self._robot.stop_motion()
        return self._finish_search_probe(probe_label, step_index, total_steps, False, "shutdown", reference_point, direction, start_xyz, max_contact_force)

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
        contact_force = self._get_search_contact_force()
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
        if final_xyz is None:
            rospy.loginfo(
                "[usb_c_insertion] event=search_probe_result label=%s step=%d total=%d success=%s reason=%s inserted_depth=%.4f travel_distance=%.4f contact_force=%.3f max_contact_force=%.3f final_pose=missing_tf",
                probe_label,
                int(step_index),
                int(total_steps),
                str(bool(success)).lower(),
                reason,
                inserted_depth,
                travel_distance,
                contact_force,
                max_contact_force,
            )
            return

        rospy.loginfo(
            "[usb_c_insertion] event=search_probe_result label=%s step=%d total=%d success=%s reason=%s inserted_depth=%.4f travel_distance=%.4f contact_force=%.3f max_contact_force=%.3f final_x=%.4f final_y=%.4f final_z=%.4f",
            probe_label,
            int(step_index),
            int(total_steps),
            str(bool(success)).lower(),
            reason,
            inserted_depth,
            travel_distance,
            contact_force,
            max_contact_force,
            final_xyz[0],
            final_xyz[1],
            final_xyz[2],
        )

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
        feedback.contact_force = float(self._get_search_contact_force())
        if reference_point is not None and probe_direction is not None:
            feedback.inserted_depth = float(self._compute_inserted_depth(reference_point, probe_direction))
        self._server.publish_feedback(feedback)

    def _abort(
        self,
        message: str,
        error_code: Optional[str] = None,
        failure_reason: Optional[str] = None,
    ) -> None:
        self._robot.stop_motion()
        self._server.set_aborted(
            self._make_result(False, False, message, error_code or message, failure_reason or "")
        )

    def _current_pose_or_empty(self) -> PoseStamped:
        pose = self._tf.get_tool_pose_in_base()
        if pose is not None:
            return pose
        empty = PoseStamped()
        empty.header.stamp = rospy.Time.now()
        empty.header.frame_id = self._base_frame
        return empty

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

    def _get_search_contact_force(self) -> float:
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
    def _dominant_axis_name(direction_xyz) -> str:
        abs_components = {"x": abs(direction_xyz[0]), "y": abs(direction_xyz[1]), "z": abs(direction_xyz[2])}
        return max(abs_components, key=abs_components.get)

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
    def _clamp(value: float, lower: float, upper: float) -> float:
        return max(lower, min(upper, value))


def main() -> None:
    rospy.init_node("usb_c_insertion_search_port_action_server")
    SearchPortActionServer()
    rospy.spin()


if __name__ == "__main__":
    main()
