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
from search_pattern import generate_raster_pattern
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
        self._search_contact_force_target = float(rospy.get_param("~search/contact_force_target", 2.0))
        self._search_contact_force_tolerance = float(rospy.get_param("~search/contact_force_tolerance", 0.5))
        self._search_force_control_gain = float(rospy.get_param("~search/force_control_gain", 0.001))
        self._search_force_control_speed_limit = float(rospy.get_param("~search/force_control_speed_limit", 0.003))
        self._search_force_control_timeout = float(rospy.get_param("~search/force_control_timeout", 2.0))
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

        try:
            pattern = generate_raster_pattern(
                step_x=self._search_step_y,
                step_y=self._search_step_z,
                width=self._search_width,
                height=self._search_height,
            )
        except ValueError as exc:
            self._abort("search_pattern_invalid: %s" % exc, "search_pattern_invalid")
            return

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

            current_xyz = (
                current_xyz[0] + offset.dx * wall_tangent[0],
                current_xyz[1] + offset.dx * wall_tangent[1],
                current_xyz[2] + offset.dy,
            )
            self._publish_feedback(
                "search_move",
                started_at,
                step_index,
                total_steps,
                reference_point,
                probe_direction,
            )
            move_success, move_error_code = self._move_to_pose(
                current_xyz,
                reference_quaternion,
                self._search_traverse_timeout,
            )
            if not move_success:
                self._abort("search_motion_failed", move_error_code)
                return

            self._publish_feedback(
                "search_force_control",
                started_at,
                step_index,
                total_steps,
                reference_point,
                probe_direction,
            )
            force_success, force_reason = self._regulate_contact_force(
                reference_point,
                probe_direction,
                self._search_contact_force_target,
                self._search_contact_force_tolerance,
                self._search_force_control_timeout,
            )
            if not force_success:
                self._abort("search_force_control_failed", force_reason, force_reason)
                return

            inserted_depth = self._compute_inserted_depth(reference_point, probe_direction)
            contact_force = self._get_search_contact_force()
            if inserted_depth >= self._search_socket_depth_threshold:
                result = self._make_result(True, True, "port_found", "", "")
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

    def _regulate_contact_force(
        self,
        reference_point,
        direction_xyz,
        target_force: float,
        tolerance: float,
        timeout: float,
    ) -> tuple[bool, str]:
        deadline = rospy.Time.now() + rospy.Duration.from_sec(max(0.1, timeout))
        rate = rospy.Rate(max(1.0, float(rospy.get_param("~motion/command_rate", 100.0))))
        direction = self._normalize_vector(direction_xyz)

        while not rospy.is_shutdown():
            if self._server.is_preempt_requested():
                self._robot.stop_motion()
                return False, "preempted"
            if rospy.Time.now() > deadline:
                self._robot.stop_motion()
                return False, "force_control_timeout"
            if self._ft.is_wrench_stale():
                self._robot.stop_motion()
                return False, "stale_wrench"
            if self._tf.get_tool_pose_in_base() is None:
                self._robot.stop_motion()
                return False, "missing_tf"

            contact_force = self._get_search_contact_force()
            if abs(target_force - contact_force) <= tolerance:
                self._robot.stop_motion()
                return True, ""

            speed = max(
                -self._search_force_control_speed_limit,
                min(
                    self._search_force_control_speed_limit,
                    self._search_force_control_gain * (target_force - contact_force),
                ),
            )
            self._robot.send_twist(
                direction[0] * speed,
                direction[1] * speed,
                direction[2] * speed,
                0.0,
                0.0,
                0.0,
            )
            rate.sleep()

        self._robot.stop_motion()
        return False, "shutdown"

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


def main() -> None:
    rospy.init_node("usb_c_insertion_search_port_action_server")
    SearchPortActionServer()
    rospy.spin()


if __name__ == "__main__":
    main()
