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

from contact_detector import ContactDetector
from ft_interface import FTInterface
from prepose_planner import compute_port_frame_target, compute_tcp_target_orientation, rotate_vector_by_quaternion, tool_offset_to_port_offset
from robot_interface import RobotInterface
from tf_interface import TFInterface
from usb_c_insertion.msg import (
    MoveToPoseAction,
    MoveToPoseGoal,
    ProbeSurfaceAction,
    ProbeSurfaceFeedback,
    ProbeSurfaceResult,
)
from wall_frame_estimator import estimate_wall_yaw
from wall_probe import WallProbe


class ProbeSurfaceActionServer:
    """
    Execute both wall probes and estimate the planar yaw correction.
    """

    def __init__(self):
        self._action_name = str(rospy.get_param("~action_name", "probe_surface")).strip()
        self._move_action_name = str(rospy.get_param("~move_action_name", "move_to_pose")).strip()
        self._base_frame = str(rospy.get_param("~frames/base_frame", "base_link"))

        self._probe_offset_tool_x = float(
            rospy.get_param(
                "~state_machine/probe_offset_tool_x",
                rospy.get_param(
                    "~state_machine/prepose_offset_tool_x",
                    -float(rospy.get_param("~state_machine/prepose_offset_port_y", 0.0)),
                ),
            )
        )
        self._probe_offset_tool_y = float(
            rospy.get_param(
                "~state_machine/probe_offset_tool_y",
                rospy.get_param(
                    "~state_machine/prepose_offset_tool_y",
                    float(rospy.get_param("~state_machine/prepose_offset_port_z", 0.0)),
                ),
            )
        )
        self._probe_offset_x, self._probe_offset_y, self._probe_offset_z = tool_offset_to_port_offset(
            (
                self._probe_offset_tool_x,
                self._probe_offset_tool_y,
                0.0,
            )
        )
        self._second_probe_y_offset = float(rospy.get_param("~probe/second_probe_y_offset", 0.02))
        self._inter_probe_backoff_distance = float(rospy.get_param("~probe/inter_probe_backoff_distance", 0.01))
        self._probe_timeout = float(rospy.get_param("~probe/probe_timeout", 10.0))
        self._position_tolerance = float(rospy.get_param("~state_machine/position_tolerance", 0.002))
        self._orientation_tolerance = float(rospy.get_param("~motion/action_orientation_tolerance", 0.05))
        self._settle_time = float(rospy.get_param("~motion/action_settle_time", 0.4))
        self._force_threshold_x = float(rospy.get_param("~contact/force_threshold_x", 2.0))

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
            ProbeSurfaceAction,
            execute_cb=self._execute,
            auto_start=False,
        )
        self._server.start()
        rospy.loginfo("[usb_c_insertion] event=probe_surface_action_ready action=%s", self._action_name)

    def _execute(self, goal) -> None:
        if not self._move_client.wait_for_server(rospy.Duration.from_sec(5.0)):
            self._abort("move_action_unavailable")
            return

        port_pose = goal.port_pose
        frame_id = port_pose.header.frame_id.strip() or self._base_frame
        if frame_id != self._base_frame:
            self._abort("unsupported_port_pose_frame")
            return

        port_pose_tuple = (
            port_pose.pose.position.x,
            port_pose.pose.position.y,
            port_pose.pose.position.z,
            port_pose.pose.orientation.x,
            port_pose.pose.orientation.y,
            port_pose.pose.orientation.z,
            port_pose.pose.orientation.w,
        )
        port_quaternion = port_pose_tuple[3:]
        current_tool_pose = self._tf.get_tool_pose_in_base()
        if current_tool_pose is None:
            self._abort("current_tool_pose_unavailable")
            return

        target_orientation = compute_tcp_target_orientation(
            (
                current_tool_pose.pose.orientation.x,
                current_tool_pose.pose.orientation.y,
                current_tool_pose.pose.orientation.z,
                current_tool_pose.pose.orientation.w,
            ),
            port_quaternion,
        )

        self._publish_stage("move_to_first_probe_offset")
        nominal_probe_xyz = compute_port_frame_target(
            port_pose_tuple,
            (self._probe_offset_x, self._probe_offset_y, self._probe_offset_z),
        )
        try:
            first_lateral_offset = self._get_tool_lateral_offset(
                target_orientation,
                -0.5 * self._second_probe_y_offset,
            )
        except ValueError as exc:
            self._abort(str(exc))
            return
        first_target_xyz = (
            nominal_probe_xyz[0] + first_lateral_offset[0],
            nominal_probe_xyz[1] + first_lateral_offset[1],
            nominal_probe_xyz[2],
        )
        if not self._move_to_pose(first_target_xyz, target_orientation):
            self._abort("move_to_first_probe_offset_failed")
            return

        self._publish_stage("probe_wall_point_1")
        try:
            probe_direction = self._get_probe_direction_from_tool_z(target_orientation)
        except ValueError as exc:
            self._abort(str(exc))
            return
        contact_axis = self._dominant_axis_name(probe_direction)
        probe_result_1 = self._wall_probe.probe_until_contact(
            direction_xyz=probe_direction,
            axis_name=contact_axis,
            threshold=self._force_threshold_x,
            timeout=self._probe_timeout,
        )
        if not probe_result_1.success or probe_result_1.contact_point is None:
            self._abort("probe_wall_point_1_failed")
            return

        self._publish_stage("move_to_second_probe_offset")
        try:
            lateral_offset = self._get_tool_lateral_offset(
                target_orientation,
                self._second_probe_y_offset,
            )
        except ValueError as exc:
            self._abort(str(exc))
            return
        second_target_xyz = (
            probe_result_1.contact_point.point.x - probe_direction[0] * self._inter_probe_backoff_distance + lateral_offset[0],
            probe_result_1.contact_point.point.y - probe_direction[1] * self._inter_probe_backoff_distance + lateral_offset[1],
            probe_result_1.contact_point.point.z - probe_direction[2] * self._inter_probe_backoff_distance + lateral_offset[2],
        )
        if not self._move_to_pose(second_target_xyz, target_orientation):
            self._abort("move_to_second_probe_offset_failed")
            return

        self._publish_stage("probe_wall_point_2")
        try:
            probe_direction = self._get_probe_direction_from_tool_z(target_orientation)
        except ValueError as exc:
            self._abort(str(exc))
            return
        contact_axis = self._dominant_axis_name(probe_direction)
        probe_result_2 = self._wall_probe.probe_until_contact(
            direction_xyz=probe_direction,
            axis_name=contact_axis,
            threshold=self._force_threshold_x,
            timeout=self._probe_timeout,
        )
        if not probe_result_2.success or probe_result_2.contact_point is None:
            self._abort("probe_wall_point_2_failed")
            return

        self._publish_stage("estimate_wall_yaw")
        try:
            wall_estimate = estimate_wall_yaw(
                probe_result_1.contact_point.point,
                probe_result_2.contact_point.point,
            )
        except ValueError as exc:
            self._abort("estimate_wall_yaw_failed: %s" % exc)
            return

        result = self._make_result(True, True, "surface_found")
        result.yaw_correction_rad = float(self._compute_yaw_correction(port_quaternion, wall_estimate))
        result.contact_point_1 = probe_result_1.contact_point
        result.contact_point_2 = probe_result_2.contact_point
        self._server.set_succeeded(result)

    def _get_probe_direction_from_tool_z(self, tool_quaternion):
        direction = rotate_vector_by_quaternion(
            0.0,
            0.0,
            1.0,
            tool_quaternion[0],
            tool_quaternion[1],
            tool_quaternion[2],
            tool_quaternion[3],
        )
        return self._normalize_vector(direction)

    def _get_tool_lateral_offset(self, tool_quaternion, lateral_distance: float):
        tool_x_direction = rotate_vector_by_quaternion(
            1.0,
            0.0,
            0.0,
            tool_quaternion[0],
            tool_quaternion[1],
            tool_quaternion[2],
            tool_quaternion[3],
        )
        horizontal_direction = self._normalize_vector(
            (tool_x_direction[0], tool_x_direction[1], 0.0)
        )
        rospy.loginfo(
            "[usb_c_insertion] event=probe_surface_tool_axes tool_x_x=%.4f tool_x_y=%.4f tool_x_z=%.4f horizontal_x_x=%.4f horizontal_x_y=%.4f horizontal_x_z=%.4f lateral_distance=%.4f",
            tool_x_direction[0],
            tool_x_direction[1],
            tool_x_direction[2],
            horizontal_direction[0],
            horizontal_direction[1],
            horizontal_direction[2],
            lateral_distance,
        )
        return (
            horizontal_direction[0] * lateral_distance,
            horizontal_direction[1] * lateral_distance,
            0.0,
        )

    def _compute_yaw_correction(self, port_quaternion, wall_estimate) -> float:
        vision_tangent = rotate_vector_by_quaternion(
            0.0,
            1.0,
            0.0,
            port_quaternion[0],
            port_quaternion[1],
            port_quaternion[2],
            port_quaternion[3],
        )
        vision_tangent_yaw = math.atan2(vision_tangent[1], vision_tangent[0])
        measured_tangent_yaw = math.atan2(
            wall_estimate.wall_direction_y,
            wall_estimate.wall_direction_x,
        )
        correction = self._normalize_line_angle(measured_tangent_yaw - vision_tangent_yaw)
        rospy.loginfo(
            "[usb_c_insertion] event=probe_surface_yaw_compare vision_tangent_yaw=%.4f measured_tangent_yaw=%.4f yaw_correction=%.4f",
            vision_tangent_yaw,
            measured_tangent_yaw,
            correction,
        )
        return correction

    def _move_to_pose(self, target_xyz, target_quaternion) -> bool:
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
        goal.position_tolerance = self._position_tolerance
        goal.orientation_tolerance = self._orientation_tolerance
        goal.settle_time = self._settle_time
        goal.timeout = self._probe_timeout

        self._move_client.send_goal(goal)
        finished = self._move_client.wait_for_result(rospy.Duration.from_sec(self._probe_timeout + 5.0))
        if not finished:
            self._move_client.cancel_goal()
            return False

        result = self._move_client.get_result()
        return bool(result is not None and result.success)

    def _publish_stage(self, stage: str) -> None:
        feedback = ProbeSurfaceFeedback()
        feedback.stage = stage
        self._server.publish_feedback(feedback)

    def _abort(self, message: str) -> None:
        self._robot.stop_motion()
        self._server.set_aborted(self._make_result(False, False, message))

    @staticmethod
    def _make_result(success: bool, surface_found: bool, message: str) -> ProbeSurfaceResult:
        result = ProbeSurfaceResult()
        result.success = bool(success)
        result.surface_found = bool(surface_found)
        result.message = str(message)
        result.yaw_correction_rad = 0.0
        return result

    @staticmethod
    def _normalize_vector(direction_xyz):
        magnitude = math.sqrt(sum(component * component for component in direction_xyz))
        if magnitude <= 1e-9:
            raise ValueError("direction_xyz must be non-zero")
        return tuple(component / magnitude for component in direction_xyz)

    @staticmethod
    def _dominant_axis_name(direction_xyz) -> str:
        abs_components = {"x": abs(direction_xyz[0]), "y": abs(direction_xyz[1]), "z": abs(direction_xyz[2])}
        return max(abs_components, key=abs_components.get)

    @staticmethod
    def _normalize_line_angle(angle_rad) -> float:
        while angle_rad > math.pi:
            angle_rad -= 2.0 * math.pi
        while angle_rad < -math.pi:
            angle_rad += 2.0 * math.pi
        if angle_rad > 0.5 * math.pi:
            angle_rad -= math.pi
        elif angle_rad < -0.5 * math.pi:
            angle_rad += math.pi
        return angle_rad


def main() -> None:
    rospy.init_node("usb_c_insertion_probe_surface_action_server")
    ProbeSurfaceActionServer()
    rospy.spin()


if __name__ == "__main__":
    main()
