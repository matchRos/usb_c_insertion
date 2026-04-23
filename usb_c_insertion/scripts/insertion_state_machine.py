#!/usr/bin/env python3

from __future__ import annotations

from enum import Enum
import math
import os
import sys
from typing import Optional

import rospy

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from contact_detector import ContactDetector
from extraction_controller import ExtractionController
from ft_interface import FTInterface
from insertion_controller import InsertionController
from post_insertion_verifier import PostInsertionVerifier
from robot_interface import RobotInterface
from search_pattern import generate_raster_pattern
from tf_interface import TFInterface
from vision_pose_loader import load_vision_pose_from_json
from wall_frame_estimator import estimate_wall_yaw
from wall_probe import ProbeResult, WallProbe


class InsertionState(Enum):
    IDLE = "IDLE"
    ZERO_FT = "ZERO_FT"
    MOVE_TO_PREPOSE = "MOVE_TO_PREPOSE"
    PROBE_WALL_POINT_1 = "PROBE_WALL_POINT_1"
    PROBE_WALL_POINT_2 = "PROBE_WALL_POINT_2"
    ESTIMATE_WALL_YAW = "ESTIMATE_WALL_YAW"
    ALIGN_TOOL_YAW = "ALIGN_TOOL_YAW"
    MOVE_TO_PORT_PRECONTACT = "MOVE_TO_PORT_PRECONTACT"
    APPROACH_WALL_NEAR_PORT = "APPROACH_WALL_NEAR_PORT"
    SEARCH_FOR_PORT = "SEARCH_FOR_PORT"
    INSERT_CABLE = "INSERT_CABLE"
    CHECK_INSERTION = "CHECK_INSERTION"
    VERIFY_INSERTION = "VERIFY_INSERTION"
    EXTRACT_CABLE = "EXTRACT_CABLE"
    SUCCESS = "SUCCESS"
    FAILURE = "FAILURE"


class InsertionStateMachine:
    """
    First conservative insertion state machine.

    This version keeps transitions explicit and behaviors simple so the pipeline
    can be debugged incrementally on a real robot.
    """

    def __init__(self):
        self._state = InsertionState.IDLE
        self._failure_reason = ""

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
        self._insertion_controller = InsertionController(self._robot, self._tf, self._ft)
        self._post_insertion_verifier = PostInsertionVerifier(self._robot, self._tf, self._ft)
        self._extraction_controller = ExtractionController(self._robot, self._tf, self._ft)

        self._command_rate = float(rospy.get_param("~motion/command_rate", 100.0))
        self._probe_speed = float(rospy.get_param("~motion/probe_speed", 0.003))
        self._search_speed = float(rospy.get_param("~motion/search_speed", 0.002))
        self._move_speed = float(rospy.get_param("~motion/move_speed", 0.01))
        self._max_probe_distance = float(rospy.get_param("~probe/max_probe_distance", 0.05))
        self._probe_timeout = float(rospy.get_param("~probe/probe_timeout", 10.0))
        self._inter_probe_backoff_distance = float(rospy.get_param("~probe/inter_probe_backoff_distance", 0.01))
        self._second_probe_y_offset = float(rospy.get_param("~probe/second_probe_y_offset", 0.02))
        self._retract_distance = float(rospy.get_param("~probe/retract_distance", 0.01))
        self._search_step_y = float(rospy.get_param("~search/step_y", 0.0015))
        self._search_step_z = float(rospy.get_param("~search/step_z", 0.0015))
        self._search_width = float(rospy.get_param("~search/max_search_width", 0.01))
        self._search_height = float(rospy.get_param("~search/max_search_height", 0.01))
        self._search_timeout = float(rospy.get_param("~search/search_timeout", 20.0))
        self._search_traverse_speed = float(rospy.get_param("~motion/search_traverse_speed", 0.012))
        self._search_contact_force_target = float(rospy.get_param("~search/contact_force_target", 4.0))
        self._search_contact_force_tolerance = float(rospy.get_param("~search/contact_force_tolerance", 0.5))
        self._search_force_control_gain = float(rospy.get_param("~search/force_control_gain", 0.001))
        self._search_force_control_speed_limit = float(rospy.get_param("~search/force_control_speed_limit", 0.002))
        self._search_force_control_timeout = float(rospy.get_param("~search/force_control_timeout", 2.0))
        self._search_socket_depth_threshold = float(rospy.get_param("~search/socket_depth_threshold", 0.002))
        self._force_threshold_x = float(rospy.get_param("~contact/force_threshold_x", 4.0))
        self._force_threshold_norm = float(rospy.get_param("~contact/force_threshold_norm", 5.0))
        self._auto_zero_ft = bool(rospy.get_param("~state_machine/auto_zero_ft", True))
        self._stop_after_state_name = str(rospy.get_param("~state_machine/stop_after_state", "")).strip().upper()

        self._port_x = float(rospy.get_param("~port_estimate/x", 0.45))
        self._port_y = float(rospy.get_param("~port_estimate/y", 0.0))
        self._port_z = float(rospy.get_param("~port_estimate/z", 0.2))
        self._port_qx = float(rospy.get_param("~port_estimate/qx", 0.0))
        self._port_qy = float(rospy.get_param("~port_estimate/qy", 0.0))
        self._port_qz = float(rospy.get_param("~port_estimate/qz", 0.0))
        self._port_qw = float(rospy.get_param("~port_estimate/qw", 1.0))
        self._vision_pose_json_path = str(rospy.get_param("~vision_pose_json_path", "")).strip()
        self._prepose_offset_port_x = float(rospy.get_param("~state_machine/prepose_offset_port_x", -0.05))
        self._prepose_offset_port_y = float(rospy.get_param("~state_machine/prepose_offset_port_y", 0.0))
        self._prepose_offset_port_z = float(rospy.get_param("~state_machine/prepose_offset_port_z", 0.0))
        self._prepose_speed = float(rospy.get_param("~state_machine/prepose_speed", self._move_speed))
        self._prepose_timeout = float(rospy.get_param("~state_machine/prepose_timeout", 30.0))
        self._precontact_offset_port_x = float(rospy.get_param("~state_machine/precontact_offset_port_x", -0.01))
        self._precontact_offset_port_y = float(rospy.get_param("~state_machine/precontact_offset_port_y", 0.0))
        self._precontact_offset_port_z = float(rospy.get_param("~state_machine/precontact_offset_port_z", 0.0))
        self._yaw_alignment_gain = float(rospy.get_param("~state_machine/yaw_alignment_gain", 0.8))
        self._yaw_tolerance = float(rospy.get_param("~state_machine/yaw_tolerance", 0.03))
        self._position_tolerance = float(rospy.get_param("~state_machine/position_tolerance", 0.002))
        self._probe_result_1: Optional[ProbeResult] = None
        self._probe_result_2: Optional[ProbeResult] = None
        self._wall_estimate = None
        self._last_search_reference_x: Optional[float] = None
        self._search_reference_point = None
        self._search_probe_direction = None

        self._load_port_estimate()

        rospy.on_shutdown(self._robot.stop_motion)

    def run(self) -> bool:
        """
        Execute the state machine until success or failure.
        """
        while not rospy.is_shutdown():
            self._log_state("tick")

            if self._state == InsertionState.IDLE:
                self._transition(InsertionState.ZERO_FT)
            elif self._state == InsertionState.ZERO_FT:
                self._handle_zero_ft()
            elif self._state == InsertionState.MOVE_TO_PREPOSE:
                self._handle_move_to_prepose()
            elif self._state == InsertionState.PROBE_WALL_POINT_1:
                self._handle_probe_wall_point_1()
            elif self._state == InsertionState.PROBE_WALL_POINT_2:
                self._handle_probe_wall_point_2()
            elif self._state == InsertionState.ESTIMATE_WALL_YAW:
                self._handle_estimate_wall_yaw()
            elif self._state == InsertionState.ALIGN_TOOL_YAW:
                self._handle_align_tool_yaw()
            elif self._state == InsertionState.MOVE_TO_PORT_PRECONTACT:
                self._handle_move_to_port_precontact()
            elif self._state == InsertionState.APPROACH_WALL_NEAR_PORT:
                self._handle_approach_wall_near_port()
            elif self._state == InsertionState.SEARCH_FOR_PORT:
                self._handle_search_for_port()
            elif self._state == InsertionState.INSERT_CABLE:
                self._handle_insert_cable()
            elif self._state == InsertionState.CHECK_INSERTION:
                self._handle_check_insertion()
            elif self._state == InsertionState.VERIFY_INSERTION:
                self._handle_verify_insertion()
            elif self._state == InsertionState.EXTRACT_CABLE:
                self._handle_extract_cable()
            elif self._state == InsertionState.SUCCESS:
                self._robot.stop_motion()
                self._log("info", "run_complete", result="success")
                return True
            elif self._state == InsertionState.FAILURE:
                self._robot.stop_motion()
                self._log("err", "run_complete", result="failure", reason=self._failure_reason)
                return False

        self._robot.stop_motion()
        return False

    def _handle_zero_ft(self) -> None:
        if self._auto_zero_ft and not self._ft.zero_sensor():
            self._fail("zero_ft_failed")
            return
        self._advance(InsertionState.MOVE_TO_PREPOSE)

    def _load_port_estimate(self) -> None:
        if not self._vision_pose_json_path:
            return

        try:
            vision_pose = load_vision_pose_from_json(self._vision_pose_json_path)
        except (OSError, ValueError, KeyError, TypeError) as exc:
            self._log(
                "warn",
                "vision_pose_load_failed",
                path=self._vision_pose_json_path,
                reason=str(exc),
            )
            return

        self._port_x = vision_pose.x
        self._port_y = vision_pose.y
        self._port_z = vision_pose.z
        self._port_qx = vision_pose.qx
        self._port_qy = vision_pose.qy
        self._port_qz = vision_pose.qz
        self._port_qw = vision_pose.qw
        self._log(
            "info",
            "vision_pose_loaded",
            path=self._vision_pose_json_path,
            x=round(self._port_x, 4),
            y=round(self._port_y, 4),
            z=round(self._port_z, 4),
            yaw_rad=round(vision_pose.yaw_rad, 4),
        )

    def _handle_move_to_prepose(self) -> None:
        target_x, target_y, target_z = self._compute_port_frame_target(
            self._prepose_offset_port_x,
            self._prepose_offset_port_y,
            self._prepose_offset_port_z,
        )
        target_orientation = self._compute_tcp_target_orientation()
        success = self._move_to_xyz(
            target_x,
            target_y,
            target_z,
            speed=self._prepose_speed,
            timeout=self._prepose_timeout,
            move_name="move_to_prepose",
            target_orientation=target_orientation,
        )
        if not success:
            self._fail("move_to_prepose_failed")
            return
        self._advance(InsertionState.PROBE_WALL_POINT_1)

    def _handle_probe_wall_point_1(self) -> None:
        target_orientation = self._compute_tcp_target_orientation()
        first_probe_target_x, first_probe_target_y, first_probe_target_z = self._compute_port_frame_target(
            self._prepose_offset_port_x,
            -0.5 * self._second_probe_y_offset,
            self._prepose_offset_port_z,
        )
        if not self._move_to_xyz(
            first_probe_target_x,
            first_probe_target_y,
            first_probe_target_z,
            speed=self._probe_speed,
            timeout=8.0,
            move_name="move_to_first_probe_offset",
            target_orientation=target_orientation,
        ):
            self._fail("move_to_first_probe_offset_failed")
            return

        probe_direction = self._compute_port_frame_direction(1.0, 0.0, 0.0)
        contact_axis = self._dominant_axis_name(probe_direction)
        self._probe_result_1 = self._wall_probe.probe_until_contact(
            direction_xyz=probe_direction,
            axis_name=contact_axis,
            threshold=self._force_threshold_x,
            max_travel_distance=self._max_probe_distance,
            timeout=self._probe_timeout,
        )
        if not self._probe_result_1.success or self._probe_result_1.contact_point is None:
            self._fail("probe_wall_point_1_failed")
            return
        self._advance(InsertionState.PROBE_WALL_POINT_2)

    def _handle_probe_wall_point_2(self) -> None:
        target_orientation = self._compute_tcp_target_orientation()
        probe_direction = self._normalize_vector(self._compute_port_frame_direction(1.0, 0.0, 0.0))
        lateral_offset = self._compute_port_frame_direction(0.0, self._second_probe_y_offset, 0.0)
        second_probe_target_x = (
            self._probe_result_1.contact_point.point.x
            - probe_direction[0] * self._inter_probe_backoff_distance
            + lateral_offset[0]
        )
        second_probe_target_y = (
            self._probe_result_1.contact_point.point.y
            - probe_direction[1] * self._inter_probe_backoff_distance
            + lateral_offset[1]
        )
        second_probe_target_z = (
            self._probe_result_1.contact_point.point.z
            - probe_direction[2] * self._inter_probe_backoff_distance
            + lateral_offset[2]
        )

        success = self._move_to_xyz(
            second_probe_target_x,
            second_probe_target_y,
            second_probe_target_z,
            speed=self._probe_speed,
            timeout=8.0,
            move_name="move_to_second_probe_offset",
            target_orientation=target_orientation,
        )
        if not success:
            self._fail("move_to_second_probe_offset_failed")
            return

        contact_axis = self._dominant_axis_name(probe_direction)
        self._probe_result_2 = self._wall_probe.probe_until_contact(
            direction_xyz=probe_direction,
            axis_name=contact_axis,
            threshold=self._force_threshold_x,
            max_travel_distance=self._max_probe_distance,
            timeout=self._probe_timeout,
        )
        if not self._probe_result_2.success or self._probe_result_2.contact_point is None:
            self._fail("probe_wall_point_2_failed")
            return
        self._advance(InsertionState.ESTIMATE_WALL_YAW)

    def _handle_estimate_wall_yaw(self) -> None:
        try:
            self._wall_estimate = estimate_wall_yaw(
                self._probe_result_1.contact_point.point,
                self._probe_result_2.contact_point.point,
            )
        except (AttributeError, ValueError) as exc:
            self._fail("estimate_wall_yaw_failed: %s" % exc)
            return
        self._advance(InsertionState.ALIGN_TOOL_YAW)

    def _handle_align_tool_yaw(self) -> None:
        if self._wall_estimate is None:
            self._fail("missing_wall_estimate")
            return
        current_pose = self._tf.get_tool_pose_in_base()
        if current_pose is None:
            self._fail("missing_tf_before_align_tool_yaw")
            return

        current_yaw = self._yaw_from_quaternion(current_pose.pose.orientation)
        yaw_correction = self._wall_estimate.wall_yaw
        target_yaw = self._normalize_angle(current_yaw + yaw_correction)
        yaw_error = self._normalize_angle(target_yaw - current_yaw)
        if not self._align_yaw(target_yaw, timeout=6.0):
            self._fail("align_tool_yaw_failed")
            return
        self._advance(InsertionState.MOVE_TO_PORT_PRECONTACT)

    def _handle_move_to_port_precontact(self) -> None:
        target_orientation = self._compute_tcp_target_orientation()
        target_x, target_y, target_z = self._compute_port_frame_target(
            self._precontact_offset_port_x,
            self._precontact_offset_port_y,
            self._precontact_offset_port_z,
        )
        success = self._move_to_xyz(
            target_x,
            target_y,
            target_z,
            speed=self._move_speed,
            timeout=8.0,
            move_name="move_to_port_precontact",
            target_orientation=target_orientation,
        )
        if not success:
            self._fail("move_to_port_precontact_failed")
            return
        self._advance(InsertionState.APPROACH_WALL_NEAR_PORT)

    def _handle_approach_wall_near_port(self) -> None:
        if not self._zero_ft_for_search():
            self._fail("zero_ft_before_search_failed")
            return

        probe_direction = self._compute_port_frame_direction(1.0, 0.0, 0.0)
        contact_axis = self._dominant_axis_name(probe_direction)
        result = self._wall_probe.probe_until_contact(
            direction_xyz=probe_direction,
            axis_name=contact_axis,
            threshold=self._force_threshold_x,
            max_travel_distance=self._max_probe_distance,
            timeout=self._probe_timeout,
            retract_distance=0.0,
        )
        if not result.success or result.contact_point is None:
            self._fail("approach_wall_near_port_failed")
            return
        self._last_search_reference_x = result.contact_point.point.x
        self._search_reference_point = (
            result.contact_point.point.x,
            result.contact_point.point.y,
            result.contact_point.point.z,
        )
        self._search_probe_direction = probe_direction
        self._advance(InsertionState.SEARCH_FOR_PORT)

    def _handle_search_for_port(self) -> None:
        if self._wall_estimate is None or self._last_search_reference_x is None:
            self._fail("missing_search_context")
            return
        if self._search_reference_point is None or self._search_probe_direction is None:
            self._fail("missing_search_reference")
            return

        started_at = rospy.Time.now()
        pattern = generate_raster_pattern(
            step_x=self._search_step_y,
            step_y=self._search_step_z,
            width=self._search_width,
            height=self._search_height,
        )
        current_pose = self._tf.get_tool_pose_in_base()
        if current_pose is None:
            self._fail("missing_tf_before_search")
            return

        current_x = current_pose.pose.position.x
        current_y = current_pose.pose.position.y
        current_z = current_pose.pose.position.z

        wall_x = self._wall_estimate.wall_direction_x
        wall_y = self._wall_estimate.wall_direction_y
        search_orientation = (
            current_pose.pose.orientation.x,
            current_pose.pose.orientation.y,
            current_pose.pose.orientation.z,
            current_pose.pose.orientation.w,
        )
        for offset in pattern:
            if rospy.is_shutdown():
                self._fail("shutdown_during_search")
                return
            elapsed = (rospy.Time.now() - started_at).to_sec()
            if elapsed > self._search_timeout:
                self._fail("search_timeout")
                return

            current_x += offset.dx * wall_x
            current_y += offset.dx * wall_y
            current_z += offset.dy

            if not self._move_linear_to_xyz(
                current_x,
                current_y,
                current_z,
                speed=self._search_traverse_speed,
                timeout=4.0,
                move_name="search_step",
            ):
                self._fail("search_motion_failed")
                return

            if not self._regulate_contact_force(
                self._search_probe_direction,
                target_force=self._search_contact_force_target,
                tolerance=self._search_contact_force_tolerance,
                timeout=self._search_force_control_timeout,
            ):
                self._fail("search_force_control_failed")
                return

            if self._is_socket_depth_reached():
                self._advance(InsertionState.INSERT_CABLE)
                return

        self._fail("search_pattern_exhausted")

    def _handle_insert_cable(self) -> None:
        if self._search_reference_point is None or self._search_probe_direction is None:
            self._fail("missing_insert_context")
            return

        result = self._insertion_controller.insert_until_depth(
            self._search_reference_point,
            self._search_probe_direction,
        )
        if not result.success:
            self._fail("insert_cable_failed: %s" % result.reason)
            return

        self._advance(InsertionState.CHECK_INSERTION)

    def _handle_check_insertion(self) -> None:
        if self._last_search_reference_x is None:
            self._fail("missing_insertion_reference")
            return

        result = self._insertion_controller.check_insertion(
            self._search_reference_point,
            self._search_probe_direction,
        )
        if result.reason == "missing_tf":
            self._fail("missing_tf_for_insertion_check")
            return

        if result.success:
            self._log(
                "info",
                "insertion_check_passed",
                inserted_depth=round(result.inserted_depth, 4),
                contact_force=round(result.contact_force, 3),
                reason=result.reason,
            )
            self._advance(InsertionState.VERIFY_INSERTION)
        else:
            self._log(
                "err",
                "insertion_check_failed",
                inserted_depth=round(result.inserted_depth, 4),
                contact_force=round(result.contact_force, 3),
                reason=result.reason,
            )
            self._fail("insertion_check_failed")

    def _handle_verify_insertion(self) -> None:
        result = self._post_insertion_verifier.verify_retention()
        if result.success:
            self._log(
                "info",
                "post_insertion_verified",
                counterforce_y=round(result.counterforce_y, 3),
                counterforce_z=round(result.counterforce_z, 3),
            )
            self._advance(InsertionState.EXTRACT_CABLE)
            return

        self._log(
            "err",
            "post_insertion_verification_failed",
            reason=result.reason,
            counterforce_y=round(result.counterforce_y, 3),
            counterforce_z=round(result.counterforce_z, 3),
        )
        self._fail("post_insertion_verification_failed")

    def _handle_extract_cable(self) -> None:
        result = self._extraction_controller.extract()
        if result.success:
            self._log(
                "info",
                "cable_extracted",
                extracted_distance=round(result.extracted_distance, 4),
                pull_force=round(result.pull_force, 3),
                lateral_force=round(result.lateral_force, 3),
                torque_norm=round(result.torque_norm, 3),
                gripper_opened=str(result.gripper_opened).lower(),
            )
            self._advance(InsertionState.SUCCESS)
            return

        self._log(
            "err",
            "cable_extraction_failed",
            reason=result.reason,
            extracted_distance=round(result.extracted_distance, 4),
            pull_force=round(result.pull_force, 3),
            lateral_force=round(result.lateral_force, 3),
            torque_norm=round(result.torque_norm, 3),
        )
        self._fail("cable_extraction_failed")

    def _move_to_xyz(
        self,
        x: float,
        y: float,
        z: float,
        speed: float,
        timeout: float,
        move_name: str,
        target_orientation=None,
    ) -> bool:
        start_pose = self._tf.get_tool_pose_in_base()
        if start_pose is None:
            self._robot.stop_motion()
            self._log("err", "move_missing_tf", move=move_name)
            return False

        start_distance = math.sqrt(
            (x - start_pose.pose.position.x) ** 2
            + (y - start_pose.pose.position.y) ** 2
            + (z - start_pose.pose.position.z) ** 2
        )
        if target_orientation is None:
            target_orientation = (
                start_pose.pose.orientation.x,
                start_pose.pose.orientation.y,
                start_pose.pose.orientation.z,
                start_pose.pose.orientation.w,
            )
        self._robot.send_pose_target(
            x,
            y,
            z,
            qx=target_orientation[0],
            qy=target_orientation[1],
            qz=target_orientation[2],
            qw=target_orientation[3],
        )
        self._robot.enable_pose_servo(True)

        deadline = rospy.Time.now() + rospy.Duration.from_sec(timeout)
        rate = rospy.Rate(max(1.0, self._command_rate))

        while not rospy.is_shutdown():
            if rospy.Time.now() > deadline:
                pose = self._tf.get_tool_pose_in_base()
                self._robot.stop_motion()
                if pose is None:
                    self._log("err", "move_timeout", move=move_name, reason="missing_tf_on_timeout")
                else:
                    remaining_distance = math.sqrt(
                        (x - pose.pose.position.x) ** 2
                        + (y - pose.pose.position.y) ** 2
                        + (z - pose.pose.position.z) ** 2
                    )
                    self._log(
                        "err",
                        "move_timeout",
                        move=move_name,
                        current_x=round(pose.pose.position.x, 4),
                        current_y=round(pose.pose.position.y, 4),
                        current_z=round(pose.pose.position.z, 4),
                        remaining_distance=round(remaining_distance, 4),
                    )
                return False

            pose = self._tf.get_tool_pose_in_base()
            if pose is None:
                self._robot.stop_motion()
                self._log("err", "move_missing_tf", move=move_name)
                return False

            error_x = x - pose.pose.position.x
            error_y = y - pose.pose.position.y
            error_z = z - pose.pose.position.z
            distance = math.sqrt(error_x * error_x + error_y * error_y + error_z * error_z)
            if distance <= self._position_tolerance:
                self._robot.stop_motion()
                return True
            rate.sleep()

        self._robot.stop_motion()
        return False

    def _move_linear_to_xyz(
        self,
        x: float,
        y: float,
        z: float,
        speed: float,
        timeout: float,
        move_name: str,
    ) -> bool:
        deadline = rospy.Time.now() + rospy.Duration.from_sec(timeout)
        rate = rospy.Rate(max(1.0, self._command_rate))

        while not rospy.is_shutdown():
            if rospy.Time.now() > deadline:
                pose = self._tf.get_tool_pose_in_base()
                self._robot.stop_motion()
                if pose is None:
                    self._log("err", "move_timeout", move=move_name, reason="missing_tf_on_timeout")
                else:
                    remaining_distance = math.sqrt(
                        (x - pose.pose.position.x) ** 2
                        + (y - pose.pose.position.y) ** 2
                        + (z - pose.pose.position.z) ** 2
                    )
                    self._log(
                        "err",
                        "move_timeout",
                        move=move_name,
                        current_x=round(pose.pose.position.x, 4),
                        current_y=round(pose.pose.position.y, 4),
                        current_z=round(pose.pose.position.z, 4),
                        remaining_distance=round(remaining_distance, 4),
                    )
                return False

            pose = self._tf.get_tool_pose_in_base()
            if pose is None:
                self._robot.stop_motion()
                self._log("err", "move_missing_tf", move=move_name)
                return False

            error_x = x - pose.pose.position.x
            error_y = y - pose.pose.position.y
            error_z = z - pose.pose.position.z
            distance = math.sqrt(error_x * error_x + error_y * error_y + error_z * error_z)
            if distance <= self._position_tolerance:
                self._robot.stop_motion()
                return True

            limited_speed = min(max(0.0, speed), distance / max(1.0 / self._command_rate, 1e-3))
            direction_scale = limited_speed / max(distance, 1e-9)
            self._robot.send_twist(
                error_x * direction_scale,
                error_y * direction_scale,
                error_z * direction_scale,
                0.0,
                0.0,
                0.0,
            )
            rate.sleep()

        self._robot.stop_motion()
        return False

    def _align_yaw(self, target_yaw: float, timeout: float) -> bool:
        current_pose = self._tf.get_tool_pose_in_base()
        if current_pose is None:
            self._robot.stop_motion()
            return False

        current_yaw = self._yaw_from_quaternion(current_pose.pose.orientation)
        yaw_error = self._normalize_angle(target_yaw - current_yaw)

        target_orientation = self._apply_yaw_delta_to_quaternion(
            (
                current_pose.pose.orientation.x,
                current_pose.pose.orientation.y,
                current_pose.pose.orientation.z,
                current_pose.pose.orientation.w,
            ),
            yaw_error,
        )
        self._robot.send_pose_target(
            current_pose.pose.position.x,
            current_pose.pose.position.y,
            current_pose.pose.position.z,
            qx=target_orientation[0],
            qy=target_orientation[1],
            qz=target_orientation[2],
            qw=target_orientation[3],
        )
        self._robot.enable_pose_servo(True)

        deadline = rospy.Time.now() + rospy.Duration.from_sec(timeout)
        rate = rospy.Rate(max(1.0, self._command_rate))
        while not rospy.is_shutdown():
            if rospy.Time.now() > deadline:
                self._robot.stop_motion()
                return False

            transform = self._tf.get_tool_transform()
            if transform is None:
                self._robot.stop_motion()
                return False

            current_yaw = self._yaw_from_quaternion(transform.transform.rotation)
            yaw_error = self._normalize_angle(target_yaw - current_yaw)
            if abs(yaw_error) <= self._yaw_tolerance:
                self._robot.stop_motion()
                return True
            rate.sleep()

        self._robot.stop_motion()
        return False

    def _regulate_contact_force(self, direction_xyz, target_force: float, tolerance: float, timeout: float) -> bool:
        deadline = rospy.Time.now() + rospy.Duration.from_sec(timeout)
        rate = rospy.Rate(max(1.0, self._command_rate))
        direction = self._normalize_vector(direction_xyz)
        start_time = rospy.Time.now()
        last_progress_log_time = rospy.Time(0)

        start_pose = self._tf.get_tool_pose_in_base()
        start_force = self._get_search_contact_force()
        if start_pose is not None:
            self._log(
                "info",
                "force_control_start",
                target_force=round(target_force, 3),
                start_force=round(start_force, 3),
                start_x=round(start_pose.pose.position.x, 4),
                start_y=round(start_pose.pose.position.y, 4),
                start_z=round(start_pose.pose.position.z, 4),
                timeout=round(timeout, 2),
            )

        while not rospy.is_shutdown():
            if rospy.Time.now() > deadline:
                pose = self._tf.get_tool_pose_in_base()
                current_force = self._get_search_contact_force()
                self._robot.stop_motion()
                if pose is None:
                    self._log(
                        "err",
                        "force_control_timeout",
                        target_force=round(target_force, 3),
                        current_force=round(current_force, 3),
                        reason="missing_tf_on_timeout",
                    )
                else:
                    inserted_depth = self._project_displacement_from_reference(
                        pose.pose.position.x,
                        pose.pose.position.y,
                        pose.pose.position.z,
                    )
                    self._log(
                        "err",
                        "force_control_timeout",
                        target_force=round(target_force, 3),
                        current_force=round(current_force, 3),
                        inserted_depth=round(inserted_depth, 4),
                        current_x=round(pose.pose.position.x, 4),
                        current_y=round(pose.pose.position.y, 4),
                        current_z=round(pose.pose.position.z, 4),
                        elapsed=round((rospy.Time.now() - start_time).to_sec(), 2),
                    )
                return False
            if self._ft.is_wrench_stale():
                self._robot.stop_motion()
                self._log("err", "force_control_stale_wrench")
                return False

            contact_force = self._get_search_contact_force()
            force_error = target_force - contact_force
            if abs(force_error) <= tolerance:
                self._robot.stop_motion()
                pose = self._tf.get_tool_pose_in_base()
                if pose is not None:
                    inserted_depth = self._project_displacement_from_reference(
                        pose.pose.position.x,
                        pose.pose.position.y,
                        pose.pose.position.z,
                    )
                    self._log(
                        "info",
                        "force_control_reached",
                        target_force=round(target_force, 3),
                        current_force=round(contact_force, 3),
                        inserted_depth=round(inserted_depth, 4),
                        elapsed=round((rospy.Time.now() - start_time).to_sec(), 2),
                    )
                return True

            speed = max(
                -self._search_force_control_speed_limit,
                min(self._search_force_control_speed_limit, self._search_force_control_gain * force_error),
            )
            now = rospy.Time.now()
            if last_progress_log_time == rospy.Time(0) or (now - last_progress_log_time).to_sec() >= 0.5:
                pose = self._tf.get_tool_pose_in_base()
                if pose is not None:
                    inserted_depth = self._project_displacement_from_reference(
                        pose.pose.position.x,
                        pose.pose.position.y,
                        pose.pose.position.z,
                    )
                    self._log(
                        "info",
                        "force_control_progress",
                        target_force=round(target_force, 3),
                        current_force=round(contact_force, 3),
                        force_error=round(force_error, 3),
                        command_speed=round(speed, 4),
                        inserted_depth=round(inserted_depth, 4),
                    )
                else:
                    self._log(
                        "info",
                        "force_control_progress",
                        target_force=round(target_force, 3),
                        current_force=round(contact_force, 3),
                        force_error=round(force_error, 3),
                        command_speed=round(speed, 4),
                        inserted_depth="missing_tf",
                    )
                last_progress_log_time = now
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
        return False

    def _is_socket_depth_reached(self) -> bool:
        pose = self._tf.get_tool_pose_in_base()
        if pose is None:
            return False
        inserted_depth = self._project_displacement_from_reference(
            pose.pose.position.x,
            pose.pose.position.y,
            pose.pose.position.z,
        )
        return inserted_depth >= self._search_socket_depth_threshold

    def _get_search_contact_force(self) -> float:
        wrench = self._ft.get_filtered_wrench()
        return max(0.0, -wrench.force_z)

    def _zero_ft_for_search(self) -> bool:
        if not self._auto_zero_ft:
            return True
        if not self._ft.zero_sensor():
            return False
        rospy.sleep(0.5)
        return True

    def _transition(self, next_state: InsertionState) -> None:
        previous_state = self._state
        self._state = next_state
        self._log(
            "info",
            "state_transition",
            from_state=previous_state.value,
            to_state=next_state.value,
        )

    def _advance(self, next_state: InsertionState) -> None:
        completed_state = self._state
        if self._stop_after_state_name and completed_state.value == self._stop_after_state_name:
            self._log(
                "warn",
                "stop_after_state_reached",
                completed_state=completed_state.value,
                next_state=next_state.value,
            )
            self._transition(InsertionState.SUCCESS)
            return
        self._transition(next_state)

    def _compute_port_frame_target(self, offset_x: float, offset_y: float, offset_z: float):
        rotated_x, rotated_y, rotated_z = self._compute_port_frame_direction(
            offset_x,
            offset_y,
            offset_z,
        )
        return (
            self._port_x + rotated_x,
            self._port_y + rotated_y,
            self._port_z + rotated_z,
        )

    def _compute_port_frame_direction(self, vector_x: float, vector_y: float, vector_z: float):
        return self._rotate_vector_by_quaternion(
            vector_x,
            vector_y,
            vector_z,
            self._port_qx,
            self._port_qy,
            self._port_qz,
            self._port_qw,
        )

    def _compute_tcp_target_orientation(self):
        """
        Build the desired TCP yaw from the perceived case plane.

        The plane x-axis points out of the case wall. For the approach pose we
        use that projected direction directly as the TCP x-axis yaw reference
        in the robot XY plane. Only the yaw around robot z matters here.
        """
        current_pose = self._tf.get_tool_pose_in_base()
        if current_pose is None:
            return (
                self._port_qx,
                self._port_qy,
                self._port_qz,
                self._port_qw,
            )

        plane_x_in_base = self._rotate_vector_by_quaternion(
            1.0,
            0.0,
            0.0,
            self._port_qx,
            self._port_qy,
            self._port_qz,
            self._port_qw,
        )
        plane_x_yaw = math.atan2(plane_x_in_base[1], plane_x_in_base[0])
        tcp_x_yaw = self._normalize_angle(plane_x_yaw)
        current_roll, current_pitch, _ = self._euler_from_quaternion(current_pose.pose.orientation)
        return self._quaternion_from_euler(current_roll, current_pitch, tcp_x_yaw)

    def _fail(self, reason: str) -> None:
        self._failure_reason = reason
        self._transition(InsertionState.FAILURE)

    def _log_state(self, event: str) -> None:
        self._log("info", event, state=self._state.value)

    @staticmethod
    def _yaw_from_quaternion(quaternion) -> float:
        siny_cosp = 2.0 * (quaternion.w * quaternion.z + quaternion.x * quaternion.y)
        cosy_cosp = 1.0 - 2.0 * (quaternion.y * quaternion.y + quaternion.z * quaternion.z)
        return math.atan2(siny_cosp, cosy_cosp)

    @staticmethod
    def _euler_from_quaternion(quaternion):
        sinr_cosp = 2.0 * (quaternion.w * quaternion.x + quaternion.y * quaternion.z)
        cosr_cosp = 1.0 - 2.0 * (quaternion.x * quaternion.x + quaternion.y * quaternion.y)
        roll = math.atan2(sinr_cosp, cosr_cosp)

        sinp = 2.0 * (quaternion.w * quaternion.y - quaternion.z * quaternion.x)
        if abs(sinp) >= 1.0:
            pitch = math.copysign(0.5 * math.pi, sinp)
        else:
            pitch = math.asin(sinp)

        siny_cosp = 2.0 * (quaternion.w * quaternion.z + quaternion.x * quaternion.y)
        cosy_cosp = 1.0 - 2.0 * (quaternion.y * quaternion.y + quaternion.z * quaternion.z)
        yaw = math.atan2(siny_cosp, cosy_cosp)
        return (roll, pitch, yaw)

    @staticmethod
    def _quaternion_from_euler(roll: float, pitch: float, yaw: float):
        half_roll = 0.5 * roll
        half_pitch = 0.5 * pitch
        half_yaw = 0.5 * yaw

        sin_roll = math.sin(half_roll)
        cos_roll = math.cos(half_roll)
        sin_pitch = math.sin(half_pitch)
        cos_pitch = math.cos(half_pitch)
        sin_yaw = math.sin(half_yaw)
        cos_yaw = math.cos(half_yaw)

        return (
            sin_roll * cos_pitch * cos_yaw - cos_roll * sin_pitch * sin_yaw,
            cos_roll * sin_pitch * cos_yaw + sin_roll * cos_pitch * sin_yaw,
            cos_roll * cos_pitch * sin_yaw - sin_roll * sin_pitch * cos_yaw,
            cos_roll * cos_pitch * cos_yaw + sin_roll * sin_pitch * sin_yaw,
        )

    @staticmethod
    def _normalize_angle(angle: float) -> float:
        while angle > math.pi:
            angle -= 2.0 * math.pi
        while angle < -math.pi:
            angle += 2.0 * math.pi
        return angle

    @staticmethod
    def _rotate_vector_by_quaternion(
        vector_x: float,
        vector_y: float,
        vector_z: float,
        quat_x: float,
        quat_y: float,
        quat_z: float,
        quat_w: float,
    ):
        norm = math.sqrt(quat_x * quat_x + quat_y * quat_y + quat_z * quat_z + quat_w * quat_w)
        if norm <= 1e-9:
            raise ValueError("Port quaternion must be non-zero.")

        qx = quat_x / norm
        qy = quat_y / norm
        qz = quat_z / norm
        qw = quat_w / norm

        rotation_matrix = (
            (
                1.0 - 2.0 * (qy * qy + qz * qz),
                2.0 * (qx * qy - qz * qw),
                2.0 * (qx * qz + qy * qw),
            ),
            (
                2.0 * (qx * qy + qz * qw),
                1.0 - 2.0 * (qx * qx + qz * qz),
                2.0 * (qy * qz - qx * qw),
            ),
            (
                2.0 * (qx * qz - qy * qw),
                2.0 * (qy * qz + qx * qw),
                1.0 - 2.0 * (qx * qx + qy * qy),
            ),
        )

        return (
            rotation_matrix[0][0] * vector_x + rotation_matrix[0][1] * vector_y + rotation_matrix[0][2] * vector_z,
            rotation_matrix[1][0] * vector_x + rotation_matrix[1][1] * vector_y + rotation_matrix[1][2] * vector_z,
            rotation_matrix[2][0] * vector_x + rotation_matrix[2][1] * vector_y + rotation_matrix[2][2] * vector_z,
        )

    @staticmethod
    def _quaternion_from_yaw(yaw: float):
        half_yaw = 0.5 * yaw
        return (0.0, 0.0, math.sin(half_yaw), math.cos(half_yaw))

    @staticmethod
    def _apply_yaw_delta_to_quaternion(quaternion_xyzw, yaw_delta: float):
        delta_quaternion = InsertionStateMachine._quaternion_from_yaw(yaw_delta)
        return InsertionStateMachine._quaternion_multiply(delta_quaternion, quaternion_xyzw)

    @staticmethod
    def _dominant_axis_name(direction_xyz) -> str:
        abs_components = {
            "x": abs(direction_xyz[0]),
            "y": abs(direction_xyz[1]),
            "z": abs(direction_xyz[2]),
        }
        return max(abs_components, key=abs_components.get)

    @staticmethod
    def _normalize_vector(direction_xyz):
        magnitude = math.sqrt(sum(component * component for component in direction_xyz))
        if magnitude <= 1e-9:
            raise ValueError("direction_xyz must be non-zero")
        return tuple(component / magnitude for component in direction_xyz)

    def _project_displacement_from_reference(self, x: float, y: float, z: float) -> float:
        reference_x, reference_y, reference_z = self._search_reference_point
        direction = self._normalize_vector(self._search_probe_direction)
        delta = (x - reference_x, y - reference_y, z - reference_z)
        return sum(delta[index] * direction[index] for index in range(3))

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
    def _normalize_quaternion(quaternion_xyzw):
        norm = math.sqrt(sum(component * component for component in quaternion_xyzw))
        if norm <= 1e-9:
            return (0.0, 0.0, 0.0, 1.0)
        return tuple(component / norm for component in quaternion_xyzw)

    @staticmethod
    def _quaternion_from_rotation_matrix(matrix_rows):
        m00, m01, m02 = matrix_rows[0]
        m10, m11, m12 = matrix_rows[1]
        m20, m21, m22 = matrix_rows[2]
        trace = m00 + m11 + m22

        if trace > 0.0:
            s = math.sqrt(trace + 1.0) * 2.0
            qw = 0.25 * s
            qx = (m21 - m12) / s
            qy = (m02 - m20) / s
            qz = (m10 - m01) / s
        elif m00 > m11 and m00 > m22:
            s = math.sqrt(1.0 + m00 - m11 - m22) * 2.0
            qw = (m21 - m12) / s
            qx = 0.25 * s
            qy = (m01 + m10) / s
            qz = (m02 + m20) / s
        elif m11 > m22:
            s = math.sqrt(1.0 + m11 - m00 - m22) * 2.0
            qw = (m02 - m20) / s
            qx = (m01 + m10) / s
            qy = 0.25 * s
            qz = (m12 + m21) / s
        else:
            s = math.sqrt(1.0 + m22 - m00 - m11) * 2.0
            qw = (m10 - m01) / s
            qx = (m02 + m20) / s
            qy = (m12 + m21) / s
            qz = 0.25 * s

        return InsertionStateMachine._normalize_quaternion((qx, qy, qz, qw))

    @staticmethod
    def _log(level: str, event: str, **fields) -> None:
        message = "[usb_c_insertion] event=%s %s" % (
            event,
            " ".join("%s=%s" % (key, fields[key]) for key in sorted(fields)),
        )
        if level == "err":
            rospy.logerr(message)
        elif level == "warn":
            rospy.logwarn(message)
        else:
            rospy.loginfo(message)


def main() -> None:
    rospy.init_node("usb_c_insertion_state_machine")
    state_machine = InsertionStateMachine()
    state_machine.run()


if __name__ == "__main__":
    main()
