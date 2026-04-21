#!/usr/bin/env python3

from __future__ import annotations

from enum import Enum
import math
from typing import Optional

import rospy

from contact_detector import ContactDetector
from ft_interface import FTInterface
from robot_interface import RobotInterface
from search_pattern import generate_raster_pattern
from tf_interface import TFInterface
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
    CHECK_INSERTION = "CHECK_INSERTION"
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

        self._command_rate = float(rospy.get_param("~motion/command_rate", 100.0))
        self._probe_speed = float(rospy.get_param("~motion/probe_speed", 0.003))
        self._search_speed = float(rospy.get_param("~motion/search_speed", 0.002))
        self._move_speed = float(rospy.get_param("~motion/move_speed", 0.01))
        self._max_probe_distance = float(rospy.get_param("~probe/max_probe_distance", 0.05))
        self._probe_timeout = float(rospy.get_param("~probe/probe_timeout", 10.0))
        self._second_probe_y_offset = float(rospy.get_param("~probe/second_probe_y_offset", 0.02))
        self._retract_distance = float(rospy.get_param("~probe/retract_distance", 0.01))
        self._search_step_y = float(rospy.get_param("~search/step_y", 0.0015))
        self._search_step_z = float(rospy.get_param("~search/step_z", 0.0015))
        self._search_width = float(rospy.get_param("~search/max_search_width", 0.01))
        self._search_height = float(rospy.get_param("~search/max_search_height", 0.01))
        self._search_timeout = float(rospy.get_param("~search/search_timeout", 20.0))
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
        self._prepose_offset_port_x = float(rospy.get_param("~state_machine/prepose_offset_port_x", -0.05))
        self._prepose_offset_port_y = float(rospy.get_param("~state_machine/prepose_offset_port_y", 0.0))
        self._prepose_offset_port_z = float(rospy.get_param("~state_machine/prepose_offset_port_z", 0.0))
        self._prepose_speed = float(rospy.get_param("~state_machine/prepose_speed", self._move_speed))
        self._prepose_timeout = float(rospy.get_param("~state_machine/prepose_timeout", 30.0))
        self._precontact_offset_x = float(rospy.get_param("~state_machine/precontact_offset_x", -0.01))
        self._yaw_alignment_gain = float(rospy.get_param("~state_machine/yaw_alignment_gain", 0.8))
        self._yaw_tolerance = float(rospy.get_param("~state_machine/yaw_tolerance", 0.03))
        self._position_tolerance = float(rospy.get_param("~state_machine/position_tolerance", 0.002))
        self._insertion_depth = float(rospy.get_param("~state_machine/insertion_depth", 0.004))

        self._probe_result_1: Optional[ProbeResult] = None
        self._probe_result_2: Optional[ProbeResult] = None
        self._wall_estimate = None
        self._last_search_reference_x: Optional[float] = None

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
            elif self._state == InsertionState.CHECK_INSERTION:
                self._handle_check_insertion()
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

    def _handle_move_to_prepose(self) -> None:
        target_x, target_y, target_z = self._compute_port_frame_target(
            self._prepose_offset_port_x,
            self._prepose_offset_port_y,
            self._prepose_offset_port_z,
        )
        self._log(
            "info",
            "move_to_prepose_target",
            offset_port_x=round(self._prepose_offset_port_x, 4),
            offset_port_y=round(self._prepose_offset_port_y, 4),
            offset_port_z=round(self._prepose_offset_port_z, 4),
            target_x=round(target_x, 4),
            target_y=round(target_y, 4),
            target_z=round(target_z, 4),
            port_qx=round(self._port_qx, 4),
            port_qy=round(self._port_qy, 4),
            port_qz=round(self._port_qz, 4),
            port_qw=round(self._port_qw, 4),
        )
        success = self._move_to_xyz(
            target_x,
            target_y,
            target_z,
            speed=self._prepose_speed,
            timeout=self._prepose_timeout,
            move_name="move_to_prepose",
        )
        if not success:
            self._fail("move_to_prepose_failed")
            return
        self._advance(InsertionState.PROBE_WALL_POINT_1)

    def _handle_probe_wall_point_1(self) -> None:
        self._probe_result_1 = self._wall_probe.probe_until_contact(
            direction_xyz=(1.0, 0.0, 0.0),
            axis_name="x",
            threshold=self._force_threshold_x,
            max_travel_distance=self._max_probe_distance,
            timeout=self._probe_timeout,
        )
        if not self._probe_result_1.success or self._probe_result_1.contact_point is None:
            self._fail("probe_wall_point_1_failed")
            return
        self._advance(InsertionState.PROBE_WALL_POINT_2)

    def _handle_probe_wall_point_2(self) -> None:
        current_pose = self._tf.get_tool_pose_in_base()
        if current_pose is None:
            self._fail("missing_tf_before_second_probe")
            return

        success = self._move_to_xyz(
            current_pose.pose.position.x,
            current_pose.pose.position.y + self._second_probe_y_offset,
            current_pose.pose.position.z,
            speed=self._probe_speed,
            timeout=8.0,
            move_name="move_to_second_probe_offset",
        )
        if not success:
            self._fail("move_to_second_probe_offset_failed")
            return

        self._probe_result_2 = self._wall_probe.probe_until_contact(
            direction_xyz=(1.0, 0.0, 0.0),
            axis_name="x",
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
        target_yaw = math.atan2(self._wall_estimate.wall_normal_y, self._wall_estimate.wall_normal_x)
        if not self._align_yaw(target_yaw, timeout=6.0):
            self._fail("align_tool_yaw_failed")
            return
        self._advance(InsertionState.MOVE_TO_PORT_PRECONTACT)

    def _handle_move_to_port_precontact(self) -> None:
        success = self._move_to_xyz(
            self._port_x + self._precontact_offset_x,
            self._port_y,
            self._port_z,
            speed=self._move_speed,
            timeout=8.0,
            move_name="move_to_port_precontact",
        )
        if not success:
            self._fail("move_to_port_precontact_failed")
            return
        self._advance(InsertionState.APPROACH_WALL_NEAR_PORT)

    def _handle_approach_wall_near_port(self) -> None:
        result = self._wall_probe.probe_until_contact(
            direction_xyz=(1.0, 0.0, 0.0),
            axis_name="x",
            threshold=self._force_threshold_x,
            max_travel_distance=self._max_probe_distance,
            timeout=self._probe_timeout,
            retract_distance=0.0,
        )
        if not result.success or result.contact_point is None:
            self._fail("approach_wall_near_port_failed")
            return
        self._last_search_reference_x = result.contact_point.point.x
        self._advance(InsertionState.SEARCH_FOR_PORT)

    def _handle_search_for_port(self) -> None:
        if self._wall_estimate is None or self._last_search_reference_x is None:
            self._fail("missing_search_context")
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

        for offset in pattern:
            if rospy.is_shutdown():
                self._fail("shutdown_during_search")
                return
            elapsed = (rospy.Time.now() - started_at).to_sec()
            if elapsed > self._search_timeout:
                self._fail("search_timeout")
                return

            current_y += offset.dx * wall_x
            current_x += offset.dx * wall_y * 0.0
            current_z += offset.dy

            if not self._move_to_xyz(
                current_x,
                current_y,
                current_z,
                speed=self._search_speed,
                timeout=4.0,
                move_name="search_step",
            ):
                self._fail("search_motion_failed")
                return

            self._robot.send_twist(self._probe_speed, 0.0, 0.0, 0.0, 0.0, 0.0)
            rospy.sleep(0.2)
            self._robot.stop_motion()

            if not self._contact_detector.detect_contact_norm(self._force_threshold_norm):
                self._advance(InsertionState.CHECK_INSERTION)
                return

        self._fail("search_pattern_exhausted")

    def _handle_check_insertion(self) -> None:
        if self._last_search_reference_x is None:
            self._fail("missing_insertion_reference")
            return

        pose = self._tf.get_tool_pose_in_base()
        if pose is None:
            self._fail("missing_tf_for_insertion_check")
            return

        inserted_depth = pose.pose.position.x - self._last_search_reference_x
        still_in_contact = self._contact_detector.detect_contact_norm(self._force_threshold_norm)
        if inserted_depth >= self._insertion_depth and not still_in_contact:
            self._advance(InsertionState.SUCCESS)
        else:
            self._fail("insertion_check_failed")

    def _move_to_xyz(self, x: float, y: float, z: float, speed: float, timeout: float, move_name: str) -> bool:
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
        self._log(
            "info",
            "move_start",
            move=move_name,
            start_x=round(start_pose.pose.position.x, 4),
            start_y=round(start_pose.pose.position.y, 4),
            start_z=round(start_pose.pose.position.z, 4),
            target_x=round(x, 4),
            target_y=round(y, 4),
            target_z=round(z, 4),
            distance=round(start_distance, 4),
            speed=round(speed, 4),
            timeout=round(timeout, 2),
        )

        target_yaw = self._yaw_from_quaternion(start_pose.pose.orientation)
        target_orientation = self._quaternion_from_yaw(target_yaw)
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
                self._log(
                    "info",
                    "move_complete",
                    move=move_name,
                    final_x=round(pose.pose.position.x, 4),
                    final_y=round(pose.pose.position.y, 4),
                    final_z=round(pose.pose.position.z, 4),
                    remaining_distance=round(distance, 4),
                )
                return True
            rate.sleep()

        self._robot.stop_motion()
        return False

    def _align_yaw(self, target_yaw: float, timeout: float) -> bool:
        current_pose = self._tf.get_tool_pose_in_base()
        if current_pose is None:
            self._robot.stop_motion()
            return False

        target_orientation = self._quaternion_from_yaw(target_yaw)
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
        rotated_x, rotated_y, rotated_z = self._rotate_vector_by_quaternion(
            offset_x,
            offset_y,
            offset_z,
            self._port_qx,
            self._port_qy,
            self._port_qz,
            self._port_qw,
        )
        return (
            self._port_x + rotated_x,
            self._port_y + rotated_y,
            self._port_z + rotated_z,
        )

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
