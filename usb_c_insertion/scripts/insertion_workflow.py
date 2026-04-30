#!/usr/bin/env python3

from __future__ import annotations

from dataclasses import dataclass
import math
import os
import sys
from typing import Optional, Tuple

from geometry_msgs.msg import PointStamped, PoseStamped
import rospy

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from contact_detector import ContactDetector
from ft_interface import FTInterface
from param_utils import required_bool_param, required_float_param, required_int_param, required_str_param
from post_insertion_verifier import PostInsertionVerifier
from prepose_planner import rotate_vector_by_quaternion
from robot_interface import RobotInterface
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


class InsertionWorkflow:
    """
    Start the insertion phase from the precontact pose.

    The workflow first touches the housing/port surface along tool z, then
    checks whether the connector is already seated. The first retention check
    gently pulls on the connector; if that is inconclusive, the workflow falls
    back to the existing lateral/vertical counterforce verification.
    """

    def __init__(self):
        self._mirror_global_config_to_private_namespace()

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
        self._lateral_verify_timeout = required_float_param("~insertion_workflow/lateral_verify_timeout")

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
        self._post_insertion_verifier = PostInsertionVerifier(self._robot, self._tf, self._ft)

    def _mirror_global_config_to_private_namespace(self) -> None:
        namespaces = (
            "frames",
            "topics",
            "motion",
            "contact",
            "probe",
            "verify",
            "gripper",
            "insertion_workflow",
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
                "[usb_c_insertion] event=insertion_workflow_params_mirrored_from_global namespaces=%s",
                ",".join(mirrored),
            )

    def run(self) -> bool:
        rospy.loginfo("[usb_c_insertion] event=insertion_workflow_start")
        if not self._robot.wait_for_motion_pipeline(
            timeout=self._pipeline_wait_timeout,
            require_pose_servo=True,
        ):
            rospy.logerr("[usb_c_insertion] event=insertion_workflow_failed reason=motion_pipeline_unavailable")
            return False

        if not self._wait_for_wrench():
            rospy.logerr("[usb_c_insertion] event=insertion_workflow_failed reason=wrench_unavailable")
            return False

        if self._zero_ft_before_contact:
            rospy.loginfo("[usb_c_insertion] event=insertion_workflow_zero_ft")
            if not self._ft.zero_sensor():
                rospy.logerr("[usb_c_insertion] event=insertion_workflow_failed reason=zero_ft_failed")
                return False
            rospy.sleep(max(0.0, self._zero_ft_settle_time))

        contact = self._approach_until_contact()
        if not contact.success or contact.contact_pose is None:
            rospy.logerr(
                "[usb_c_insertion] event=insertion_workflow_failed reason=contact_approach_%s",
                contact.reason,
            )
            return False

        rospy.sleep(max(0.0, self._contact_settle_time))
        pull_result = self._verify_pull_retention(contact.contact_pose)
        if pull_result.success:
            rospy.loginfo(
                "[usb_c_insertion] event=insertion_workflow_complete inserted_verified=true method=pull_retention pull_force=%.3f retraction=%.4f",
                pull_result.pull_force,
                pull_result.retraction_distance,
            )
            return True

        rospy.loginfo(
            "[usb_c_insertion] event=insertion_workflow_pull_retention_not_verified reason=%s pull_force=%.3f retraction=%.4f",
            pull_result.reason,
            pull_result.pull_force,
            pull_result.retraction_distance,
        )
        if self._return_to_contact_after_failed_pull:
            if not self._move_to_pose(contact.contact_pose, "return_to_contact_after_failed_pull", self._return_timeout):
                rospy.loginfo(
                    "[usb_c_insertion] event=insertion_workflow_complete inserted_verified=false method=return_to_contact reason=return_to_contact_failed"
                )
                return False

        if not self._lateral_verify_after_failed_pull:
            rospy.loginfo("[usb_c_insertion] event=insertion_workflow_complete inserted_verified=false method=pull_only")
            return False

        lateral_result = self._post_insertion_verifier.verify_retention(
            move_timeout=self._lateral_verify_timeout
        )
        if lateral_result.success:
            rospy.loginfo(
                "[usb_c_insertion] event=insertion_workflow_complete inserted_verified=true method=lateral_counterforce counterforce_y=%.3f counterforce_z=%.3f",
                lateral_result.counterforce_y,
                lateral_result.counterforce_z,
            )
            return True

        rospy.loginfo(
            "[usb_c_insertion] event=insertion_workflow_complete inserted_verified=false method=lateral_counterforce reason=%s counterforce_y=%.3f counterforce_z=%.3f",
            lateral_result.reason,
            lateral_result.counterforce_y,
            lateral_result.counterforce_z,
        )
        return False

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
            "[usb_c_insertion] event=insertion_pull_retention_start direction=(%.4f,%.4f,%.4f) target_force=%.3f required_force=%.3f max_retraction=%.4f",
            pull_direction[0],
            pull_direction[1],
            pull_direction[2],
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

            pull_force = self._get_opposing_force_along_direction(pull_direction)
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
                "[usb_c_insertion] event=insertion_pull_retention_progress pull_force=%.3f target_force=%.3f required_force=%.3f retraction=%.4f max_retraction=%.4f speed=%.4f",
                pull_force,
                self._pull_force_target,
                required_force,
                retraction,
                self._pull_max_retraction,
                speed,
            )
            rate.sleep()

        self._robot.stop_motion()
        return PullRetentionResult(False, "shutdown", max_force, max_retraction)

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
                rospy.logwarn("[usb_c_insertion] event=insertion_workflow_return_failed name=%s reason=missing_tf", name)
                return False
            position_error = self._pose_distance(pose, target_pose)
            orientation_error = self._orientation_error(pose, target_pose)
            if position_error <= self._position_tolerance and orientation_error <= self._orientation_tolerance:
                self._robot.enable_pose_servo(False)
                rospy.loginfo(
                    "[usb_c_insertion] event=insertion_workflow_return_complete name=%s position_error=%.5f orientation_error=%.5f",
                    name,
                    position_error,
                    orientation_error,
                )
                return True
            if rospy.Time.now() > deadline:
                self._robot.stop_motion()
                rospy.logwarn(
                    "[usb_c_insertion] event=insertion_workflow_return_failed name=%s reason=timeout position_error=%.5f orientation_error=%.5f",
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

    @staticmethod
    def _tool_z_direction(pose: PoseStamped, sign: float) -> Tuple[float, float, float]:
        qx = pose.pose.orientation.x
        qy = pose.pose.orientation.y
        qz = pose.pose.orientation.z
        qw = pose.pose.orientation.w
        direction = rotate_vector_by_quaternion(0.0, 0.0, sign, qx, qy, qz, qw)
        return InsertionWorkflow._normalize_vector(direction)

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


def main() -> None:
    rospy.init_node("usb_c_insertion_insertion_workflow")
    success = InsertionWorkflow().run()
    if not success:
        sys.exit(1)


if __name__ == "__main__":
    main()
