#!/usr/bin/env python3

from __future__ import annotations

import math
import os
import sys
from dataclasses import dataclass
from threading import RLock
from typing import Optional, Tuple

import actionlib
import cv2
from geometry_msgs.msg import PoseStamped
import numpy as np
import rospy
from sensor_msgs.msg import Image

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from prepose_planner import rotate_vector_by_quaternion
from robot_interface import RobotInterface
from tf_interface import TFInterface
from usb_c_insertion.msg import (
    CenterPortInImageAction,
    CenterPortInImageFeedback,
    CenterPortInImageResult,
)


@dataclass(frozen=True)
class PortDetection:
    stamp: rospy.Time
    image_width: int
    image_height: int
    found: bool
    center_x: float = 0.0
    center_y: float = 0.0
    area: float = 0.0
    error_x: float = 0.0
    error_y: float = 0.0
    error_norm: float = 0.0
    message: str = ""


class CenterPortInImageActionServer:
    """
    Center a green port marker in the camera image by moving in the TCP XY plane.

    The action intentionally commands zero velocity along tool Z, so it does not
    move closer to the port while centering the image.
    """

    def __init__(self):
        self._action_name = str(rospy.get_param("~center_port/action_name", "center_port_in_image")).strip()
        self._default_image_topic = str(
            rospy.get_param("~center_port/image_topic", "/zedm/zed_node/left/image_rect_color")
        ).strip()
        self._base_frame = str(rospy.get_param("~frames/base_frame", "base_link")).strip()
        self._command_rate = max(1.0, float(rospy.get_param("~center_port/command_rate", 20.0)))
        self._default_timeout = float(rospy.get_param("~center_port/timeout", 10.0))
        self._default_pixel_tolerance = float(rospy.get_param("~center_port/pixel_tolerance", 12.0))
        self._default_stable_time = float(rospy.get_param("~center_port/stable_time", 0.35))
        self._default_max_velocity = float(rospy.get_param("~center_port/max_velocity", 0.006))
        self._default_gain = float(rospy.get_param("~center_port/gain", 0.00002))
        self._default_min_blob_area = float(rospy.get_param("~center_port/min_blob_area", 120.0))
        self._max_acceleration = max(0.0, float(rospy.get_param("~center_port/max_acceleration", 0.03)))
        self._output_smoothing_alpha = max(
            0.0,
            min(1.0, float(rospy.get_param("~center_port/output_smoothing_alpha", 0.35))),
        )
        self._max_image_age = float(rospy.get_param("~center_port/max_image_age", 0.5))
        self._max_lost_time = float(rospy.get_param("~center_port/max_lost_time", 1.0))
        self._image_rotation_deg = self._normalize_image_rotation_deg(
            float(rospy.get_param("~center_port/image_rotation_deg", 0.0))
        )
        self._image_to_tool_rotation_rad = math.radians(
            float(rospy.get_param("~center_port/image_to_tool_rotation_deg", 0.0))
        )
        self._image_error_to_tool_x_sign = float(
            rospy.get_param("~center_port/image_error_to_tool_x_sign", 1.0)
        )
        self._image_error_to_tool_y_sign = float(
            rospy.get_param("~center_port/image_error_to_tool_y_sign", 1.0)
        )
        self._hsv_lower = self._read_hsv_param("~center_port/hsv_lower", (35, 70, 40))
        self._hsv_upper = self._read_hsv_param("~center_port/hsv_upper", (90, 255, 255))
        self._morph_kernel_size = max(0, int(rospy.get_param("~center_port/morph_kernel_size", 5)))
        self._motion_pipeline_wait_timeout = float(
            rospy.get_param("~motion/action_pipeline_wait_timeout", 2.0)
        )

        self._robot = RobotInterface()
        self._tf = TFInterface()
        self._lock = RLock()
        self._image_topic = ""
        self._image_subscriber = None
        self._latest_detection: Optional[PortDetection] = None

        self._subscribe_image_topic(self._default_image_topic)
        self._server = actionlib.SimpleActionServer(
            self._action_name,
            CenterPortInImageAction,
            execute_cb=self._execute,
            auto_start=False,
        )
        rospy.on_shutdown(self._handle_shutdown)
        self._server.start()
        rospy.loginfo(
            "[usb_c_insertion] event=center_port_in_image_action_ready action=%s image_topic=%s image_rotation_deg=%.1f image_to_tool_rotation_deg=%.1f image_error_to_tool_x_sign=%.1f image_error_to_tool_y_sign=%.1f",
            self._action_name,
            self._image_topic,
            self._image_rotation_deg,
            math.degrees(self._image_to_tool_rotation_rad),
            self._image_error_to_tool_x_sign,
            self._image_error_to_tool_y_sign,
        )

    def _execute(self, goal) -> None:
        image_topic = str(goal.image_topic).strip() or self._default_image_topic
        self._subscribe_image_topic(image_topic)

        timeout = self._goal_or_default(goal.timeout, self._default_timeout)
        pixel_tolerance = self._goal_or_default(goal.pixel_tolerance, self._default_pixel_tolerance)
        stable_time = self._goal_or_default(goal.stable_time, self._default_stable_time)
        max_velocity = self._goal_or_default(goal.max_velocity, self._default_max_velocity)
        gain = self._goal_or_default(goal.gain, self._default_gain)
        min_blob_area = self._goal_or_default(goal.min_blob_area, self._default_min_blob_area)

        if not self._robot.wait_for_motion_pipeline(self._motion_pipeline_wait_timeout):
            self._abort("motion_pipeline_unavailable")
            return

        started_at = rospy.Time.now()
        deadline = started_at + rospy.Duration.from_sec(max(0.1, timeout))
        centered_since: Optional[rospy.Time] = None
        last_seen = rospy.Time(0)
        latest_feedback_detection: Optional[PortDetection] = None
        last_command_time = started_at
        command_tool_x = 0.0
        command_tool_y = 0.0
        rate = rospy.Rate(self._command_rate)

        rospy.loginfo(
            "[usb_c_insertion] event=center_port_started image_topic=%s pixel_tolerance=%.1f max_velocity=%.4f max_acceleration=%.4f smoothing_alpha=%.2f gain=%.7f min_blob_area=%.1f",
            image_topic,
            pixel_tolerance,
            max_velocity,
            self._max_acceleration,
            self._output_smoothing_alpha,
            gain,
            min_blob_area,
        )

        while not rospy.is_shutdown():
            if self._server.is_preempt_requested():
                self._robot.stop_motion()
                self._server.set_preempted(self._make_result(False, "preempted", "preempted", latest_feedback_detection, started_at))
                return

            now = rospy.Time.now()
            if now > deadline:
                self._robot.stop_motion()
                self._abort("center_port_timeout", latest_feedback_detection, started_at)
                return

            detection = self._get_latest_detection(min_blob_area)
            latest_feedback_detection = detection
            if detection is None:
                command_tool_x = 0.0
                command_tool_y = 0.0
                self._robot.send_zero_twist()
                self._publish_feedback("waiting_for_image", started_at, None, 0.0, 0.0)
                rate.sleep()
                continue

            if self._is_detection_stale(detection, now):
                command_tool_x = 0.0
                command_tool_y = 0.0
                self._robot.send_zero_twist()
                self._publish_feedback("image_stale", started_at, detection, 0.0, 0.0)
                rate.sleep()
                continue

            if detection.found:
                last_seen = now
                if detection.error_norm <= pixel_tolerance:
                    command_tool_x, command_tool_y = self._smooth_tool_velocity(
                        command_tool_x,
                        command_tool_y,
                        0.0,
                        0.0,
                        self._command_dt(last_command_time, now),
                    )
                    last_command_time = now
                    if not self._send_tool_xy_twist(command_tool_x, command_tool_y):
                        self._abort("missing_tool_pose", detection, started_at)
                        return
                    if centered_since is None:
                        centered_since = now
                    self._publish_feedback("centered_settle", started_at, detection, command_tool_x, command_tool_y)
                    if (now - centered_since).to_sec() >= max(0.0, stable_time):
                        self._robot.send_zero_twist()
                        self._server.set_succeeded(
                            self._make_result(True, "centered", "", detection, started_at)
                        )
                        rospy.loginfo(
                            "[usb_c_insertion] event=center_port_complete error_norm=%.2f blob_area=%.1f elapsed=%.2f",
                            detection.error_norm,
                            detection.area,
                            (now - started_at).to_sec(),
                        )
                        return
                    rate.sleep()
                    continue

                centered_since = None
                desired_tool_x, desired_tool_y = self._compute_tool_velocity(
                    detection,
                    gain,
                    max_velocity,
                )
                command_tool_x, command_tool_y = self._smooth_tool_velocity(
                    command_tool_x,
                    command_tool_y,
                    desired_tool_x,
                    desired_tool_y,
                    self._command_dt(last_command_time, now),
                )
                last_command_time = now
                if not self._send_tool_xy_twist(command_tool_x, command_tool_y):
                    self._abort("missing_tool_pose", detection, started_at)
                    return
                self._publish_feedback("centering", started_at, detection, command_tool_x, command_tool_y)
                rospy.loginfo_throttle(
                    0.5,
                    "[usb_c_insertion] event=center_port_progress error_x=%.1f error_y=%.1f error_norm=%.1f area=%.1f command_tool_x=%.4f command_tool_y=%.4f",
                    detection.error_x,
                    detection.error_y,
                    detection.error_norm,
                    detection.area,
                    command_tool_x,
                    command_tool_y,
                )
                rate.sleep()
                continue

            command_tool_x = 0.0
            command_tool_y = 0.0
            self._robot.send_zero_twist()
            centered_since = None
            if last_seen != rospy.Time(0) and (now - last_seen).to_sec() > self._max_lost_time:
                self._abort("green_marker_lost", detection, started_at)
                return
            self._publish_feedback("marker_not_found", started_at, detection, 0.0, 0.0)
            rate.sleep()

        self._abort("shutdown", latest_feedback_detection, started_at)

    def _subscribe_image_topic(self, image_topic: str) -> None:
        topic = image_topic.strip()
        if not topic:
            topic = self._default_image_topic
        with self._lock:
            if topic == self._image_topic and self._image_subscriber is not None:
                return
            if self._image_subscriber is not None:
                self._image_subscriber.unregister()
            self._image_topic = topic
            self._latest_detection = None
            self._image_subscriber = rospy.Subscriber(topic, Image, self._image_callback, queue_size=1)

    def _image_callback(self, msg: Image) -> None:
        try:
            detection = self._detect_green_blob(msg)
        except ValueError as exc:
            detection = PortDetection(
                stamp=msg.header.stamp if msg.header.stamp != rospy.Time() else rospy.Time.now(),
                image_width=int(msg.width),
                image_height=int(msg.height),
                found=False,
                message=str(exc),
            )
        with self._lock:
            self._latest_detection = detection

    def _detect_green_blob(self, msg: Image) -> PortDetection:
        bgr = self._rotate_image_for_processing(self._image_to_bgr(msg))
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(
            hsv,
            np.array(self._hsv_lower, dtype=np.uint8),
            np.array(self._hsv_upper, dtype=np.uint8),
        )
        if self._morph_kernel_size > 1:
            kernel = np.ones((self._morph_kernel_size, self._morph_kernel_size), dtype=np.uint8)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        label_count, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, 8)
        if label_count <= 1:
            return self._make_detection(msg, False, message="no_green_blob")

        areas = stats[1:, cv2.CC_STAT_AREA]
        largest_index = int(np.argmax(areas)) + 1
        area = float(stats[largest_index, cv2.CC_STAT_AREA])
        center_x, center_y = centroids[largest_index]
        return self._make_detection(msg, True, float(center_x), float(center_y), area)

    def _make_detection(
        self,
        msg: Image,
        found: bool,
        center_x: float = 0.0,
        center_y: float = 0.0,
        area: float = 0.0,
        message: str = "",
    ) -> PortDetection:
        image_width = int(msg.width)
        image_height = int(msg.height)
        image_center_x = 0.5 * float(max(0, image_width - 1))
        image_center_y = 0.5 * float(max(0, image_height - 1))
        error_x = float(center_x) - image_center_x if found else 0.0
        error_y = float(center_y) - image_center_y if found else 0.0
        error_norm = math.sqrt(error_x * error_x + error_y * error_y)
        stamp = msg.header.stamp if msg.header.stamp != rospy.Time() else rospy.Time.now()
        return PortDetection(
            stamp=stamp,
            image_width=image_width,
            image_height=image_height,
            found=found,
            center_x=float(center_x),
            center_y=float(center_y),
            area=float(area),
            error_x=error_x,
            error_y=error_y,
            error_norm=error_norm,
            message=message,
        )

    def _image_to_bgr(self, msg: Image) -> np.ndarray:
        encoding = msg.encoding.lower().strip()
        width = int(msg.width)
        height = int(msg.height)
        step = int(msg.step)
        if width <= 0 or height <= 0 or step <= 0:
            raise ValueError("invalid_image_dimensions")

        if encoding in ("bgr8", "rgb8"):
            channels = 3
        elif encoding in ("bgra8", "rgba8"):
            channels = 4
        else:
            raise ValueError("unsupported_image_encoding:%s" % msg.encoding)

        expected_row_bytes = width * channels
        if step < expected_row_bytes:
            raise ValueError("invalid_image_step")

        data = np.frombuffer(msg.data, dtype=np.uint8)
        required_bytes = step * height
        if data.size < required_bytes:
            raise ValueError("short_image_data")

        rows = data[:required_bytes].reshape((height, step))
        image = rows[:, :expected_row_bytes].reshape((height, width, channels))
        if encoding == "bgr8":
            return image.copy()
        if encoding == "rgb8":
            return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        if encoding == "bgra8":
            return cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
        return cv2.cvtColor(image, cv2.COLOR_RGBA2BGR)

    def _rotate_image_for_processing(self, bgr: np.ndarray) -> np.ndarray:
        if self._image_rotation_deg == 0.0:
            return bgr
        if self._image_rotation_deg == 90.0:
            return cv2.rotate(bgr, cv2.ROTATE_90_CLOCKWISE)
        if self._image_rotation_deg == 180.0:
            return cv2.rotate(bgr, cv2.ROTATE_180)
        return cv2.rotate(bgr, cv2.ROTATE_90_COUNTERCLOCKWISE)

    def _get_latest_detection(self, min_blob_area: float) -> Optional[PortDetection]:
        with self._lock:
            detection = self._latest_detection
        if detection is None:
            return None
        if detection.found and detection.area < max(0.0, min_blob_area):
            return PortDetection(
                stamp=detection.stamp,
                image_width=detection.image_width,
                image_height=detection.image_height,
                found=False,
                message="green_blob_too_small",
            )
        return detection

    def _compute_tool_velocity(
        self,
        detection: PortDetection,
        gain: float,
        max_velocity: float,
    ) -> Tuple[float, float]:
        cos_angle = math.cos(self._image_to_tool_rotation_rad)
        sin_angle = math.sin(self._image_to_tool_rotation_rad)
        tool_error_x = cos_angle * detection.error_x - sin_angle * detection.error_y
        tool_error_y = sin_angle * detection.error_x + cos_angle * detection.error_y
        command_tool_x = self._image_error_to_tool_x_sign * float(gain) * tool_error_x
        command_tool_y = self._image_error_to_tool_y_sign * float(gain) * tool_error_y
        norm = math.sqrt(command_tool_x * command_tool_x + command_tool_y * command_tool_y)
        limit = max(0.0, float(max_velocity))
        if norm > limit > 0.0:
            scale = limit / norm
            command_tool_x *= scale
            command_tool_y *= scale
        return command_tool_x, command_tool_y

    def _smooth_tool_velocity(
        self,
        current_tool_x: float,
        current_tool_y: float,
        desired_tool_x: float,
        desired_tool_y: float,
        dt: float,
    ) -> Tuple[float, float]:
        alpha = self._output_smoothing_alpha
        filtered_tool_x = current_tool_x + alpha * (desired_tool_x - current_tool_x)
        filtered_tool_y = current_tool_y + alpha * (desired_tool_y - current_tool_y)

        delta_x = filtered_tool_x - current_tool_x
        delta_y = filtered_tool_y - current_tool_y
        max_delta = self._max_acceleration * max(0.0, float(dt))
        delta_norm = math.sqrt(delta_x * delta_x + delta_y * delta_y)
        if delta_norm > max_delta > 0.0:
            scale = max_delta / delta_norm
            delta_x *= scale
            delta_y *= scale
        return current_tool_x + delta_x, current_tool_y + delta_y

    def _command_dt(self, previous_time: rospy.Time, current_time: rospy.Time) -> float:
        elapsed = (current_time - previous_time).to_sec()
        if elapsed > 0.0:
            return elapsed
        return 1.0 / self._command_rate

    def _send_tool_xy_twist(self, command_tool_x: float, command_tool_y: float) -> bool:
        pose = self._tf.get_tool_pose_in_base()
        if pose is None:
            self._robot.stop_motion()
            return False

        quaternion = (
            pose.pose.orientation.x,
            pose.pose.orientation.y,
            pose.pose.orientation.z,
            pose.pose.orientation.w,
        )
        tool_x = rotate_vector_by_quaternion(1.0, 0.0, 0.0, *quaternion)
        tool_y = rotate_vector_by_quaternion(0.0, 1.0, 0.0, *quaternion)
        velocity_base = (
            tool_x[0] * command_tool_x + tool_y[0] * command_tool_y,
            tool_x[1] * command_tool_x + tool_y[1] * command_tool_y,
            tool_x[2] * command_tool_x + tool_y[2] * command_tool_y,
        )
        self._robot.send_twist(velocity_base[0], velocity_base[1], velocity_base[2], 0.0, 0.0, 0.0)
        return True

    def _publish_feedback(
        self,
        stage: str,
        started_at: rospy.Time,
        detection: Optional[PortDetection],
        command_tool_x: float,
        command_tool_y: float,
    ) -> None:
        feedback = CenterPortInImageFeedback()
        feedback.stage = stage
        feedback.elapsed = float((rospy.Time.now() - started_at).to_sec())
        if detection is not None:
            feedback.blob_center_x = float(detection.center_x)
            feedback.blob_center_y = float(detection.center_y)
            feedback.blob_area = float(detection.area)
            feedback.error_x = float(detection.error_x)
            feedback.error_y = float(detection.error_y)
            feedback.error_norm = float(detection.error_norm)
            feedback.image_width = int(detection.image_width)
            feedback.image_height = int(detection.image_height)
        feedback.command_tool_x = float(command_tool_x)
        feedback.command_tool_y = float(command_tool_y)
        self._server.publish_feedback(feedback)

    def _make_result(
        self,
        success: bool,
        message: str,
        error_code: str,
        detection: Optional[PortDetection],
        started_at: Optional[rospy.Time] = None,
    ) -> CenterPortInImageResult:
        result = CenterPortInImageResult()
        result.success = bool(success)
        result.message = str(message)
        result.error_code = str(error_code)
        result.final_pose = self._current_pose_or_empty()
        if detection is not None:
            result.blob_center_x = float(detection.center_x)
            result.blob_center_y = float(detection.center_y)
            result.blob_area = float(detection.area)
            result.error_x = float(detection.error_x)
            result.error_y = float(detection.error_y)
            result.error_norm = float(detection.error_norm)
            result.image_width = int(detection.image_width)
            result.image_height = int(detection.image_height)
        if started_at is not None:
            result.elapsed = float((rospy.Time.now() - started_at).to_sec())
        return result

    def _abort(
        self,
        message: str,
        detection: Optional[PortDetection] = None,
        started_at: Optional[rospy.Time] = None,
        error_code: Optional[str] = None,
    ) -> None:
        self._robot.stop_motion()
        self._server.set_aborted(
            self._make_result(False, message, error_code or message, detection, started_at)
        )

    def _current_pose_or_empty(self) -> PoseStamped:
        pose = self._tf.get_tool_pose_in_base()
        if pose is not None:
            return pose
        empty = PoseStamped()
        empty.header.stamp = rospy.Time.now()
        empty.header.frame_id = self._base_frame
        return empty

    def _is_detection_stale(self, detection: PortDetection, now: rospy.Time) -> bool:
        if self._max_image_age <= 0.0:
            return False
        return (now - detection.stamp).to_sec() > self._max_image_age

    def _handle_shutdown(self) -> None:
        self._robot.stop_motion()

    @staticmethod
    def _goal_or_default(value: float, default: float) -> float:
        return float(value) if float(value) > 0.0 else float(default)

    @staticmethod
    def _read_hsv_param(param_name: str, default_value) -> Tuple[int, int, int]:
        value = rospy.get_param(param_name, list(default_value))
        if not isinstance(value, (list, tuple)) or len(value) != 3:
            rospy.logwarn("[usb_c_insertion] event=center_port_invalid_hsv_param param=%s", param_name)
            value = default_value
        return tuple(max(0, min(255, int(component))) for component in value)

    @staticmethod
    def _normalize_image_rotation_deg(rotation_deg: float) -> float:
        normalized = float(rotation_deg) % 360.0
        allowed = (0.0, 90.0, 180.0, 270.0)
        closest = min(allowed, key=lambda value: abs(value - normalized))
        if abs(closest - normalized) > 1e-3:
            rospy.logwarn(
                "[usb_c_insertion] event=center_port_image_rotation_rounded requested_deg=%.1f applied_deg=%.1f",
                rotation_deg,
                closest,
            )
        return closest


def main() -> None:
    rospy.init_node("usb_c_insertion_center_port_in_image_action_server")
    CenterPortInImageActionServer()
    rospy.spin()


if __name__ == "__main__":
    main()
