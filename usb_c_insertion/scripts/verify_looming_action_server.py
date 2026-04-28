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
    VerifyLoomingAction,
    VerifyLoomingFeedback,
    VerifyLoomingResult,
)


@dataclass(frozen=True)
class GreenBlobDetection:
    stamp: rospy.Time
    image_width: int
    image_height: int
    found: bool
    center_x: float = 0.0
    center_y: float = 0.0
    area: float = 0.0
    aspect_ratio: float = 0.0
    message: str = ""


class VerifyLoomingActionServer:
    """
    Verify optical expansion of the centered green marker during tool-Z motion.

    In an aligned approach the marker should mostly scale around the same image
    center. A large center drift indicates that the camera ray is not passing
    through the marker center yet.
    """

    def __init__(self):
        self._action_name = str(rospy.get_param("~looming/action_name", "verify_looming")).strip()
        self._default_image_topic = str(
            rospy.get_param("~looming/image_topic", "/zedm/zed_node/left/image_rect_color")
        ).strip()
        self._base_frame = str(rospy.get_param("~frames/base_frame", "base_link")).strip()
        self._command_rate = max(1.0, float(rospy.get_param("~looming/command_rate", 40.0)))
        self._default_travel_distance = float(rospy.get_param("~looming/travel_distance", 0.025))
        self._default_travel_speed = float(rospy.get_param("~looming/travel_speed", 0.006))
        self._default_timeout = float(rospy.get_param("~looming/timeout", 8.0))
        self._default_min_blob_area = float(rospy.get_param("~looming/min_blob_area", 120.0))
        self._default_min_scale_ratio = float(rospy.get_param("~looming/min_scale_ratio", 1.12))
        self._default_max_center_shift_px = float(rospy.get_param("~looming/max_center_shift_px", 10.0))
        self._default_max_aspect_ratio_change = float(
            rospy.get_param("~looming/max_aspect_ratio_change", 0.35)
        )
        self._default_tool_z_direction_sign = float(rospy.get_param("~looming/tool_z_direction_sign", 1.0))
        self._max_image_age = float(rospy.get_param("~looming/max_image_age", 0.5))
        self._max_lost_time = float(rospy.get_param("~looming/max_lost_time", 0.5))
        self._image_rotation_deg = self._normalize_image_rotation_deg(
            float(rospy.get_param("~looming/image_rotation_deg", 0.0))
        )
        self._hsv_lower = self._read_hsv_param("~looming/hsv_lower", (35, 70, 40))
        self._hsv_upper = self._read_hsv_param("~looming/hsv_upper", (90, 255, 255))
        self._morph_kernel_size = max(0, int(rospy.get_param("~looming/morph_kernel_size", 5)))
        self._motion_pipeline_wait_timeout = float(
            rospy.get_param("~motion/action_pipeline_wait_timeout", 2.0)
        )

        self._robot = RobotInterface()
        self._tf = TFInterface()
        self._lock = RLock()
        self._image_topic = ""
        self._image_subscriber = None
        self._latest_detection: Optional[GreenBlobDetection] = None

        self._subscribe_image_topic(self._default_image_topic)
        self._server = actionlib.SimpleActionServer(
            self._action_name,
            VerifyLoomingAction,
            execute_cb=self._execute,
            auto_start=False,
        )
        rospy.on_shutdown(self._handle_shutdown)
        self._server.start()
        rospy.loginfo(
            "[usb_c_insertion] event=verify_looming_action_ready action=%s image_topic=%s image_rotation_deg=%.1f",
            self._action_name,
            self._image_topic,
            self._image_rotation_deg,
        )

    def _execute(self, goal) -> None:
        image_topic = str(goal.image_topic).strip() or self._default_image_topic
        self._subscribe_image_topic(image_topic)

        travel_distance = abs(self._goal_or_default(goal.travel_distance, self._default_travel_distance))
        travel_speed = abs(self._goal_or_default(goal.travel_speed, self._default_travel_speed))
        timeout = self._goal_or_default(goal.timeout, self._default_timeout)
        min_blob_area = self._goal_or_default(goal.min_blob_area, self._default_min_blob_area)
        min_scale_ratio = max(1.0, self._goal_or_default(goal.min_scale_ratio, self._default_min_scale_ratio))
        max_center_shift_px = self._goal_or_default(goal.max_center_shift_px, self._default_max_center_shift_px)
        max_aspect_ratio_change = self._goal_or_default(
            goal.max_aspect_ratio_change,
            self._default_max_aspect_ratio_change,
        )
        direction_sign = self._goal_or_default(goal.tool_z_direction_sign, self._default_tool_z_direction_sign)
        direction_sign = 1.0 if direction_sign >= 0.0 else -1.0

        if travel_speed <= 0.0:
            self._abort("invalid_travel_speed")
            return
        if travel_distance <= 0.0:
            self._abort("invalid_travel_distance")
            return
        if not self._robot.wait_for_motion_pipeline(self._motion_pipeline_wait_timeout):
            self._abort("motion_pipeline_unavailable")
            return

        started_at = rospy.Time.now()
        deadline = started_at + rospy.Duration.from_sec(max(0.1, timeout))
        initial_detection = self._wait_for_initial_detection(deadline, min_blob_area)
        if initial_detection is None:
            self._abort("initial_marker_not_found", None, None, started_at)
            return

        verified_center_pose = self._tf.get_tool_pose_in_base()
        if verified_center_pose is None:
            self._abort("missing_initial_tool_pose", initial_detection, initial_detection, started_at)
            return

        start_xyz = (
            verified_center_pose.pose.position.x,
            verified_center_pose.pose.position.y,
            verified_center_pose.pose.position.z,
        )
        quaternion = (
            verified_center_pose.pose.orientation.x,
            verified_center_pose.pose.orientation.y,
            verified_center_pose.pose.orientation.z,
            verified_center_pose.pose.orientation.w,
        )
        tool_z = rotate_vector_by_quaternion(0.0, 0.0, direction_sign, *quaternion)
        last_seen_time = rospy.Time.now()
        latest_detection = initial_detection
        rate = rospy.Rate(self._command_rate)

        rospy.loginfo(
            "[usb_c_insertion] event=verify_looming_started image_topic=%s travel_distance=%.4f travel_speed=%.4f direction_sign=%.1f min_scale_ratio=%.3f max_center_shift_px=%.1f",
            image_topic,
            travel_distance,
            travel_speed,
            direction_sign,
            min_scale_ratio,
            max_center_shift_px,
        )

        while not rospy.is_shutdown():
            now = rospy.Time.now()
            if self._server.is_preempt_requested():
                self._robot.stop_motion()
                self._server.set_preempted(
                    self._make_result(False, "preempted", "preempted", verified_center_pose, latest_detection, initial_detection, start_xyz, started_at)
                )
                return
            if now > deadline:
                self._robot.stop_motion()
                self._abort("verify_looming_timeout", latest_detection, initial_detection, started_at, verified_center_pose, start_xyz)
                return

            detection = self._get_latest_detection(min_blob_area)
            if detection is None or self._is_detection_stale(detection, now):
                self._robot.send_zero_twist()
                self._publish_feedback("waiting_for_marker", started_at, initial_detection, latest_detection, start_xyz, 0.0)
                rate.sleep()
                continue
            if not detection.found:
                self._robot.send_zero_twist()
                if (now - last_seen_time).to_sec() > self._max_lost_time:
                    self._abort("marker_lost", detection, initial_detection, started_at, verified_center_pose, start_xyz)
                    return
                self._publish_feedback("marker_not_found", started_at, initial_detection, detection, start_xyz, 0.0)
                rate.sleep()
                continue

            latest_detection = detection
            last_seen_time = now
            center_shift = self._center_shift(initial_detection, detection)
            if center_shift > max_center_shift_px:
                self._robot.stop_motion()
                self._abort("center_shift_too_large", detection, initial_detection, started_at, verified_center_pose, start_xyz)
                return

            aspect_change = self._aspect_ratio_change(initial_detection, detection)
            if aspect_change > max_aspect_ratio_change:
                self._robot.stop_motion()
                self._abort("aspect_ratio_changed", detection, initial_detection, started_at, verified_center_pose, start_xyz)
                return

            traveled_distance = self._traveled_distance(start_xyz, tool_z)
            scale_ratio = self._scale_ratio(initial_detection, detection)
            self._publish_feedback(
                "verifying",
                started_at,
                initial_detection,
                detection,
                start_xyz,
                direction_sign * travel_speed,
            )

            if traveled_distance >= travel_distance:
                self._robot.stop_motion()
                scale_ok = scale_ratio >= min_scale_ratio if direction_sign > 0.0 else scale_ratio <= 1.0 / min_scale_ratio
                if scale_ok:
                    result = self._make_result(
                        True,
                        "looming_verified",
                        "",
                        verified_center_pose,
                        detection,
                        initial_detection,
                        start_xyz,
                        started_at,
                    )
                    self._server.set_succeeded(result)
                    rospy.loginfo(
                        "[usb_c_insertion] event=verify_looming_complete scale_ratio=%.3f center_shift_px=%.2f traveled_distance=%.4f",
                        scale_ratio,
                        center_shift,
                        traveled_distance,
                    )
                    return
                self._abort("scale_change_too_small", detection, initial_detection, started_at, verified_center_pose, start_xyz)
                return

            self._robot.send_twist(
                tool_z[0] * travel_speed,
                tool_z[1] * travel_speed,
                tool_z[2] * travel_speed,
                0.0,
                0.0,
                0.0,
            )
            rospy.loginfo_throttle(
                0.5,
                "[usb_c_insertion] event=verify_looming_progress traveled_distance=%.4f scale_ratio=%.3f center_shift_px=%.2f aspect_ratio_change=%.3f",
                traveled_distance,
                scale_ratio,
                center_shift,
                aspect_change,
            )
            rate.sleep()

        self._abort("shutdown", latest_detection, initial_detection, started_at, verified_center_pose, start_xyz)

    def _wait_for_initial_detection(
        self,
        deadline: rospy.Time,
        min_blob_area: float,
    ) -> Optional[GreenBlobDetection]:
        rate = rospy.Rate(self._command_rate)
        while not rospy.is_shutdown() and rospy.Time.now() < deadline:
            detection = self._get_latest_detection(min_blob_area)
            if detection is not None and detection.found and not self._is_detection_stale(detection, rospy.Time.now()):
                return detection
            rate.sleep()
        return None

    def _subscribe_image_topic(self, image_topic: str) -> None:
        topic = image_topic.strip() or self._default_image_topic
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
            detection = GreenBlobDetection(
                stamp=msg.header.stamp if msg.header.stamp != rospy.Time() else rospy.Time.now(),
                image_width=int(msg.width),
                image_height=int(msg.height),
                found=False,
                message=str(exc),
            )
        with self._lock:
            self._latest_detection = detection

    def _detect_green_blob(self, msg: Image) -> GreenBlobDetection:
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
        width = float(stats[largest_index, cv2.CC_STAT_WIDTH])
        height = float(stats[largest_index, cv2.CC_STAT_HEIGHT])
        aspect_ratio = width / height if height > 1e-6 else 0.0
        return self._make_detection(
            msg,
            True,
            float(center_x),
            float(center_y),
            area,
            aspect_ratio,
        )

    def _make_detection(
        self,
        msg: Image,
        found: bool,
        center_x: float = 0.0,
        center_y: float = 0.0,
        area: float = 0.0,
        aspect_ratio: float = 0.0,
        message: str = "",
    ) -> GreenBlobDetection:
        stamp = msg.header.stamp if msg.header.stamp != rospy.Time() else rospy.Time.now()
        return GreenBlobDetection(
            stamp=stamp,
            image_width=int(msg.width),
            image_height=int(msg.height),
            found=found,
            center_x=float(center_x),
            center_y=float(center_y),
            area=float(area),
            aspect_ratio=float(aspect_ratio),
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

    def _get_latest_detection(self, min_blob_area: float) -> Optional[GreenBlobDetection]:
        with self._lock:
            detection = self._latest_detection
        if detection is None:
            return None
        if detection.found and detection.area < max(0.0, min_blob_area):
            return GreenBlobDetection(
                stamp=detection.stamp,
                image_width=detection.image_width,
                image_height=detection.image_height,
                found=False,
                message="green_blob_too_small",
            )
        return detection

    def _publish_feedback(
        self,
        stage: str,
        started_at: rospy.Time,
        initial_detection: GreenBlobDetection,
        current_detection: Optional[GreenBlobDetection],
        start_xyz,
        command_tool_z: float,
    ) -> None:
        feedback = VerifyLoomingFeedback()
        feedback.stage = stage
        feedback.elapsed = float((rospy.Time.now() - started_at).to_sec())
        feedback.traveled_distance = self._traveled_distance(start_xyz, None)
        feedback.initial_center_x = float(initial_detection.center_x)
        feedback.initial_center_y = float(initial_detection.center_y)
        feedback.initial_area = float(initial_detection.area)
        if current_detection is not None:
            feedback.current_center_x = float(current_detection.center_x)
            feedback.current_center_y = float(current_detection.center_y)
            feedback.current_area = float(current_detection.area)
            feedback.scale_ratio = self._scale_ratio(initial_detection, current_detection)
            feedback.center_shift_px = self._center_shift(initial_detection, current_detection)
            feedback.aspect_ratio_change = self._aspect_ratio_change(initial_detection, current_detection)
        feedback.command_tool_z = float(command_tool_z)
        self._server.publish_feedback(feedback)

    def _make_result(
        self,
        success: bool,
        message: str,
        error_code: str,
        verified_center_pose: Optional[PoseStamped],
        current_detection: Optional[GreenBlobDetection],
        initial_detection: Optional[GreenBlobDetection],
        start_xyz,
        started_at: Optional[rospy.Time] = None,
    ) -> VerifyLoomingResult:
        result = VerifyLoomingResult()
        result.success = bool(success)
        result.message = str(message)
        result.error_code = str(error_code)
        if verified_center_pose is not None:
            result.verified_center_pose = verified_center_pose
        result.final_pose = self._current_pose_or_empty()
        result.traveled_distance = self._traveled_distance(start_xyz, None) if start_xyz is not None else 0.0
        if initial_detection is not None:
            result.initial_center_x = float(initial_detection.center_x)
            result.initial_center_y = float(initial_detection.center_y)
            result.initial_area = float(initial_detection.area)
            result.initial_aspect_ratio = float(initial_detection.aspect_ratio)
        if current_detection is not None:
            result.current_center_x = float(current_detection.center_x)
            result.current_center_y = float(current_detection.center_y)
            result.current_area = float(current_detection.area)
            result.current_aspect_ratio = float(current_detection.aspect_ratio)
        if initial_detection is not None and current_detection is not None:
            result.scale_ratio = self._scale_ratio(initial_detection, current_detection)
            result.center_shift_px = self._center_shift(initial_detection, current_detection)
            result.aspect_ratio_change = self._aspect_ratio_change(initial_detection, current_detection)
        if started_at is not None:
            result.elapsed = float((rospy.Time.now() - started_at).to_sec())
        return result

    def _abort(
        self,
        message: str,
        current_detection: Optional[GreenBlobDetection] = None,
        initial_detection: Optional[GreenBlobDetection] = None,
        started_at: Optional[rospy.Time] = None,
        verified_center_pose: Optional[PoseStamped] = None,
        start_xyz=None,
        error_code: Optional[str] = None,
    ) -> None:
        self._robot.stop_motion()
        self._server.set_aborted(
            self._make_result(
                False,
                message,
                error_code or message,
                verified_center_pose,
                current_detection,
                initial_detection,
                start_xyz,
                started_at,
            )
        )

    def _traveled_distance(self, start_xyz, direction_xyz) -> float:
        if start_xyz is None:
            return 0.0
        pose = self._tf.get_tool_pose_in_base()
        if pose is None:
            return 0.0
        displacement = (
            pose.pose.position.x - start_xyz[0],
            pose.pose.position.y - start_xyz[1],
            pose.pose.position.z - start_xyz[2],
        )
        if direction_xyz is None:
            return math.sqrt(sum(component * component for component in displacement))
        return max(0.0, sum(displacement[index] * direction_xyz[index] for index in range(3)))

    @staticmethod
    def _scale_ratio(initial_detection: GreenBlobDetection, current_detection: GreenBlobDetection) -> float:
        if initial_detection.area <= 1e-6:
            return 0.0
        return float(current_detection.area) / float(initial_detection.area)

    @staticmethod
    def _center_shift(initial_detection: GreenBlobDetection, current_detection: GreenBlobDetection) -> float:
        dx = current_detection.center_x - initial_detection.center_x
        dy = current_detection.center_y - initial_detection.center_y
        return math.sqrt(dx * dx + dy * dy)

    @staticmethod
    def _aspect_ratio_change(initial_detection: GreenBlobDetection, current_detection: GreenBlobDetection) -> float:
        if initial_detection.aspect_ratio <= 1e-6:
            return 0.0
        return abs(current_detection.aspect_ratio - initial_detection.aspect_ratio) / initial_detection.aspect_ratio

    def _is_detection_stale(self, detection: GreenBlobDetection, now: rospy.Time) -> bool:
        if self._max_image_age <= 0.0:
            return False
        return (now - detection.stamp).to_sec() > self._max_image_age

    def _current_pose_or_empty(self) -> PoseStamped:
        pose = self._tf.get_tool_pose_in_base()
        if pose is not None:
            return pose
        empty = PoseStamped()
        empty.header.stamp = rospy.Time.now()
        empty.header.frame_id = self._base_frame
        return empty

    def _handle_shutdown(self) -> None:
        self._robot.stop_motion()

    @staticmethod
    def _goal_or_default(value: float, default: float) -> float:
        return float(value) if float(value) > 0.0 else float(default)

    @staticmethod
    def _read_hsv_param(param_name: str, default_value) -> Tuple[int, int, int]:
        value = rospy.get_param(param_name, list(default_value))
        if not isinstance(value, (list, tuple)) or len(value) != 3:
            rospy.logwarn("[usb_c_insertion] event=verify_looming_invalid_hsv_param param=%s", param_name)
            value = default_value
        return tuple(max(0, min(255, int(component))) for component in value)

    @staticmethod
    def _normalize_image_rotation_deg(rotation_deg: float) -> float:
        normalized = float(rotation_deg) % 360.0
        allowed = (0.0, 90.0, 180.0, 270.0)
        closest = min(allowed, key=lambda value: abs(value - normalized))
        if abs(closest - normalized) > 1e-3:
            rospy.logwarn(
                "[usb_c_insertion] event=verify_looming_image_rotation_rounded requested_deg=%.1f applied_deg=%.1f",
                rotation_deg,
                closest,
            )
        return closest


def main() -> None:
    rospy.init_node("usb_c_insertion_verify_looming_action_server")
    VerifyLoomingActionServer()
    rospy.spin()


if __name__ == "__main__":
    main()
