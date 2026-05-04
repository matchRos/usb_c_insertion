#!/usr/bin/env python3

from __future__ import annotations

import math
import os
import sys
import time
from dataclasses import dataclass
from threading import RLock
from typing import Optional, Tuple

import actionlib
import cv2
from geometry_msgs.msg import PoseStamped
import numpy as np
import rospy
from sensor_msgs.msg import Image
from std_msgs.msg import String

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from param_utils import (
    get_param,
    required_bool_param,
    required_float_param,
    required_int_param,
    required_str_param,
    required_vector_param,
)
from prepose_planner import rotate_vector_by_quaternion
from robot_interface import RobotInterface
from tf_interface import TFInterface
from usb_card_target_selector import UsbCardTargetSelector
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
    bgr: Optional[np.ndarray] = None
    gray: Optional[np.ndarray] = None
    mask: Optional[np.ndarray] = None
    bbox: Optional[Tuple[int, int, int, int]] = None


class VerifyLoomingActionServer:
    """
    Verify optical expansion of the centered green marker during tool-Z motion.

    In an aligned approach the marker should mostly scale around the same image
    center. A large center drift indicates that the camera ray is not passing
    through the marker center yet.
    """

    def __init__(self):
        self._action_name = required_str_param("~looming/action_name")
        self._default_image_topic = required_str_param("~looming/image_topic")
        self._base_frame = required_str_param("~frames/base_frame")
        self._command_rate = max(1.0, required_float_param("~looming/command_rate"))
        self._default_travel_distance = required_float_param("~looming/travel_distance")
        self._default_travel_speed = required_float_param("~looming/travel_speed")
        self._default_timeout = required_float_param("~looming/timeout")
        self._default_min_blob_area = required_float_param("~looming/min_blob_area")
        self._default_min_scale_ratio = required_float_param("~looming/min_scale_ratio")
        self._default_max_center_shift_px = required_float_param("~looming/max_center_shift_px")
        self._default_max_aspect_ratio_change = required_float_param("~looming/max_aspect_ratio_change")
        self._default_tool_z_direction_sign = required_float_param("~looming/tool_z_direction_sign")
        self._max_image_age = required_float_param("~looming/max_image_age")
        self._max_lost_time = required_float_param("~looming/max_lost_time")
        self._image_rotation_deg = self._normalize_image_rotation_deg(
            required_float_param("~looming/image_rotation_deg")
        )
        self._hsv_lower = self._read_hsv_param("~looming/hsv_lower")
        self._hsv_upper = self._read_hsv_param("~looming/hsv_upper")
        self._morph_kernel_size = max(0, required_int_param("~looming/morph_kernel_size"))
        self._foe_debug_enabled = required_bool_param("~looming/foe_debug_enabled")
        self._foe_debug_output_dir = required_str_param("~looming/foe_debug_output_dir")
        self._frame_debug_enabled = required_bool_param("~looming/frame_debug_enabled")
        self._frame_debug_output_dir = required_str_param("~looming/frame_debug_output_dir")
        self._frame_debug_rate_hz = max(0.1, required_float_param("~looming/frame_debug_rate_hz"))
        self._frame_debug_max_frames = max(0, required_int_param("~looming/frame_debug_max_frames"))
        self._frame_debug_max_image_detection_dt = max(
            0.0,
            required_float_param("~looming/frame_debug_max_image_detection_dt"),
        )
        self._detection_source = str(get_param("~looming/detection_source", "green_marker")).strip().lower()
        self._usb_card_detections_topic = str(
            get_param(
                "~looming/usb_card_detections_topic",
                get_param("~usb_card_detector/detections_topic", "/usb_c_insertion/usb_card_detector/detections"),
            )
        ).strip()
        self._usb_card_selector = UsbCardTargetSelector.from_ros_params("looming")
        self._motion_pipeline_wait_timeout = required_float_param("~motion/action_pipeline_wait_timeout")

        self._robot = RobotInterface()
        self._tf = TFInterface()
        self._lock = RLock()
        self._image_topic = ""
        self._image_subscriber = None
        self._usb_card_detections_subscriber = None
        self._latest_detection: Optional[GreenBlobDetection] = None
        self._latest_bgr: Optional[np.ndarray] = None
        self._latest_bgr_stamp = rospy.Time(0)
        self._frame_debug_run_dir = ""
        self._frame_debug_count = 0
        self._frame_debug_last_save = rospy.Time(0)

        self._subscribe_image_topic(self._default_image_topic)
        self._subscribe_usb_card_detections()
        self._server = actionlib.SimpleActionServer(
            self._action_name,
            VerifyLoomingAction,
            execute_cb=self._execute,
            auto_start=False,
        )
        rospy.on_shutdown(self._handle_shutdown)
        self._server.start()
        rospy.loginfo(
            "[usb_c_insertion] event=verify_looming_action_ready action=%s image_topic=%s detection_source=%s usb_card_detections_topic=%s target_card_index=%d image_rotation_deg=%.1f",
            self._action_name,
            self._image_topic,
            self._detection_source,
            self._usb_card_detections_topic,
            self._usb_card_selector.target_card_index,
            self._image_rotation_deg,
        )

    def _execute(self, goal) -> None:
        image_topic = str(goal.image_topic).strip() or self._default_image_topic
        self._subscribe_image_topic(image_topic)
        self._subscribe_usb_card_detections()
        self._refresh_usb_card_selector()

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
        self._start_frame_debug_run(started_at)
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
        self._save_frame_debug_image("initial", initial_detection, initial_detection)

        rospy.loginfo(
            "[usb_c_insertion] event=verify_looming_started image_topic=%s target_card_index=%d target_point=%s require_connector=%s travel_distance=%.4f travel_speed=%.4f direction_sign=%.1f min_scale_ratio=%.3f max_center_shift_px=%.1f",
            image_topic,
            self._usb_card_selector.target_card_index,
            self._usb_card_selector.target_point,
            str(self._usb_card_selector.require_connector).lower(),
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
                self._save_frame_debug_image("not_found", detection, initial_detection)
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
            self._save_frame_debug_image("verifying", detection, initial_detection)
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
        if self._uses_usb_card_detector():
            try:
                bgr = self._rotate_image_for_processing(self._image_to_bgr(msg))
                stamp = msg.header.stamp if msg.header.stamp != rospy.Time() else rospy.Time.now()
                with self._lock:
                    self._latest_bgr = bgr
                    self._latest_bgr_stamp = stamp
            except ValueError as exc:
                rospy.logwarn_throttle(
                    2.0,
                    "[usb_c_insertion] event=verify_looming_frame_image_failed error=%s",
                    exc,
                )
            return
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
            self._latest_bgr = detection.bgr
            self._latest_bgr_stamp = detection.stamp

    def _usb_card_detections_callback(self, msg: String) -> None:
        if not self._uses_usb_card_detector():
            return
        target = self._usb_card_selector.select_from_json(msg.data)
        detection = GreenBlobDetection(
            stamp=target.stamp,
            image_width=target.image_width,
            image_height=target.image_height,
            found=target.found,
            center_x=target.center_x,
            center_y=target.center_y,
            area=target.area,
            aspect_ratio=target.aspect_ratio,
            message=target.message,
            bbox=target.bbox,
        )
        with self._lock:
            self._latest_detection = detection

    def _detect_green_blob(self, msg: Image) -> GreenBlobDetection:
        bgr = self._rotate_image_for_processing(self._image_to_bgr(msg))
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
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
            return self._make_detection(
                msg,
                False,
                message="no_green_blob",
                bgr=bgr,
                gray=gray,
                mask=np.zeros_like(mask),
            )

        areas = stats[1:, cv2.CC_STAT_AREA]
        largest_index = int(np.argmax(areas)) + 1
        area = float(stats[largest_index, cv2.CC_STAT_AREA])
        center_x, center_y = centroids[largest_index]
        width = float(stats[largest_index, cv2.CC_STAT_WIDTH])
        height = float(stats[largest_index, cv2.CC_STAT_HEIGHT])
        aspect_ratio = width / height if height > 1e-6 else 0.0
        component_mask = np.zeros_like(mask)
        component_mask[labels == largest_index] = 255
        return self._make_detection(
            msg,
            True,
            float(center_x),
            float(center_y),
            area,
            aspect_ratio,
            bgr=bgr,
            gray=gray,
            mask=component_mask,
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
        bgr: Optional[np.ndarray] = None,
        gray: Optional[np.ndarray] = None,
        mask: Optional[np.ndarray] = None,
    ) -> GreenBlobDetection:
        stamp = msg.header.stamp if msg.header.stamp != rospy.Time() else rospy.Time.now()
        if bgr is not None:
            image_height, image_width = bgr.shape[:2]
        else:
            image_width = int(msg.width)
            image_height = int(msg.height)
        return GreenBlobDetection(
            stamp=stamp,
            image_width=int(image_width),
            image_height=int(image_height),
            found=found,
            center_x=float(center_x),
            center_y=float(center_y),
            area=float(area),
            aspect_ratio=float(aspect_ratio),
            message=message,
            bgr=bgr,
            gray=gray,
            mask=mask,
            bbox=None,
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
                center_x=detection.center_x,
                center_y=detection.center_y,
                area=detection.area,
                aspect_ratio=detection.aspect_ratio,
                message="%s_too_small" % self._target_label(),
                bgr=detection.bgr,
                gray=detection.gray,
                mask=detection.mask,
                bbox=detection.bbox,
            )
        return detection

    def _subscribe_usb_card_detections(self) -> None:
        if not self._uses_usb_card_detector() or not self._usb_card_detections_topic:
            return
        with self._lock:
            if self._usb_card_detections_subscriber is not None:
                return
            self._usb_card_detections_subscriber = rospy.Subscriber(
                self._usb_card_detections_topic,
                String,
                self._usb_card_detections_callback,
                queue_size=1,
            )

    def _refresh_usb_card_selector(self) -> None:
        if not self._uses_usb_card_detector():
            return
        next_selector = UsbCardTargetSelector.from_ros_params("looming")
        previous_selector = self._usb_card_selector
        changed = (
            previous_selector.target_card_index != next_selector.target_card_index
            or previous_selector.target_point != next_selector.target_point
            or previous_selector.require_connector != next_selector.require_connector
            or previous_selector.order_axis != next_selector.order_axis
            or previous_selector.order_direction != next_selector.order_direction
            or previous_selector.expected_card_count != next_selector.expected_card_count
            or previous_selector.estimated_slot_requires_complete
            != next_selector.estimated_slot_requires_complete
        )
        if not changed:
            return
        with self._lock:
            self._usb_card_selector = next_selector
            self._latest_detection = None
        rospy.loginfo(
            "[usb_c_insertion] event=verify_looming_usb_card_selector_updated target_card_index=%d target_point=%s require_connector=%s order_axis=%s order_direction=%s expected_card_count=%d estimated_slot_requires_complete=%s",
            next_selector.target_card_index,
            next_selector.target_point,
            str(next_selector.require_connector).lower(),
            next_selector.order_axis,
            next_selector.order_direction,
            next_selector.expected_card_count,
            str(next_selector.estimated_slot_requires_complete).lower(),
        )

    def _uses_usb_card_detector(self) -> bool:
        return self._detection_source in ("usb_card", "usb_card_detector", "card")

    def _target_label(self) -> str:
        return "usb_card_target" if self._uses_usb_card_detector() else "green_blob"

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

    def _start_frame_debug_run(self, started_at: rospy.Time) -> None:
        self._frame_debug_count = 0
        self._frame_debug_last_save = rospy.Time(0)
        self._frame_debug_run_dir = ""
        if not self._frame_debug_enabled:
            return
        stamp = "%s_%09d" % (time.strftime("%Y%m%d_%H%M%S"), started_at.nsecs)
        self._frame_debug_run_dir = os.path.join(self._frame_debug_output_dir, "looming_%s" % stamp)
        try:
            os.makedirs(self._frame_debug_run_dir, exist_ok=True)
        except OSError as exc:
            rospy.logwarn(
                "[usb_c_insertion] event=verify_looming_frame_debug_dir_failed dir=%s error=%s disabling=true",
                self._frame_debug_run_dir,
                exc,
            )
            self._frame_debug_run_dir = ""
            return
        rospy.loginfo(
            "[usb_c_insertion] event=verify_looming_frame_debug_started dir=%s rate_hz=%.2f max_frames=%d",
            self._frame_debug_run_dir,
            self._frame_debug_rate_hz,
            self._frame_debug_max_frames,
        )

    def _save_frame_debug_image(
        self,
        stage: str,
        detection: Optional[GreenBlobDetection],
        initial_detection: Optional[GreenBlobDetection],
    ) -> None:
        if not self._frame_debug_enabled or not self._frame_debug_run_dir:
            return
        if self._frame_debug_max_frames > 0 and self._frame_debug_count >= self._frame_debug_max_frames:
            return
        now = rospy.Time.now()
        min_interval = 1.0 / max(0.1, self._frame_debug_rate_hz)
        if self._frame_debug_count > 0 and (now - self._frame_debug_last_save).to_sec() < min_interval:
            return

        image_stamp = rospy.Time(0)
        bgr = detection.bgr if detection is not None and detection.bgr is not None else None
        if bgr is None:
            with self._lock:
                bgr = None if self._latest_bgr is None else self._latest_bgr.copy()
                image_stamp = self._latest_bgr_stamp
        else:
            bgr = bgr.copy()
            image_stamp = detection.stamp
        if bgr is None:
            rospy.loginfo_throttle(2.0, "[usb_c_insertion] event=verify_looming_frame_debug_skipped reason=missing_image")
            return
        if (
            detection is not None
            and detection.found
            and image_stamp != rospy.Time(0)
            and detection.stamp != rospy.Time(0)
        ):
            image_detection_dt = abs((image_stamp - detection.stamp).to_sec())
            if (
                self._frame_debug_max_image_detection_dt > 0.0
                and image_detection_dt > self._frame_debug_max_image_detection_dt
            ):
                rospy.loginfo_throttle(
                    2.0,
                    "[usb_c_insertion] event=verify_looming_frame_debug_skipped reason=image_detection_dt_too_large dt=%.3f max_dt=%.3f",
                    image_detection_dt,
                    self._frame_debug_max_image_detection_dt,
                )
                return

        mask = self._frame_debug_mask(bgr.shape[:2], detection)
        debug = bgr.copy()
        if mask is not None and np.any(mask > 0):
            tint = debug.copy()
            tint[mask > 0] = (255, 0, 255)
            debug = cv2.addWeighted(debug, 0.72, tint, 0.28, 0.0)
            contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(debug, contours, -1, (255, 0, 255), 2)
            self._put_label(debug, "Port mask", 12.0, 22.0, (255, 0, 255))

        height, width = debug.shape[:2]
        self._draw_cross(debug, width * 0.5, height * 0.5, (255, 255, 0), 10, 2)
        self._put_label(debug, "Image center", 12.0, 42.0, (255, 255, 0))

        self._frame_debug_count += 1
        self._frame_debug_last_save = now
        filename = "frame_%03d_%s.png" % (self._frame_debug_count, self._safe_filename_fragment(stage))
        path = os.path.join(self._frame_debug_run_dir, filename)
        if not cv2.imwrite(path, debug):
            rospy.logwarn("[usb_c_insertion] event=verify_looming_frame_debug_write_failed path=%s", path)

    def _frame_debug_mask(self, image_shape, detection: Optional[GreenBlobDetection]) -> Optional[np.ndarray]:
        height, width = image_shape[:2]
        if detection is None or not detection.found:
            return None
        if detection.mask is not None and detection.mask.shape[:2] == (height, width):
            return detection.mask.astype(np.uint8)

        mask = np.zeros((height, width), dtype=np.uint8)
        if detection.bbox is not None:
            x, y, box_width, box_height = detection.bbox
            x0 = max(0, min(width - 1, int(x)))
            y0 = max(0, min(height - 1, int(y)))
            x1 = max(x0 + 1, min(width, int(x + box_width)))
            y1 = max(y0 + 1, min(height, int(y + box_height)))
            mask[y0:y1, x0:x1] = 255
            return mask

        radius = max(4, int(round(math.sqrt(max(1.0, float(detection.area))) * 0.08)))
        cv2.circle(
            mask,
            (int(round(detection.center_x)), int(round(detection.center_y))),
            radius,
            255,
            thickness=-1,
        )
        return mask

    @staticmethod
    def _put_label(image: np.ndarray, text: str, x: float, y: float, color) -> None:
        px = max(4, min(image.shape[1] - 4, int(round(float(x)))))
        py = max(16, min(image.shape[0] - 4, int(round(float(y)))))
        cv2.putText(
            image,
            text,
            (px, py),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (0, 0, 0),
            3,
            cv2.LINE_AA,
        )
        cv2.putText(
            image,
            text,
            (px, py),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            color,
            1,
            cv2.LINE_AA,
        )

    def _log_foe_debug_analysis(
        self,
        success: bool,
        message: str,
        initial_detection: Optional[GreenBlobDetection],
        current_detection: Optional[GreenBlobDetection],
    ) -> None:
        if not self._foe_debug_enabled:
            return
        if initial_detection is None or current_detection is None:
            rospy.loginfo("[usb_c_insertion] event=verify_looming_foe_skipped reason=missing_detection")
            return
        if initial_detection.gray is None or current_detection.gray is None:
            rospy.loginfo("[usb_c_insertion] event=verify_looming_foe_skipped reason=missing_image")
            return
        if current_detection.bgr is None or current_detection.mask is None:
            rospy.loginfo("[usb_c_insertion] event=verify_looming_foe_skipped reason=missing_current_mask")
            return
        if initial_detection.gray.shape != current_detection.gray.shape:
            rospy.logwarn(
                "[usb_c_insertion] event=verify_looming_foe_skipped reason=image_size_changed initial_shape=%s current_shape=%s",
                str(initial_detection.gray.shape),
                str(current_detection.gray.shape),
            )
            return

        mask = current_detection.mask > 0
        mask_pixel_count = int(np.count_nonzero(mask))
        if mask_pixel_count < 8:
            rospy.loginfo(
                "[usb_c_insertion] event=verify_looming_foe_skipped reason=mask_too_small mask_pixels=%d",
                mask_pixel_count,
            )
            return

        flow = cv2.calcOpticalFlowFarneback(
            initial_detection.gray,
            current_detection.gray,
            None,
            0.5,
            3,
            25,
            3,
            5,
            1.2,
            0,
        )
        y_coords, x_coords = np.nonzero(mask)
        vectors = flow[y_coords, x_coords, :]
        magnitudes = np.linalg.norm(vectors, axis=1)
        finite = np.isfinite(magnitudes)
        significant = finite & (magnitudes >= 0.05)
        point_count = int(np.count_nonzero(significant))
        if point_count < 2:
            debug_path = self._write_foe_debug_image(
                current_detection,
                mask,
                flow,
                None,
                message,
                initial_detection,
            )
            rospy.loginfo(
                "[usb_c_insertion] event=verify_looming_foe_analysis success=%s reason=%s mask_pixels=%d flow_points=%d mean_flow_x=0.000 mean_flow_y=0.000 mean_flow_mag=0.000 median_flow_mag=0.000 max_flow_mag=0.000 foe_x=nan foe_y=nan foe_rms_px=nan foe_to_marker_center_px=nan foe_to_image_center_px=nan radial_mean_cos=nan outward_ratio=nan debug_image=%s",
                str(success).lower(),
                message,
                mask_pixel_count,
                point_count,
                debug_path,
            )
            return

        points = np.column_stack((x_coords[significant], y_coords[significant])).astype(np.float32)
        vectors = vectors[significant].astype(np.float32)
        magnitudes = magnitudes[significant].astype(np.float32)
        mean_flow = np.mean(vectors, axis=0)
        mean_magnitude = float(np.mean(magnitudes))
        median_magnitude = float(np.median(magnitudes))
        max_magnitude = float(np.max(magnitudes))
        foe, foe_rms = self._estimate_focus_of_expansion(points, vectors)
        radial_mean_cos = float("nan")
        outward_ratio = float("nan")
        if foe is not None:
            radial = points - foe.reshape((1, 2))
            radial_norm = np.linalg.norm(radial, axis=1)
            vector_norm = np.linalg.norm(vectors, axis=1)
            valid_radial = (radial_norm > 1e-6) & (vector_norm > 1e-6)
            if np.any(valid_radial):
                cosine = np.sum(radial[valid_radial] * vectors[valid_radial], axis=1)
                cosine = cosine / (radial_norm[valid_radial] * vector_norm[valid_radial])
                radial_mean_cos = float(np.mean(cosine))
                outward_ratio = float(np.count_nonzero(cosine > 0.0)) / float(cosine.size)

        debug_path = self._write_foe_debug_image(
            current_detection,
            mask,
            flow,
            foe,
            message,
            initial_detection,
        )
        foe_x = float(foe[0]) if foe is not None else float("nan")
        foe_y = float(foe[1]) if foe is not None else float("nan")
        foe_to_marker_center = float("nan")
        foe_to_image_center = float("nan")
        if foe is not None:
            marker_dx = foe_x - float(current_detection.center_x)
            marker_dy = foe_y - float(current_detection.center_y)
            image_dx = foe_x - (float(current_detection.image_width) * 0.5)
            image_dy = foe_y - (float(current_detection.image_height) * 0.5)
            foe_to_marker_center = math.sqrt(marker_dx * marker_dx + marker_dy * marker_dy)
            foe_to_image_center = math.sqrt(image_dx * image_dx + image_dy * image_dy)
        rospy.loginfo(
            "[usb_c_insertion] event=verify_looming_foe_analysis success=%s reason=%s mask_pixels=%d flow_points=%d mean_flow_x=%.3f mean_flow_y=%.3f mean_flow_mag=%.3f median_flow_mag=%.3f max_flow_mag=%.3f foe_x=%.2f foe_y=%.2f foe_rms_px=%.3f foe_to_marker_center_px=%.2f foe_to_image_center_px=%.2f radial_mean_cos=%.3f outward_ratio=%.3f debug_image=%s",
            str(success).lower(),
            message,
            mask_pixel_count,
            point_count,
            float(mean_flow[0]),
            float(mean_flow[1]),
            mean_magnitude,
            median_magnitude,
            max_magnitude,
            foe_x,
            foe_y,
            foe_rms,
            foe_to_marker_center,
            foe_to_image_center,
            radial_mean_cos,
            outward_ratio,
            debug_path,
        )

    @staticmethod
    def _estimate_focus_of_expansion(points: np.ndarray, vectors: np.ndarray):
        if points.shape[0] < 2:
            return None, float("nan")
        normals = np.column_stack((-vectors[:, 1], vectors[:, 0])).astype(np.float64)
        normal_norm = np.linalg.norm(normals, axis=1)
        valid = normal_norm > 1e-6
        if np.count_nonzero(valid) < 2:
            return None, float("nan")
        normals = normals[valid] / normal_norm[valid].reshape((-1, 1))
        points = points[valid].astype(np.float64)
        rhs = np.sum(normals * points, axis=1)
        try:
            foe, _, rank, _ = np.linalg.lstsq(normals, rhs, rcond=None)
        except np.linalg.LinAlgError:
            return None, float("nan")
        if rank < 2:
            return None, float("nan")
        residual = normals.dot(foe) - rhs
        return foe.astype(np.float32), float(math.sqrt(np.mean(residual * residual)))

    def _write_foe_debug_image(
        self,
        current_detection: GreenBlobDetection,
        mask: np.ndarray,
        flow: np.ndarray,
        foe,
        message: str,
        initial_detection: GreenBlobDetection,
    ) -> str:
        if current_detection.bgr is None:
            return ""
        debug = current_detection.bgr.copy()
        mask_bool = mask.astype(bool)
        if np.any(mask_bool):
            tint = debug.copy()
            tint[mask_bool] = (0, 160, 0)
            debug = cv2.addWeighted(debug, 0.72, tint, 0.28, 0.0)
            contours, _ = cv2.findContours(mask.astype(np.uint8) * 255, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(debug, contours, -1, (0, 255, 0), 2)

        height, width = mask.shape[:2]
        vector_stride = 16
        vector_scale = 4.0
        max_vectors = 260
        drawn = 0
        for y in range(vector_stride // 2, height, vector_stride):
            for x in range(vector_stride // 2, width, vector_stride):
                if not mask_bool[y, x]:
                    continue
                dx = float(flow[y, x, 0])
                dy = float(flow[y, x, 1])
                if not math.isfinite(dx) or not math.isfinite(dy):
                    continue
                if math.sqrt(dx * dx + dy * dy) < 0.05:
                    continue
                start = (int(x), int(y))
                end = (int(round(x + dx * vector_scale)), int(round(y + dy * vector_scale)))
                cv2.arrowedLine(debug, start, end, (0, 255, 255), 1, tipLength=0.35)
                drawn += 1
                if drawn >= max_vectors:
                    break
            if drawn >= max_vectors:
                break

        self._draw_cross(debug, initial_detection.center_x, initial_detection.center_y, (255, 180, 0), 8, 1)
        self._draw_cross(debug, current_detection.center_x, current_detection.center_y, (255, 0, 255), 8, 1)
        self._draw_cross(debug, width * 0.5, height * 0.5, (255, 255, 0), 8, 1)
        if foe is not None:
            self._draw_cross(debug, float(foe[0]), float(foe[1]), (0, 0, 255), 12, 2)
            cv2.putText(
                debug,
                "FOE",
                (int(round(float(foe[0]) + 8)), int(round(float(foe[1]) - 8))),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (0, 0, 255),
                1,
                cv2.LINE_AA,
            )
        cv2.putText(
            debug,
            "yellow: optical flow  orange: initial center  magenta: current center  cyan: image center",
            (10, max(20, height - 14)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

        os.makedirs(self._foe_debug_output_dir, exist_ok=True)
        now = rospy.Time.now()
        filename = "foe_%s_%s.png" % (
            "%s_%09d" % (time.strftime("%Y%m%d_%H%M%S"), now.nsecs),
            self._safe_filename_fragment(message),
        )
        path = os.path.join(self._foe_debug_output_dir, filename)
        if not cv2.imwrite(path, debug):
            rospy.logwarn("[usb_c_insertion] event=verify_looming_foe_debug_write_failed path=%s", path)
            return ""
        return path

    @staticmethod
    def _draw_cross(image: np.ndarray, x: float, y: float, color, radius: int, thickness: int) -> None:
        cx = int(round(float(x)))
        cy = int(round(float(y)))
        cv2.line(image, (cx - radius, cy), (cx + radius, cy), color, thickness)
        cv2.line(image, (cx, cy - radius), (cx, cy + radius), color, thickness)

    @staticmethod
    def _safe_filename_fragment(text: str) -> str:
        cleaned = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in str(text).strip())
        return cleaned[:48] or "looming"

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
        try:
            self._log_foe_debug_analysis(success, message, initial_detection, current_detection)
        except Exception as exc:
            rospy.logwarn("[usb_c_insertion] event=verify_looming_foe_analysis_failed error=%s", exc)
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
    def _read_hsv_param(param_name: str) -> Tuple[int, int, int]:
        value = required_vector_param(param_name)
        if not isinstance(value, (list, tuple)) or len(value) != 3:
            rospy.logwarn("[usb_c_insertion] event=verify_looming_invalid_hsv_param param=%s", param_name)
            raise ValueError("Invalid HSV ROS parameter: %s" % rospy.resolve_name(param_name))
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
