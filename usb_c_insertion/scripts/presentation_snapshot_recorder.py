#!/usr/bin/env python3

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import cv2
from geometry_msgs.msg import PoseStamped
import numpy as np
import rospy
from sensor_msgs.msg import CameraInfo, Image

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from prepose_planner import normalize_quaternion, quaternion_multiply, rotate_vector_by_quaternion
from tf_interface import TFInterface


@dataclass(frozen=True)
class CapturedImage:
    msg: Image
    view_bgr: np.ndarray


@dataclass(frozen=True)
class BlobCenter:
    x: float
    y: float
    area: float = 0.0


class PresentationSnapshotRecorder:
    """
    Temporary helper for presentation images from the pre-insertion workflow.

    The recorder is intentionally non-fatal: if a camera image, CameraInfo, TF,
    or marker detection is unavailable, it logs a warning and lets the robot
    workflow continue.
    """

    def __init__(self):
        self._enabled = self._bool_param("~presentation_snapshots/enabled", False)
        self._output_dir = self._str_param(
            "~presentation_snapshots/output_dir",
            "/tmp/usb_c_insertion_presentation_snapshots",
        )
        self._image_topic = self._str_param(
            "~presentation_snapshots/image_topic",
            self._str_param(
                "~center_port/image_topic",
                self._str_param("~align_housing_yaw/image_topic", ""),
            ),
        )
        self._camera_info_topic = self._str_param(
            "~presentation_snapshots/camera_info_topic",
            self._infer_camera_info_topic(self._image_topic),
        )
        self._camera_frame = self._str_param(
            "~presentation_snapshots/camera_frame",
            self._str_param("~workflow/refine_camera_frame", ""),
        )
        self._base_frame = self._str_param("~frames/base_frame", "")
        self._image_rotation_deg = self._normalize_image_rotation_deg(
            self._float_param(
                "~presentation_snapshots/image_rotation_deg",
                self._float_param("~center_port/image_rotation_deg", 0.0),
            )
        )
        self._wait_timeout = max(0.05, self._float_param("~presentation_snapshots/wait_timeout", 1.0))
        self._pose_axis_length = max(
            0.001,
            self._float_param("~presentation_snapshots/pose_axis_length", 0.025),
        )
        self._min_blob_area = max(
            0.0,
            self._float_param(
                "~presentation_snapshots/min_blob_area",
                self._float_param("~center_port/min_blob_area", 120.0),
            ),
        )
        self._hsv_lower = self._vector_param(
            "~presentation_snapshots/hsv_lower",
            self._vector_param("~center_port/hsv_lower", [35, 70, 40]),
        )
        self._hsv_upper = self._vector_param(
            "~presentation_snapshots/hsv_upper",
            self._vector_param("~center_port/hsv_upper", [90, 255, 255]),
        )
        self._morph_kernel_size = max(
            0,
            self._int_param(
                "~presentation_snapshots/morph_kernel_size",
                self._int_param("~center_port/morph_kernel_size", 5),
            ),
        )
        self._tf = TFInterface() if self._enabled else None

        if not self._enabled:
            rospy.loginfo("[usb_c_insertion] event=presentation_snapshots_disabled")
            return

        try:
            os.makedirs(self._output_dir, exist_ok=True)
        except OSError as exc:
            rospy.logwarn(
                "[usb_c_insertion] event=presentation_snapshot_output_dir_failed dir=%s error=%s",
                self._output_dir,
                exc,
            )
            self._enabled = False
            return

        rospy.loginfo(
            "[usb_c_insertion] event=presentation_snapshots_enabled dir=%s image_topic=%s camera_info_topic=%s rotation_deg=%.1f",
            self._output_dir,
            self._image_topic,
            self._camera_info_topic,
            self._image_rotation_deg,
        )

    def capture_port_pose_axes(self, filename: str, title: str, port_pose: PoseStamped) -> bool:
        if not self._enabled:
            return False
        try:
            captured = self._capture_image()
            if captured is None:
                return False
            annotated = captured.view_bgr.copy()
            axes = self._project_pose_axes(port_pose, captured.msg)
            if axes:
                self._draw_pose_axes(annotated, axes)
                subtitle = "Initial 6-DoF estimate"
            else:
                subtitle = "Initial estimate projection unavailable"
            self._draw_title(annotated, title, subtitle)
            return self._write_image(filename, annotated)
        except Exception as exc:
            rospy.logwarn(
                "[usb_c_insertion] event=presentation_snapshot_failed filename=%s error=%s",
                filename,
                exc,
            )
            return False

    def capture_current_view(self, filename: str, title: str) -> bool:
        if not self._enabled:
            return False
        try:
            captured = self._capture_image()
            if captured is None:
                return False
            annotated = captured.view_bgr.copy()
            self._draw_title(annotated, title, "")
            return self._write_image(filename, annotated)
        except Exception as exc:
            rospy.logwarn(
                "[usb_c_insertion] event=presentation_snapshot_failed filename=%s error=%s",
                filename,
                exc,
            )
            return False

    def capture_marker_alignment(
        self,
        filename: str,
        title: str,
        fallback_marker_center: Optional[Tuple[float, float]] = None,
    ) -> bool:
        if not self._enabled:
            return False
        try:
            captured = self._capture_image()
            if captured is None:
                return False
            annotated = captured.view_bgr.copy()
            detected_center = self._detect_green_blob_center(captured.view_bgr)
            marker_center = detected_center
            source = "detected"
            if marker_center is None and fallback_marker_center is not None:
                marker_center = BlobCenter(float(fallback_marker_center[0]), float(fallback_marker_center[1]))
                source = "action result"
            self._draw_image_center(annotated)
            if marker_center is not None:
                self._draw_blob_center(annotated, marker_center)
                subtitle = "Image center + circle center (%s)" % source
            else:
                subtitle = "Image center marked; circle center unavailable"
            self._draw_title(annotated, title, subtitle)
            return self._write_image(filename, annotated)
        except Exception as exc:
            rospy.logwarn(
                "[usb_c_insertion] event=presentation_snapshot_failed filename=%s error=%s",
                filename,
                exc,
            )
            return False

    @staticmethod
    def center_from_center_result(result) -> Optional[Tuple[float, float]]:
        if result is None:
            return None
        if float(getattr(result, "blob_area", 0.0)) <= 0.0:
            return None
        return (
            float(getattr(result, "blob_center_x", 0.0)),
            float(getattr(result, "blob_center_y", 0.0)),
        )

    @staticmethod
    def center_from_looming_result(result) -> Optional[Tuple[float, float]]:
        if result is None:
            return None
        if float(getattr(result, "current_area", 0.0)) <= 0.0:
            return None
        return (
            float(getattr(result, "current_center_x", 0.0)),
            float(getattr(result, "current_center_y", 0.0)),
        )

    def _capture_image(self) -> Optional[CapturedImage]:
        if not self._image_topic:
            rospy.logwarn("[usb_c_insertion] event=presentation_snapshot_skipped reason=image_topic_empty")
            return None
        try:
            msg = rospy.wait_for_message(self._image_topic, Image, timeout=self._wait_timeout)
        except rospy.ROSException as exc:
            rospy.logwarn(
                "[usb_c_insertion] event=presentation_snapshot_skipped reason=image_timeout topic=%s timeout=%.2f error=%s",
                self._image_topic,
                self._wait_timeout,
                exc,
            )
            return None

        raw_bgr = self._image_to_bgr(msg)
        view_bgr = self._rotate_image(raw_bgr)
        return CapturedImage(msg, view_bgr)

    def _camera_info(self) -> Optional[CameraInfo]:
        if not self._camera_info_topic:
            return None
        try:
            return rospy.wait_for_message(self._camera_info_topic, CameraInfo, timeout=self._wait_timeout)
        except rospy.ROSException as exc:
            rospy.logwarn(
                "[usb_c_insertion] event=presentation_snapshot_camera_info_unavailable topic=%s timeout=%.2f error=%s",
                self._camera_info_topic,
                self._wait_timeout,
                exc,
            )
            return None

    def _project_pose_axes(self, pose: PoseStamped, image_msg: Image) -> Optional[Dict[str, Tuple[float, float]]]:
        camera_info = self._camera_info()
        if camera_info is None or self._tf is None:
            return None

        camera_frame = self._camera_frame or camera_info.header.frame_id.strip()
        source_frame = pose.header.frame_id.strip() or self._base_frame
        if not camera_frame or not source_frame:
            return None

        transform = self._tf.lookup_transform(camera_frame, source_frame)
        if transform is None:
            return None

        transform_quaternion = normalize_quaternion(
            (
                transform.transform.rotation.x,
                transform.transform.rotation.y,
                transform.transform.rotation.z,
                transform.transform.rotation.w,
            )
        )
        transform_translation = (
            transform.transform.translation.x,
            transform.transform.translation.y,
            transform.transform.translation.z,
        )
        origin_in_source = (
            pose.pose.position.x,
            pose.pose.position.y,
            pose.pose.position.z,
        )
        origin_in_camera = self._transform_point(origin_in_source, transform_translation, transform_quaternion)
        pose_quaternion = normalize_quaternion(
            (
                pose.pose.orientation.x,
                pose.pose.orientation.y,
                pose.pose.orientation.z,
                pose.pose.orientation.w,
            )
        )
        pose_quaternion_camera = normalize_quaternion(
            quaternion_multiply(transform_quaternion, pose_quaternion)
        )

        points_3d = {
            "origin": origin_in_camera,
            "x": self._axis_endpoint(origin_in_camera, pose_quaternion_camera, (self._pose_axis_length, 0.0, 0.0)),
            "y": self._axis_endpoint(origin_in_camera, pose_quaternion_camera, (0.0, self._pose_axis_length, 0.0)),
            "z": self._axis_endpoint(origin_in_camera, pose_quaternion_camera, (0.0, 0.0, self._pose_axis_length)),
        }

        projected = {}
        for name, point in points_3d.items():
            raw_pixel = self._project_point(point, camera_info)
            if raw_pixel is None:
                return None
            projected[name] = self._rotate_pixel(raw_pixel, int(image_msg.width), int(image_msg.height))
        return projected

    @staticmethod
    def _transform_point(point_xyz, translation_xyz, quaternion_xyzw):
        rotated = rotate_vector_by_quaternion(
            point_xyz[0],
            point_xyz[1],
            point_xyz[2],
            *quaternion_xyzw,
        )
        return (
            rotated[0] + translation_xyz[0],
            rotated[1] + translation_xyz[1],
            rotated[2] + translation_xyz[2],
        )

    @staticmethod
    def _axis_endpoint(origin_xyz, quaternion_xyzw, axis_vector_xyz):
        rotated_axis = rotate_vector_by_quaternion(
            axis_vector_xyz[0],
            axis_vector_xyz[1],
            axis_vector_xyz[2],
            *quaternion_xyzw,
        )
        return (
            origin_xyz[0] + rotated_axis[0],
            origin_xyz[1] + rotated_axis[1],
            origin_xyz[2] + rotated_axis[2],
        )

    @staticmethod
    def _project_point(point_xyz, camera_info: CameraInfo) -> Optional[Tuple[float, float]]:
        x, y, z = point_xyz
        if z <= 1e-6:
            return None
        fx = float(camera_info.K[0]) if len(camera_info.K) >= 9 else 0.0
        fy = float(camera_info.K[4]) if len(camera_info.K) >= 9 else 0.0
        cx = float(camera_info.K[2]) if len(camera_info.K) >= 9 else 0.0
        cy = float(camera_info.K[5]) if len(camera_info.K) >= 9 else 0.0
        if fx <= 0.0 or fy <= 0.0:
            fx = float(camera_info.P[0]) if len(camera_info.P) >= 12 else 0.0
            fy = float(camera_info.P[5]) if len(camera_info.P) >= 12 else 0.0
            cx = float(camera_info.P[2]) if len(camera_info.P) >= 12 else 0.0
            cy = float(camera_info.P[6]) if len(camera_info.P) >= 12 else 0.0
        if fx <= 0.0 or fy <= 0.0:
            return None
        return (fx * x / z + cx, fy * y / z + cy)

    def _detect_green_blob_center(self, bgr: np.ndarray) -> Optional[BlobCenter]:
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

        label_count, _, stats, centroids = cv2.connectedComponentsWithStats(mask, 8)
        if label_count <= 1:
            return None
        areas = stats[1:, cv2.CC_STAT_AREA]
        largest_index = int(np.argmax(areas)) + 1
        area = float(stats[largest_index, cv2.CC_STAT_AREA])
        if area < self._min_blob_area:
            return None
        center_x, center_y = centroids[largest_index]
        return BlobCenter(float(center_x), float(center_y), area)

    def _draw_title(self, image: np.ndarray, title: str, subtitle: str) -> None:
        lines = [title]
        if subtitle:
            lines.append(subtitle)
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = 0.55
        thickness = 1
        margin = 10
        line_height = 22
        width = max(cv2.getTextSize(line, font, scale, thickness)[0][0] for line in lines)
        box_right = min(image.shape[1] - 1, margin * 2 + width + 10)
        box_bottom = min(image.shape[0] - 1, margin * 2 + line_height * len(lines))
        cv2.rectangle(image, (margin, margin), (box_right, box_bottom), (0, 0, 0), -1)
        for index, line in enumerate(lines):
            y = margin + 17 + index * line_height
            cv2.putText(image, line, (margin + 7, y), font, scale, (255, 255, 255), thickness, cv2.LINE_AA)

    def _draw_pose_axes(self, image: np.ndarray, axes: Dict[str, Tuple[float, float]]) -> None:
        origin = self._point_tuple(axes["origin"])
        colors = {
            "x": (0, 0, 255),
            "y": (0, 255, 0),
            "z": (255, 0, 0),
        }
        for axis_name in ("x", "y", "z"):
            endpoint = self._point_tuple(axes[axis_name])
            cv2.arrowedLine(image, origin, endpoint, colors[axis_name], 2, cv2.LINE_AA, tipLength=0.22)
            self._draw_label(image, axis_name.upper(), endpoint, colors[axis_name])
        cv2.circle(image, origin, 6, (255, 255, 255), -1, cv2.LINE_AA)
        cv2.circle(image, origin, 6, (0, 0, 0), 1, cv2.LINE_AA)
        self._draw_label(image, "Port estimate", (origin[0] + 9, origin[1] - 9), (255, 255, 255))

    def _draw_image_center(self, image: np.ndarray) -> None:
        center = (
            int(round(0.5 * float(max(0, image.shape[1] - 1)))),
            int(round(0.5 * float(max(0, image.shape[0] - 1)))),
        )
        self._draw_cross(image, center, (255, 255, 0), 15, 1)
        cv2.circle(image, center, 12, (255, 255, 0), 1, cv2.LINE_AA)
        self._draw_label(image, "Image center", (center[0] + 12, center[1] - 12), (255, 255, 0))

    def _draw_blob_center(self, image: np.ndarray, center: BlobCenter) -> None:
        point = self._point_tuple((center.x, center.y))
        self._draw_cross(image, point, (255, 0, 255), 15, 1)
        cv2.circle(image, point, 13, (255, 0, 255), 1, cv2.LINE_AA)
        self._draw_label(image, "Circle center", (point[0] + 12, point[1] + 22), (255, 0, 255))

    @staticmethod
    def _draw_cross(image: np.ndarray, center: Tuple[int, int], color, half_size: int, thickness: int) -> None:
        x, y = center
        cv2.line(image, (x - half_size, y), (x + half_size, y), color, thickness, cv2.LINE_AA)
        cv2.line(image, (x, y - half_size), (x, y + half_size), color, thickness, cv2.LINE_AA)

    @staticmethod
    def _draw_label(image: np.ndarray, text: str, origin: Tuple[int, int], color) -> None:
        x = max(6, min(image.shape[1] - 120, int(origin[0])))
        y = max(18, min(image.shape[0] - 8, int(origin[1])))
        cv2.putText(image, text, (x + 1, y + 1), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 2, cv2.LINE_AA)
        cv2.putText(image, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)

    def _write_image(self, filename: str, image: np.ndarray) -> bool:
        path = os.path.join(self._output_dir, filename)
        if not cv2.imwrite(path, image):
            rospy.logwarn("[usb_c_insertion] event=presentation_snapshot_write_failed path=%s", path)
            return False
        rospy.loginfo("[usb_c_insertion] event=presentation_snapshot_saved path=%s", path)
        return True

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
        elif encoding in ("mono8", "8uc1"):
            channels = 1
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
        if encoding == "rgba8":
            return cv2.cvtColor(image, cv2.COLOR_RGBA2BGR)
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    def _rotate_image(self, bgr: np.ndarray) -> np.ndarray:
        if self._image_rotation_deg == 0.0:
            return bgr.copy()
        if self._image_rotation_deg == 90.0:
            return cv2.rotate(bgr, cv2.ROTATE_90_CLOCKWISE)
        if self._image_rotation_deg == 180.0:
            return cv2.rotate(bgr, cv2.ROTATE_180)
        return cv2.rotate(bgr, cv2.ROTATE_90_COUNTERCLOCKWISE)

    def _rotate_pixel(self, pixel: Tuple[float, float], raw_width: int, raw_height: int) -> Tuple[float, float]:
        u, v = pixel
        if self._image_rotation_deg == 0.0:
            return (u, v)
        if self._image_rotation_deg == 90.0:
            return (float(raw_height - 1) - v, u)
        if self._image_rotation_deg == 180.0:
            return (float(raw_width - 1) - u, float(raw_height - 1) - v)
        return (v, float(raw_width - 1) - u)

    @staticmethod
    def _point_tuple(point: Tuple[float, float]) -> Tuple[int, int]:
        return (int(round(point[0])), int(round(point[1])))

    @staticmethod
    def _normalize_image_rotation_deg(rotation_deg: float) -> float:
        normalized = float(rotation_deg) % 360.0
        allowed = (0.0, 90.0, 180.0, 270.0)
        return min(allowed, key=lambda value: abs(value - normalized))

    @staticmethod
    def _infer_camera_info_topic(image_topic: str) -> str:
        topic = image_topic.strip()
        suffixes = (
            "/image_rect_color",
            "/image_rect",
            "/image_raw",
            "/image_color",
        )
        for suffix in suffixes:
            if topic.endswith(suffix):
                return topic[: -len(suffix)] + "/camera_info"
        return ""

    @classmethod
    def _param(cls, name: str, default):
        if name.startswith("~"):
            global_name = "/" + name[1:].lstrip("/")
            if rospy.has_param(global_name):
                return rospy.get_param(global_name)
        if rospy.has_param(name):
            return rospy.get_param(name)
        return default

    @classmethod
    def _str_param(cls, name: str, default: str) -> str:
        return str(cls._param(name, default)).strip()

    @classmethod
    def _float_param(cls, name: str, default: float) -> float:
        return float(cls._param(name, default))

    @classmethod
    def _int_param(cls, name: str, default: int) -> int:
        return int(cls._param(name, default))

    @classmethod
    def _bool_param(cls, name: str, default: bool) -> bool:
        value = cls._param(name, default)
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            return value.strip().lower() in ("true", "1", "yes", "on")
        return bool(value)

    @classmethod
    def _vector_param(cls, name: str, default):
        value = cls._param(name, default)
        if not isinstance(value, (list, tuple)) or len(value) != 3:
            return tuple(int(component) for component in default)
        return tuple(max(0, min(255, int(component))) for component in value)
