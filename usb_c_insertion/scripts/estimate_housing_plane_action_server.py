#!/usr/bin/env python3

from __future__ import annotations

import math
import os
import random
import sys
from dataclasses import dataclass
from threading import RLock
from typing import Optional, Tuple

import actionlib
import cv2
from geometry_msgs.msg import PointStamped, Vector3
import numpy as np
import rospy
from sensor_msgs.msg import Image, PointCloud2
import sensor_msgs.point_cloud2 as point_cloud2
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
from tf_interface import TFInterface
from usb_card_target_selector import UsbCardTargetSelector
from usb_c_insertion.msg import (
    EstimateHousingPlaneAction,
    EstimateHousingPlaneFeedback,
    EstimateHousingPlaneResult,
)

CAMERA_FRAME_TO_OPTICAL_QUATERNION = (-0.5, 0.5, -0.5, -0.5)


@dataclass(frozen=True)
class GreenBlobDetection:
    stamp: rospy.Time
    image_width: int
    image_height: int
    found: bool
    center_x: float = 0.0
    center_y: float = 0.0
    area: float = 0.0
    message: str = ""


@dataclass(frozen=True)
class PlaneEstimate:
    point: np.ndarray
    normal: np.ndarray
    inlier_mask: np.ndarray
    rms_error: float
    max_error: float


class EstimateHousingPlaneActionServer:
    """
    Estimate the local housing wall plane around the green port marker.

    The server uses the green marker in the RGB image to choose a registered
    point-cloud ROI, filters depth outliers, fits a robust RANSAC plane, and
    optionally refits that plane with SVD on the inliers.
    """

    def __init__(self):
        self._action_name = required_str_param("~housing_plane/action_name")
        self._default_image_topic = required_str_param("~housing_plane/image_topic")
        self._default_cloud_topic = required_str_param("~housing_plane/cloud_topic")
        self._base_frame = required_str_param("~frames/base_frame")
        self._base_transform_frame = required_str_param("~housing_plane/base_transform_frame")
        self._cloud_to_base_transform_rotation = required_str_param(
            "~housing_plane/cloud_to_base_transform_rotation"
        ).lower()
        self._command_rate = max(1.0, required_float_param("~housing_plane/command_rate"))
        self._default_timeout = required_float_param("~housing_plane/timeout")
        self._default_min_blob_area = required_float_param("~housing_plane/min_blob_area")
        self._default_roi_radius_px = required_int_param("~housing_plane/roi_radius_px")
        self._default_roi_stride_px = required_int_param("~housing_plane/roi_stride_px")
        self._default_depth_window_m = required_float_param("~housing_plane/depth_window_m")
        self._default_ransac_iterations = required_int_param("~housing_plane/ransac_iterations")
        self._default_ransac_distance_threshold = required_float_param(
            "~housing_plane/ransac_distance_threshold"
        )
        self._default_min_inliers = required_int_param("~housing_plane/min_inliers")
        self._default_use_svd_refit = required_bool_param("~housing_plane/use_svd_refit")
        self._default_use_largest_component = required_bool_param("~housing_plane/use_largest_component")
        self._marker_ray_window_px = max(0, required_int_param("~housing_plane/marker_ray_window_px"))
        self._max_image_age = required_float_param("~housing_plane/max_image_age")
        self._max_cloud_age = required_float_param("~housing_plane/max_cloud_age")
        self._image_rotation_deg = self._normalize_image_rotation_deg(
            required_float_param("~housing_plane/image_rotation_deg")
        )
        self._hsv_lower = self._read_hsv_param("~housing_plane/hsv_lower")
        self._hsv_upper = self._read_hsv_param("~housing_plane/hsv_upper")
        self._morph_kernel_size = max(0, required_int_param("~housing_plane/morph_kernel_size"))
        self._detection_source = str(get_param("~housing_plane/detection_source", "green_marker")).strip().lower()
        self._usb_card_detections_topic = str(
            get_param(
                "~housing_plane/usb_card_detections_topic",
                get_param("~usb_card_detector/detections_topic", "/usb_c_insertion/usb_card_detector/detections"),
            )
        ).strip()
        self._usb_card_selector = UsbCardTargetSelector.from_ros_params("housing_plane")

        self._tf = TFInterface()
        self._lock = RLock()
        self._image_topic = ""
        self._cloud_topic = ""
        self._image_subscriber = None
        self._cloud_subscriber = None
        self._usb_card_detections_subscriber = None
        self._latest_detection: Optional[GreenBlobDetection] = None
        self._latest_cloud: Optional[PointCloud2] = None

        self._subscribe_image_topic(self._default_image_topic)
        self._subscribe_cloud_topic(self._default_cloud_topic)
        self._subscribe_usb_card_detections()
        self._server = actionlib.SimpleActionServer(
            self._action_name,
            EstimateHousingPlaneAction,
            execute_cb=self._execute,
            auto_start=False,
        )
        self._server.start()
        rospy.loginfo(
            "[usb_c_insertion] event=estimate_housing_plane_action_ready action=%s image_topic=%s cloud_topic=%s detection_source=%s usb_card_detections_topic=%s target_card_index=%d base_transform_frame=%s cloud_to_base_transform_rotation=%s image_rotation_deg=%.1f",
            self._action_name,
            self._image_topic,
            self._cloud_topic,
            self._detection_source,
            self._usb_card_detections_topic,
            self._usb_card_selector.target_card_index,
            self._base_transform_frame or "<cloud_frame>",
            self._cloud_to_base_transform_rotation,
            self._image_rotation_deg,
        )

    def _execute(self, goal) -> None:
        started_at = rospy.Time.now()
        image_topic = str(goal.image_topic).strip() or self._default_image_topic
        cloud_topic = str(goal.cloud_topic).strip() or self._default_cloud_topic
        self._refresh_usb_card_selector()
        self._subscribe_image_topic(image_topic)
        self._subscribe_cloud_topic(cloud_topic)
        self._subscribe_usb_card_detections()

        timeout = self._goal_or_default(goal.timeout, self._default_timeout)
        min_blob_area = self._goal_or_default(goal.min_blob_area, self._default_min_blob_area)
        roi_radius_px = self._goal_int_or_default(goal.roi_radius_px, self._default_roi_radius_px)
        roi_stride_px = max(1, self._goal_int_or_default(goal.roi_stride_px, self._default_roi_stride_px))
        depth_window_m = self._goal_or_default(goal.depth_window_m, self._default_depth_window_m)
        ransac_iterations = self._goal_int_or_default(goal.ransac_iterations, self._default_ransac_iterations)
        ransac_threshold = self._goal_or_default(
            goal.ransac_distance_threshold,
            self._default_ransac_distance_threshold,
        )
        min_inliers = self._goal_int_or_default(goal.min_inliers, self._default_min_inliers)
        use_svd_refit = bool(goal.use_svd_refit)
        use_largest_component = bool(goal.use_largest_component)

        deadline = started_at + rospy.Duration.from_sec(max(0.1, timeout))
        detection, cloud = self._wait_for_inputs(deadline, min_blob_area)
        if detection is None:
            self._abort("%s_not_found" % self._target_label(), started_at)
            return
        if cloud is None:
            self._abort("cloud_not_available", started_at, detection)
            return
        if cloud.height <= 1:
            self._abort("point_cloud_not_organized", started_at, detection, cloud)
            return

        raw_u, raw_v = self._rotated_to_raw_pixel(
            detection.center_x,
            detection.center_y,
            detection.image_width,
            detection.image_height,
            cloud.width,
            cloud.height,
        )
        point_grid, roi_mask = self._collect_roi_points(
            cloud,
            raw_u,
            raw_v,
            roi_radius_px,
            roi_stride_px,
        )
        roi_point_count = int(np.count_nonzero(roi_mask))
        if roi_point_count < min_inliers:
            self._abort("not_enough_roi_points", started_at, detection, cloud, raw_u, raw_v, roi_point_count)
            return

        filtered_points, filtered_mask = self._filter_depth_window(point_grid, roi_mask, depth_window_m)
        if use_largest_component:
            filtered_points, filtered_mask = self._filter_largest_component(point_grid, filtered_mask)
        if filtered_points.shape[0] < min_inliers:
            self._abort(
                "not_enough_filtered_points",
                started_at,
                detection,
                cloud,
                raw_u,
                raw_v,
                roi_point_count,
                filtered_points.shape[0],
            )
            return

        estimate = self._fit_plane_ransac(
            filtered_points,
            iterations=ransac_iterations,
            distance_threshold=ransac_threshold,
            min_inliers=min_inliers,
            use_svd_refit=use_svd_refit,
        )
        if estimate is None:
            self._abort(
                "plane_fit_failed",
                started_at,
                detection,
                cloud,
                raw_u,
                raw_v,
                roi_point_count,
                filtered_points.shape[0],
            )
            return

        estimate = self._orient_normal_toward_camera(estimate)
        marker_plane_point = self._estimate_marker_plane_point(cloud, raw_u, raw_v, estimate)
        base_lookup_frame = self._base_lookup_frame(cloud.header.frame_id)
        if base_lookup_frame != self._base_frame:
            transform = self._tf.lookup_transform(self._base_frame, base_lookup_frame)
            if transform is None:
                self._abort(
                    "plane_base_tf_unavailable",
                    started_at,
                    detection,
                    cloud,
                    raw_u,
                    raw_v,
                    roi_point_count,
                    filtered_points.shape[0],
                )
                return
        result = self._make_result(
            True,
            "housing_plane_estimated",
            "",
            started_at,
            detection,
            cloud,
            raw_u,
            raw_v,
            roi_point_count,
            filtered_points.shape[0],
            estimate,
            marker_plane_point,
        )
        self._publish_feedback(
            "complete",
            started_at,
            detection,
            raw_u,
            raw_v,
            roi_point_count,
            filtered_points.shape[0],
            estimate,
        )
        self._server.set_succeeded(result)
        rospy.loginfo(
            "[usb_c_insertion] event=estimate_housing_plane_complete cloud_frame=%s base_transform_frame=%s base_frame=%s inliers=%d filtered=%d ratio=%.3f rms_error=%.4f normal_cloud=(%.4f,%.4f,%.4f) normal_base=(%.4f,%.4f,%.4f)",
            cloud.header.frame_id,
            base_lookup_frame,
            result.plane_point_base.header.frame_id,
            int(np.count_nonzero(estimate.inlier_mask)),
            filtered_points.shape[0],
            float(np.count_nonzero(estimate.inlier_mask)) / max(1.0, float(filtered_points.shape[0])),
            estimate.rms_error,
            estimate.normal[0],
            estimate.normal[1],
            estimate.normal[2],
            result.plane_normal_base.x,
            result.plane_normal_base.y,
            result.plane_normal_base.z,
        )

    def _wait_for_inputs(
        self,
        deadline: rospy.Time,
        min_blob_area: float,
    ) -> Tuple[Optional[GreenBlobDetection], Optional[PointCloud2]]:
        rate = rospy.Rate(self._command_rate)
        last_detection = None
        last_cloud = None
        while not rospy.is_shutdown() and rospy.Time.now() < deadline:
            now = rospy.Time.now()
            with self._lock:
                detection = self._latest_detection
                cloud = self._latest_cloud
            if detection is not None and detection.found and detection.area >= min_blob_area:
                if not self._is_detection_stale(detection, now):
                    last_detection = detection
            if cloud is not None and not self._is_cloud_stale(cloud, now):
                last_cloud = cloud
            if last_detection is not None and last_cloud is not None:
                return last_detection, last_cloud
            feedback_detection = detection if detection is not None else last_detection
            if feedback_detection is not None:
                self._publish_feedback("waiting_for_inputs", rospy.Time.now(), feedback_detection, 0.0, 0.0, 0, 0, None)
            rate.sleep()
        return last_detection, last_cloud

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

    def _subscribe_cloud_topic(self, cloud_topic: str) -> None:
        topic = cloud_topic.strip() or self._default_cloud_topic
        with self._lock:
            if topic == self._cloud_topic and self._cloud_subscriber is not None:
                return
            if self._cloud_subscriber is not None:
                self._cloud_subscriber.unregister()
            self._cloud_topic = topic
            self._latest_cloud = None
            self._cloud_subscriber = rospy.Subscriber(topic, PointCloud2, self._cloud_callback, queue_size=1)

    def _image_callback(self, msg: Image) -> None:
        if self._uses_usb_card_detector():
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
            message=target.message,
        )
        with self._lock:
            self._latest_detection = detection

    def _cloud_callback(self, msg: PointCloud2) -> None:
        with self._lock:
            self._latest_cloud = msg

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
        next_selector = UsbCardTargetSelector.from_ros_params("housing_plane")
        previous_selector = self._usb_card_selector
        changed = (
            previous_selector.target_card_index != next_selector.target_card_index
            or previous_selector.target_point != next_selector.target_point
            or previous_selector.require_connector != next_selector.require_connector
            or previous_selector.order_axis != next_selector.order_axis
            or previous_selector.order_direction != next_selector.order_direction
        )
        if not changed:
            return
        with self._lock:
            self._usb_card_selector = next_selector
            self._latest_detection = None
        rospy.loginfo(
            "[usb_c_insertion] event=estimate_housing_plane_usb_card_selector_updated target_card_index=%d target_point=%s require_connector=%s order_axis=%s order_direction=%s",
            next_selector.target_card_index,
            next_selector.target_point,
            str(next_selector.require_connector).lower(),
            next_selector.order_axis,
            next_selector.order_direction,
        )

    def _uses_usb_card_detector(self) -> bool:
        return self._detection_source in ("usb_card", "usb_card_detector", "card")

    def _target_label(self) -> str:
        return "usb_card_target" if self._uses_usb_card_detector() else "marker"

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
        return self._make_detection(msg, True, float(center_x), float(center_y), area)

    def _make_detection(
        self,
        msg: Image,
        found: bool,
        center_x: float = 0.0,
        center_y: float = 0.0,
        area: float = 0.0,
        message: str = "",
    ) -> GreenBlobDetection:
        bgr_width, bgr_height = self._rotated_image_size(int(msg.width), int(msg.height))
        stamp = msg.header.stamp if msg.header.stamp != rospy.Time() else rospy.Time.now()
        return GreenBlobDetection(
            stamp=stamp,
            image_width=bgr_width,
            image_height=bgr_height,
            found=found,
            center_x=float(center_x),
            center_y=float(center_y),
            area=float(area),
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

    def _rotated_image_size(self, raw_width: int, raw_height: int) -> Tuple[int, int]:
        if self._image_rotation_deg in (90.0, 270.0):
            return raw_height, raw_width
        return raw_width, raw_height

    def _rotated_to_raw_pixel(
        self,
        rotated_u: float,
        rotated_v: float,
        rotated_width: int,
        rotated_height: int,
        raw_width: int,
        raw_height: int,
    ) -> Tuple[float, float]:
        if self._image_rotation_deg == 0.0:
            return rotated_u, rotated_v
        if self._image_rotation_deg == 90.0:
            return rotated_v, float(raw_height - 1) - rotated_u
        if self._image_rotation_deg == 180.0:
            return float(raw_width - 1) - rotated_u, float(raw_height - 1) - rotated_v
        return float(raw_width - 1) - rotated_v, rotated_u

    def _collect_roi_points(
        self,
        cloud: PointCloud2,
        center_u: float,
        center_v: float,
        roi_radius_px: int,
        roi_stride_px: int,
    ) -> Tuple[np.ndarray, np.ndarray]:
        center_u_int = int(round(center_u))
        center_v_int = int(round(center_v))
        radius = max(1, int(roi_radius_px))
        stride = max(1, int(roi_stride_px))
        u_min = max(0, center_u_int - radius)
        u_max = min(int(cloud.width) - 1, center_u_int + radius)
        v_min = max(0, center_v_int - radius)
        v_max = min(int(cloud.height) - 1, center_v_int + radius)
        roi_width = ((u_max - u_min) // stride) + 1
        roi_height = ((v_max - v_min) // stride) + 1

        uvs = []
        index_lookup = {}
        for row_index, v in enumerate(range(v_min, v_max + 1, stride)):
            for col_index, u in enumerate(range(u_min, u_max + 1, stride)):
                index_lookup[(u, v)] = (row_index, col_index)
                uvs.append((u, v))

        point_grid = np.full((roi_height, roi_width, 3), np.nan, dtype=np.float64)
        mask = np.zeros((roi_height, roi_width), dtype=np.uint8)
        for point, uv in zip(
            point_cloud2.read_points(cloud, field_names=("x", "y", "z"), skip_nans=False, uvs=uvs),
            uvs,
        ):
            x, y, z = point[:3]
            if not np.isfinite(x) or not np.isfinite(y) or not np.isfinite(z):
                continue
            if abs(x) <= 1e-9 and abs(y) <= 1e-9 and abs(z) <= 1e-9:
                continue
            row_index, col_index = index_lookup[uv]
            point_grid[row_index, col_index, :] = (float(x), float(y), float(z))
            mask[row_index, col_index] = 1

        return point_grid, mask

    def _filter_depth_window(self, point_grid: np.ndarray, roi_mask: np.ndarray, depth_window_m: float):
        valid_mask = roi_mask.astype(bool)
        if not np.any(valid_mask):
            return np.zeros((0, 3), dtype=np.float64), roi_mask

        z_values = point_grid[:, :, 2][valid_mask]
        median_z = float(np.median(z_values))
        depth_mask = valid_mask & (np.abs(point_grid[:, :, 2] - median_z) <= max(0.0, float(depth_window_m)))
        filtered_points = point_grid[depth_mask]
        return filtered_points, depth_mask.astype(np.uint8)

    def _filter_largest_component(self, point_grid: np.ndarray, roi_mask: np.ndarray):
        if not np.any(roi_mask) or roi_mask.size <= 1:
            return point_grid[roi_mask.astype(bool)], roi_mask
        label_count, labels, stats, _ = cv2.connectedComponentsWithStats(roi_mask.astype(np.uint8), 8)
        if label_count <= 1:
            return point_grid[roi_mask.astype(bool)], roi_mask
        largest_index = int(np.argmax(stats[1:, cv2.CC_STAT_AREA])) + 1
        component_area = int(stats[largest_index, cv2.CC_STAT_AREA])
        if component_area <= 0:
            return point_grid[roi_mask.astype(bool)], roi_mask
        component_mask = labels == largest_index
        return point_grid[component_mask], component_mask.astype(np.uint8)

    def _fit_plane_ransac(
        self,
        points: np.ndarray,
        iterations: int,
        distance_threshold: float,
        min_inliers: int,
        use_svd_refit: bool,
    ) -> Optional[PlaneEstimate]:
        if points.shape[0] < 3:
            return None

        best_mask = None
        best_point = None
        best_normal = None
        best_count = 0
        threshold = max(1e-5, float(distance_threshold))
        sample_count = int(points.shape[0])
        iterations = max(1, int(iterations))
        for _ in range(iterations):
            indices = random.sample(range(sample_count), 3)
            plane = self._plane_from_three_points(points[indices[0]], points[indices[1]], points[indices[2]])
            if plane is None:
                continue
            point, normal = plane
            distances = np.abs(np.dot(points - point, normal))
            mask = distances <= threshold
            count = int(np.count_nonzero(mask))
            if count > best_count:
                best_count = count
                best_mask = mask
                best_point = point
                best_normal = normal

        if best_mask is None or best_point is None or best_normal is None or best_count < int(min_inliers):
            return None

        inlier_points = points[best_mask]
        if use_svd_refit and inlier_points.shape[0] >= 3:
            point, normal = self._fit_plane_svd(inlier_points)
        else:
            point, normal = best_point, best_normal
        distances = np.abs(np.dot(inlier_points - point, normal))
        rms_error = float(math.sqrt(float(np.mean(distances * distances)))) if distances.size else 0.0
        max_error = float(np.max(distances)) if distances.size else 0.0
        return PlaneEstimate(point=point, normal=normal, inlier_mask=best_mask, rms_error=rms_error, max_error=max_error)

    @staticmethod
    def _plane_from_three_points(first: np.ndarray, second: np.ndarray, third: np.ndarray):
        normal = np.cross(second - first, third - first)
        norm = float(np.linalg.norm(normal))
        if norm <= 1e-9:
            return None
        return first, normal / norm

    @staticmethod
    def _fit_plane_svd(points: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        centroid = np.mean(points, axis=0)
        _, _, vh = np.linalg.svd(points - centroid, full_matrices=False)
        normal = vh[-1, :]
        norm = float(np.linalg.norm(normal))
        if norm <= 1e-9:
            normal = np.array((0.0, 0.0, 1.0), dtype=np.float64)
        else:
            normal = normal / norm
        return centroid, normal

    def _estimate_marker_plane_point(
        self,
        cloud: PointCloud2,
        raw_u: float,
        raw_v: float,
        estimate: PlaneEstimate,
    ) -> Optional[np.ndarray]:
        ray_point = self._read_marker_ray_point(cloud, raw_u, raw_v)
        if ray_point is None:
            return None
        ray_norm = float(np.linalg.norm(ray_point))
        if ray_norm <= 1e-9:
            return None
        direction = ray_point / ray_norm
        denominator = float(np.dot(estimate.normal, direction))
        if abs(denominator) <= 1e-7:
            return None
        distance_along_ray = float(np.dot(estimate.normal, estimate.point) / denominator)
        if distance_along_ray <= 0.0:
            return None
        return direction * distance_along_ray

    def _read_marker_ray_point(self, cloud: PointCloud2, raw_u: float, raw_v: float) -> Optional[np.ndarray]:
        center_u = int(round(raw_u))
        center_v = int(round(raw_v))
        radius = max(0, int(self._marker_ray_window_px))
        u_min = max(0, center_u - radius)
        u_max = min(int(cloud.width) - 1, center_u + radius)
        v_min = max(0, center_v - radius)
        v_max = min(int(cloud.height) - 1, center_v + radius)
        uvs = [(u, v) for v in range(v_min, v_max + 1) for u in range(u_min, u_max + 1)]
        points = []
        for point in point_cloud2.read_points(cloud, field_names=("x", "y", "z"), skip_nans=False, uvs=uvs):
            x, y, z = point[:3]
            if not np.isfinite(x) or not np.isfinite(y) or not np.isfinite(z):
                continue
            if abs(x) <= 1e-9 and abs(y) <= 1e-9 and abs(z) <= 1e-9:
                continue
            points.append((float(x), float(y), float(z)))
        if not points:
            return None
        return np.median(np.asarray(points, dtype=np.float64), axis=0)

    @staticmethod
    def _orient_normal_toward_camera(estimate: PlaneEstimate) -> PlaneEstimate:
        normal = estimate.normal
        if np.dot(normal, estimate.point) > 0.0:
            normal = -normal
        return PlaneEstimate(
            point=estimate.point,
            normal=normal,
            inlier_mask=estimate.inlier_mask,
            rms_error=estimate.rms_error,
            max_error=estimate.max_error,
        )

    def _publish_feedback(
        self,
        stage: str,
        started_at: rospy.Time,
        detection: GreenBlobDetection,
        raw_u: float,
        raw_v: float,
        roi_point_count: int,
        filtered_point_count: int,
        estimate: Optional[PlaneEstimate],
    ) -> None:
        feedback = EstimateHousingPlaneFeedback()
        feedback.stage = stage
        feedback.elapsed = float((rospy.Time.now() - started_at).to_sec())
        feedback.marker_center_x = float(detection.center_x)
        feedback.marker_center_y = float(detection.center_y)
        feedback.raw_center_u = float(raw_u)
        feedback.raw_center_v = float(raw_v)
        feedback.marker_area = float(detection.area)
        feedback.roi_point_count = int(roi_point_count)
        feedback.filtered_point_count = int(filtered_point_count)
        if estimate is not None:
            feedback.inlier_count = int(np.count_nonzero(estimate.inlier_mask))
            feedback.inlier_ratio = float(feedback.inlier_count) / max(1.0, float(filtered_point_count))
            feedback.rms_error = float(estimate.rms_error)
        self._server.publish_feedback(feedback)

    def _make_result(
        self,
        success: bool,
        message: str,
        error_code: str,
        started_at: rospy.Time,
        detection: Optional[GreenBlobDetection] = None,
        cloud: Optional[PointCloud2] = None,
        raw_u: float = 0.0,
        raw_v: float = 0.0,
        roi_point_count: int = 0,
        filtered_point_count: int = 0,
        estimate: Optional[PlaneEstimate] = None,
        marker_plane_point: Optional[np.ndarray] = None,
    ) -> EstimateHousingPlaneResult:
        result = EstimateHousingPlaneResult()
        result.success = bool(success)
        result.message = str(message)
        result.error_code = str(error_code)
        result.elapsed = float((rospy.Time.now() - started_at).to_sec())
        result.raw_center_u = float(raw_u)
        result.raw_center_v = float(raw_v)
        result.roi_point_count = int(roi_point_count)
        result.filtered_point_count = int(filtered_point_count)
        if detection is not None:
            result.marker_center_x = float(detection.center_x)
            result.marker_center_y = float(detection.center_y)
            result.marker_area = float(detection.area)
        if estimate is not None and cloud is not None:
            result.inlier_count = int(np.count_nonzero(estimate.inlier_mask))
            result.inlier_ratio = float(result.inlier_count) / max(1.0, float(filtered_point_count))
            result.rms_error = float(estimate.rms_error)
            result.max_error = float(estimate.max_error)
            result.plane_point = self._point_stamped(cloud.header.frame_id, estimate.point)
            result.plane_normal = self._vector3(estimate.normal)
            base_point, base_normal = self._transform_plane_to_base(cloud.header.frame_id, estimate.point, estimate.normal)
            result.plane_point_base = self._point_stamped(self._base_frame, base_point)
            result.plane_normal_base = self._vector3(base_normal)
            marker_point = marker_plane_point if marker_plane_point is not None else estimate.point
            marker_base_point, _ = self._transform_plane_to_base(cloud.header.frame_id, marker_point, estimate.normal)
            result.marker_plane_point = self._point_stamped(cloud.header.frame_id, marker_point)
            result.marker_plane_point_base = self._point_stamped(self._base_frame, marker_base_point)
        return result

    def _abort(
        self,
        message: str,
        started_at: rospy.Time,
        detection: Optional[GreenBlobDetection] = None,
        cloud: Optional[PointCloud2] = None,
        raw_u: float = 0.0,
        raw_v: float = 0.0,
        roi_point_count: int = 0,
        filtered_point_count: int = 0,
        error_code: Optional[str] = None,
    ) -> None:
        self._server.set_aborted(
            self._make_result(
                False,
                message,
                error_code or message,
                started_at,
                detection,
                cloud,
                raw_u,
                raw_v,
                roi_point_count,
                filtered_point_count,
            )
        )

    def _transform_plane_to_base(self, frame_id: str, point: np.ndarray, normal: np.ndarray):
        source_frame = self._base_lookup_frame(frame_id)
        point_for_tf, normal_for_tf = self._plane_in_base_lookup_frame(point, normal)
        if source_frame == self._base_frame:
            return point_for_tf, normal_for_tf
        transform = self._tf.lookup_transform(self._base_frame, source_frame)
        if transform is None:
            return point_for_tf, normal_for_tf
        rotation = transform.transform.rotation
        translation = transform.transform.translation
        normal_base = np.asarray(
            rotate_vector_by_quaternion(
                normal_for_tf[0],
                normal_for_tf[1],
                normal_for_tf[2],
                rotation.x,
                rotation.y,
                rotation.z,
                rotation.w,
            ),
            dtype=np.float64,
        )
        point_rotated = rotate_vector_by_quaternion(
            point_for_tf[0],
            point_for_tf[1],
            point_for_tf[2],
            rotation.x,
            rotation.y,
            rotation.z,
            rotation.w,
        )
        point_base = np.asarray(
            (
                point_rotated[0] + translation.x,
                point_rotated[1] + translation.y,
                point_rotated[2] + translation.z,
            ),
            dtype=np.float64,
        )
        norm = float(np.linalg.norm(normal_base))
        if norm > 1e-9:
            normal_base = normal_base / norm
        return point_base, normal_base

    def _base_lookup_frame(self, cloud_frame_id: str) -> str:
        return self._base_transform_frame or cloud_frame_id.strip()

    def _plane_in_base_lookup_frame(self, point: np.ndarray, normal: np.ndarray):
        if not self._base_transform_frame:
            return point, normal
        if self._cloud_to_base_transform_rotation in ("", "identity", "none"):
            return point, normal
        if self._cloud_to_base_transform_rotation == "camera_frame_to_optical":
            return self._rotate_plane(point, normal, CAMERA_FRAME_TO_OPTICAL_QUATERNION)
        rospy.logwarn_throttle(
            2.0,
            "[usb_c_insertion] event=estimate_housing_plane_unknown_cloud_rotation mode=%s",
            self._cloud_to_base_transform_rotation,
        )
        return point, normal

    @staticmethod
    def _rotate_plane(point: np.ndarray, normal: np.ndarray, quaternion_xyzw):
        qx, qy, qz, qw = quaternion_xyzw
        rotated_point = np.asarray(
            rotate_vector_by_quaternion(point[0], point[1], point[2], qx, qy, qz, qw),
            dtype=np.float64,
        )
        rotated_normal = np.asarray(
            rotate_vector_by_quaternion(normal[0], normal[1], normal[2], qx, qy, qz, qw),
            dtype=np.float64,
        )
        norm = float(np.linalg.norm(rotated_normal))
        if norm > 1e-9:
            rotated_normal = rotated_normal / norm
        return rotated_point, rotated_normal

    @staticmethod
    def _point_stamped(frame_id: str, point: np.ndarray) -> PointStamped:
        stamped = PointStamped()
        stamped.header.stamp = rospy.Time.now()
        stamped.header.frame_id = frame_id
        stamped.point.x = float(point[0])
        stamped.point.y = float(point[1])
        stamped.point.z = float(point[2])
        return stamped

    @staticmethod
    def _vector3(vector: np.ndarray) -> Vector3:
        msg = Vector3()
        msg.x = float(vector[0])
        msg.y = float(vector[1])
        msg.z = float(vector[2])
        return msg

    def _is_detection_stale(self, detection: GreenBlobDetection, now: rospy.Time) -> bool:
        if self._max_image_age <= 0.0:
            return False
        return (now - detection.stamp).to_sec() > self._max_image_age

    def _is_cloud_stale(self, cloud: PointCloud2, now: rospy.Time) -> bool:
        if self._max_cloud_age <= 0.0:
            return False
        stamp = cloud.header.stamp if cloud.header.stamp != rospy.Time() else now
        return (now - stamp).to_sec() > self._max_cloud_age

    @staticmethod
    def _goal_or_default(value: float, default: float) -> float:
        return float(value) if float(value) > 0.0 else float(default)

    @staticmethod
    def _goal_int_or_default(value: int, default: int) -> int:
        return int(value) if int(value) > 0 else int(default)

    @staticmethod
    def _read_hsv_param(param_name: str) -> Tuple[int, int, int]:
        value = required_vector_param(param_name)
        if not isinstance(value, (list, tuple)) or len(value) != 3:
            rospy.logwarn("[usb_c_insertion] event=estimate_housing_plane_invalid_hsv_param param=%s", param_name)
            raise ValueError("Invalid HSV ROS parameter: %s" % rospy.resolve_name(param_name))
        return tuple(max(0, min(255, int(component))) for component in value)

    @staticmethod
    def _normalize_image_rotation_deg(rotation_deg: float) -> float:
        normalized = float(rotation_deg) % 360.0
        allowed = (0.0, 90.0, 180.0, 270.0)
        closest = min(allowed, key=lambda value: abs(value - normalized))
        if abs(closest - normalized) > 1e-3:
            rospy.logwarn(
                "[usb_c_insertion] event=estimate_housing_plane_image_rotation_rounded requested_deg=%.1f applied_deg=%.1f",
                rotation_deg,
                closest,
            )
        return closest


def main() -> None:
    rospy.init_node("usb_c_insertion_estimate_housing_plane_action_server")
    EstimateHousingPlaneActionServer()
    rospy.spin()


if __name__ == "__main__":
    main()
