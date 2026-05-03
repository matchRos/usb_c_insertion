#!/usr/bin/env python3

from __future__ import annotations

from dataclasses import dataclass
import json
import math
import os
import sys
from typing import List, Optional, Tuple

import cv2
import numpy as np
import rospy
from sensor_msgs.msg import Image
from std_msgs.msg import String

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from param_utils import get_param


@dataclass(frozen=True)
class ConnectorDetection:
    center_x: float
    center_y: float
    area: float
    bbox: Tuple[int, int, int, int]
    center_offset_norm: float


@dataclass(frozen=True)
class CardDetection:
    center_x: float
    center_y: float
    width: float
    height: float
    angle_deg: float
    area: float
    rectangularity: float
    aspect_ratio: float
    vertical_error_deg: float
    score: float
    box_points: Tuple[Tuple[int, int], Tuple[int, int], Tuple[int, int], Tuple[int, int]]
    connector: Optional[ConnectorDetection]


@dataclass(frozen=True)
class CardGroupEstimate:
    observed_count: int
    expected_count: int
    complete: bool
    bbox: Tuple[int, int, int, int]
    center_x: float
    center_y: float
    width: float
    height: float
    order_axis: str
    order_direction: str
    slot_centers: Tuple[Tuple[float, float], ...]


class UsbCardDetectorNode:
    """
    Detect bright rectangular USB-C cards on a black case.

    This node is intentionally standalone: it publishes image-space detections
    and an annotated debug image, but it does not feed the robot workflow yet.
    """

    def __init__(self):
        self._image_topic = self._str_param(
            "~usb_card_detector/image_topic",
            self._str_param("~center_port/image_topic", "/zedm/zed_node/left/image_rect_color"),
        )
        self._detections_topic = self._str_param(
            "~usb_card_detector/detections_topic",
            "/usb_c_insertion/usb_card_detector/detections",
        )
        self._debug_image_topic = self._str_param(
            "~usb_card_detector/debug_image_topic",
            "/usb_c_insertion/usb_card_detector/debug_image",
        )
        self._mask_image_topic = self._str_param(
            "~usb_card_detector/mask_image_topic",
            "/usb_c_insertion/usb_card_detector/mask",
        )
        self._publish_debug_image = self._bool_param("~usb_card_detector/publish_debug_image", True)
        self._publish_mask = self._private_bool_override(
            "~publish_mask",
            "~usb_card_detector/publish_mask",
            False,
        )
        self._show_gui = self._private_bool_override(
            "~show_gui",
            "~usb_card_detector/show_gui",
            False,
        )
        self._debug_window_name = self._str_param(
            "~usb_card_detector/debug_window_name",
            "USB card detector",
        )
        self._debug_rate_hz = max(0.1, self._float_param("~usb_card_detector/debug_rate_hz", 5.0))
        self._log_rate_hz = max(0.1, self._float_param("~usb_card_detector/log_rate_hz", 1.0))
        self._image_rotation_deg = self._normalize_image_rotation_deg(
            self._float_param("~usb_card_detector/image_rotation_deg", 180.0)
        )

        self._card_min_value = self._int_param("~usb_card_detector/card_min_value", 125)
        self._card_max_saturation = self._int_param("~usb_card_detector/card_max_saturation", 85)
        self._card_min_channel_value = self._int_param("~usb_card_detector/card_min_channel_value", 80)
        self._card_max_channel_delta = self._int_param("~usb_card_detector/card_max_channel_delta", 95)
        self._mask_overlay_alpha = max(0.0, min(1.0, self._float_param("~usb_card_detector/mask_overlay_alpha", 0.12)))
        self._morph_kernel_size = max(0, self._int_param("~usb_card_detector/morph_kernel_size", 7))
        self._morph_close_iterations = max(0, self._int_param("~usb_card_detector/morph_close_iterations", 2))
        self._morph_open_iterations = max(0, self._int_param("~usb_card_detector/morph_open_iterations", 1))

        self._min_card_area = self._float_param("~usb_card_detector/min_card_area", 250.0)
        self._max_card_area = self._float_param("~usb_card_detector/max_card_area", 25000.0)
        self._min_card_aspect_ratio = self._float_param("~usb_card_detector/min_card_aspect_ratio", 1.15)
        self._max_card_aspect_ratio = self._float_param("~usb_card_detector/max_card_aspect_ratio", 8.0)
        self._min_card_rectangularity = self._float_param("~usb_card_detector/min_card_rectangularity", 0.55)
        self._min_card_extent = self._float_param("~usb_card_detector/min_card_extent", 0.40)
        self._require_vertical_cards = self._bool_param("~usb_card_detector/require_vertical_cards", True)
        self._max_vertical_error_deg = self._float_param("~usb_card_detector/max_vertical_error_deg", 20.0)
        self._max_detections = max(1, self._int_param("~usb_card_detector/max_detections", 4))
        self._expected_card_count = max(
            0,
            self._int_param(
                "~usb_card_detector/expected_card_count",
                int(get_param("~workflow/usb_card_total_count", 0)),
            ),
        )
        self._card_order_axis = self._str_param("~usb_card_detector/card_order_axis", "x").lower()
        self._card_order_direction = self._str_param("~usb_card_detector/card_order_direction", "ascending").lower()
        self._require_card_group = self._bool_param("~usb_card_detector/require_card_group", True)
        self._min_cards_in_group = max(1, self._int_param("~usb_card_detector/min_cards_in_group", 2))
        self._fallback_single_card_with_connector = self._bool_param(
            "~usb_card_detector/fallback_single_card_with_connector",
            True,
        )
        self._fallback_single_card_when_no_group = self._bool_param(
            "~usb_card_detector/fallback_single_card_when_no_group",
            True,
        )
        self._group_max_center_y_delta_px = self._float_param("~usb_card_detector/group_max_center_y_delta_px", 45.0)
        self._group_max_center_x_gap_px = self._float_param("~usb_card_detector/group_max_center_x_gap_px", 120.0)
        self._group_max_center_x_gap_ratio = self._float_param("~usb_card_detector/group_max_center_x_gap_ratio", 0.55)
        self._group_max_long_side_delta_ratio = self._float_param(
            "~usb_card_detector/group_max_long_side_delta_ratio",
            0.35,
        )
        self._group_max_short_side_delta_ratio = self._float_param(
            "~usb_card_detector/group_max_short_side_delta_ratio",
            0.45,
        )
        self._group_max_vertical_error_delta_deg = self._float_param(
            "~usb_card_detector/group_max_vertical_error_delta_deg",
            10.0,
        )
        self._split_enabled = self._bool_param("~usb_card_detector/split_enabled", True)
        self._split_min_gap_px = max(1, self._int_param("~usb_card_detector/split_min_gap_px", 3))
        self._split_gap_fill_ratio = max(
            0.0,
            min(1.0, self._float_param("~usb_card_detector/split_gap_fill_ratio", 0.12)),
        )
        self._split_min_child_area_ratio = max(
            0.0,
            min(1.0, self._float_param("~usb_card_detector/split_min_child_area_ratio", 0.18)),
        )
        self._split_max_depth = max(0, self._int_param("~usb_card_detector/split_max_depth", 4))

        self._connector_max_value = self._int_param("~usb_card_detector/connector_max_value", 150)
        self._connector_min_area = self._float_param("~usb_card_detector/connector_min_area", 80.0)
        self._connector_max_area_ratio = self._float_param("~usb_card_detector/connector_max_area_ratio", 0.25)
        self._connector_min_aspect_ratio = self._float_param(
            "~usb_card_detector/connector_min_aspect_ratio",
            1.8,
        )
        self._connector_max_aspect_ratio = self._float_param(
            "~usb_card_detector/connector_max_aspect_ratio",
            7.0,
        )
        self._connector_max_center_offset_norm = self._float_param(
            "~usb_card_detector/connector_max_center_offset_norm",
            0.45,
        )
        self._connector_mask_erode_px = max(0, self._int_param("~usb_card_detector/connector_mask_erode_px", 5))
        self._connector_morph_kernel_size = max(
            0,
            self._int_param("~usb_card_detector/connector_morph_kernel_size", 3),
        )

        self._last_debug_publish = rospy.Time(0)
        self._last_log = rospy.Time(0)
        self._detections_publisher = rospy.Publisher(self._detections_topic, String, queue_size=10)
        self._debug_image_publisher = rospy.Publisher(self._debug_image_topic, Image, queue_size=1)
        self._mask_image_publisher = rospy.Publisher(self._mask_image_topic, Image, queue_size=1)
        self._image_subscriber = rospy.Subscriber(self._image_topic, Image, self._image_callback, queue_size=1)
        rospy.on_shutdown(self._handle_shutdown)

        rospy.loginfo(
            "[usb_c_insertion] event=usb_card_detector_ready image_topic=%s detections_topic=%s debug_image_topic=%s show_gui=%s rotation_deg=%.1f",
            self._image_topic,
            self._detections_topic,
            self._debug_image_topic,
            str(self._show_gui).lower(),
            self._image_rotation_deg,
        )

    def _image_callback(self, msg: Image) -> None:
        try:
            raw_bgr = self._image_to_bgr(msg)
            bgr = self._rotate_image_for_processing(raw_bgr)
            mask = self._build_card_mask(bgr)
            detections, card_group = self._detect_cards(bgr, mask)
            self._publish_detections(msg, bgr, detections, card_group)
            self._publish_debug_outputs(msg, bgr, mask, detections, card_group)
            self._log_detections(detections, card_group)
        except Exception as exc:
            rospy.logwarn_throttle(
                1.0,
                "[usb_c_insertion] event=usb_card_detector_failed error=%s",
                exc,
            )

    def _build_card_mask(self, bgr: np.ndarray) -> np.ndarray:
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        min_channel = np.min(bgr, axis=2)
        max_channel = np.max(bgr, axis=2)
        channel_delta = max_channel.astype(np.int16) - min_channel.astype(np.int16)

        bright_ok = gray >= int(self._card_min_value)
        neutral_ok = hsv[:, :, 1] <= int(self._card_max_saturation)
        channel_floor_ok = min_channel >= int(self._card_min_channel_value)
        channel_balance_ok = channel_delta <= int(self._card_max_channel_delta)
        mask = np.where(
            bright_ok & neutral_ok & channel_floor_ok & channel_balance_ok,
            255,
            0,
        ).astype(np.uint8)

        kernel_size = int(self._morph_kernel_size)
        if kernel_size > 1:
            kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
            if self._morph_open_iterations > 0:
                mask = cv2.morphologyEx(
                    mask,
                    cv2.MORPH_OPEN,
                    kernel,
                    iterations=int(self._morph_open_iterations),
                )
            if self._morph_close_iterations > 0:
                mask = cv2.morphologyEx(
                    mask,
                    cv2.MORPH_CLOSE,
                    kernel,
                    iterations=int(self._morph_close_iterations),
                )
        return mask

    def _detect_cards(self, bgr: np.ndarray, mask: np.ndarray) -> Tuple[List[CardDetection], Optional[CardGroupEstimate]]:
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        candidates: List[CardDetection] = []
        for contour in contours:
            for segment_mask, segment_contour in self._split_card_contour(mask, contour):
                detection = self._build_card_detection(bgr, segment_mask, segment_contour)
                if detection is not None:
                    candidates.append(detection)

        candidates = self._filter_card_group(candidates)
        card_group = self._build_card_group_estimate(candidates)
        candidates.sort(key=lambda item: item.score, reverse=True)
        selected = candidates[: self._max_detections]
        return self._sort_detections_for_indexing(selected), card_group

    def _build_card_group_estimate(self, detections: List[CardDetection]) -> Optional[CardGroupEstimate]:
        if not detections:
            return None
        points = []
        for detection in detections:
            points.extend(detection.box_points)
        if not points:
            return None
        xs = [float(point[0]) for point in points]
        ys = [float(point[1]) for point in points]
        min_x = max(0.0, min(xs))
        max_x = max(min_x + 1.0, max(xs))
        min_y = max(0.0, min(ys))
        max_y = max(min_y + 1.0, max(ys))
        width = max_x - min_x
        height = max_y - min_y
        center_x = min_x + 0.5 * width
        center_y = min_y + 0.5 * height
        expected_count = int(self._expected_card_count)
        slot_count = expected_count if expected_count > 0 else len(detections)
        complete = expected_count <= 0 or len(detections) >= expected_count
        order_axis = self._slot_order_axis()
        reverse = self._card_order_direction in ("descending", "desc", "reverse")
        slot_centers = []
        for index in range(slot_count):
            fraction_index = slot_count - index - 1 if reverse else index
            fraction = (float(fraction_index) + 0.5) / max(1.0, float(slot_count))
            if order_axis == "y":
                slot_centers.append((center_x, min_y + fraction * height))
            else:
                slot_centers.append((min_x + fraction * width, center_y))
        return CardGroupEstimate(
            observed_count=len(detections),
            expected_count=expected_count,
            complete=complete,
            bbox=(int(round(min_x)), int(round(min_y)), int(round(width)), int(round(height))),
            center_x=center_x,
            center_y=center_y,
            width=width,
            height=height,
            order_axis=order_axis,
            order_direction=self._card_order_direction,
            slot_centers=tuple(slot_centers),
        )

    def _slot_order_axis(self) -> str:
        if self._card_order_axis == "y":
            return "y"
        return "x"

    def _filter_card_group(self, candidates: List[CardDetection]) -> List[CardDetection]:
        if not self._require_card_group:
            return candidates
        if self._min_cards_in_group <= 1:
            return candidates
        if len(candidates) < self._min_cards_in_group:
            return self._single_card_connector_fallback(candidates)

        groups: List[List[CardDetection]] = []
        current_group: List[CardDetection] = []
        for candidate in sorted(candidates, key=lambda item: item.center_x):
            if not current_group:
                current_group = [candidate]
                continue
            if self._cards_are_group_neighbors(current_group[-1], candidate):
                current_group.append(candidate)
                continue
            groups.append(current_group)
            current_group = [candidate]
        if current_group:
            groups.append(current_group)

        groups = [group for group in groups if len(group) >= self._min_cards_in_group]
        if not groups:
            return self._single_card_connector_fallback(candidates)
        groups.sort(
            key=lambda group: (
                len(group),
                sum(card.score for card in group),
                -self._group_center_y_spread(group),
            ),
            reverse=True,
        )
        return groups[0]

    def _single_card_connector_fallback(self, candidates: List[CardDetection]) -> List[CardDetection]:
        if self._fallback_single_card_when_no_group and len(candidates) == 1:
            return [candidates[0]]
        if not self._fallback_single_card_with_connector:
            return []
        connector_candidates = [candidate for candidate in candidates if candidate.connector is not None]
        if not connector_candidates:
            return []
        connector_candidates.sort(key=lambda item: item.score, reverse=True)
        return [connector_candidates[0]]

    def _cards_are_group_neighbors(self, first: CardDetection, second: CardDetection) -> bool:
        center_y_delta = abs(float(first.center_y) - float(second.center_y))
        if center_y_delta > max(0.0, self._group_max_center_y_delta_px):
            return False

        center_x_gap = abs(float(first.center_x) - float(second.center_x))
        average_long_side = 0.5 * (float(first.width) + float(second.width))
        max_center_x_gap = max(
            max(0.0, self._group_max_center_x_gap_px),
            max(0.0, self._group_max_center_x_gap_ratio) * average_long_side,
        )
        if center_x_gap > max_center_x_gap:
            return False

        if self._relative_delta(first.width, second.width) > max(0.0, self._group_max_long_side_delta_ratio):
            return False
        if self._relative_delta(first.height, second.height) > max(0.0, self._group_max_short_side_delta_ratio):
            return False

        vertical_error_delta = abs(float(first.vertical_error_deg) - float(second.vertical_error_deg))
        return vertical_error_delta <= max(0.0, self._group_max_vertical_error_delta_deg)

    @staticmethod
    def _relative_delta(first: float, second: float) -> float:
        return abs(float(first) - float(second)) / max(1e-6, max(abs(float(first)), abs(float(second))))

    @staticmethod
    def _group_center_y_spread(group: List[CardDetection]) -> float:
        values = [float(card.center_y) for card in group]
        return max(values) - min(values) if values else 0.0

    def _sort_detections_for_indexing(self, detections: List[CardDetection]) -> List[CardDetection]:
        axis = self._card_order_axis
        reverse = self._card_order_direction in ("descending", "desc", "reverse")
        if axis == "y":
            return sorted(detections, key=lambda item: item.center_y, reverse=reverse)
        if axis == "score":
            return sorted(detections, key=lambda item: item.score, reverse=reverse)
        return sorted(detections, key=lambda item: item.center_x, reverse=reverse)

    def _build_card_detection(
        self,
        bgr: np.ndarray,
        card_mask: np.ndarray,
        contour,
    ) -> Optional[CardDetection]:
        area = float(cv2.contourArea(contour))
        if area < self._min_card_area or area > self._max_card_area:
            return None

        rect = cv2.minAreaRect(contour)
        (center_x, center_y), (rect_width, rect_height), angle_deg = rect
        if rect_width <= 1.0 or rect_height <= 1.0:
            return None

        long_side = max(float(rect_width), float(rect_height))
        short_side = min(float(rect_width), float(rect_height))
        aspect_ratio = long_side / max(1e-6, short_side)
        if aspect_ratio < self._min_card_aspect_ratio or aspect_ratio > self._max_card_aspect_ratio:
            return None

        rect_area = float(rect_width * rect_height)
        rectangularity = area / max(1e-6, rect_area)
        if rectangularity < self._min_card_rectangularity:
            return None

        x, y, width, height = cv2.boundingRect(contour)
        extent = area / max(1.0, float(width * height))
        if extent < self._min_card_extent:
            return None

        box = cv2.boxPoints(rect).astype(np.int32)
        vertical_error_deg = self._long_axis_vertical_error_deg(box)
        if self._require_vertical_cards and vertical_error_deg > self._max_vertical_error_deg:
            return None

        connector_mask = np.zeros(card_mask.shape, dtype=np.uint8)
        cv2.drawContours(connector_mask, [contour], -1, 255, thickness=-1)
        connector = self._detect_connector(bgr, connector_mask, contour, rect_area, center_x, center_y)
        score = area * rectangularity * min(1.5, aspect_ratio)
        if connector is not None:
            score *= 1.35

        box_points = tuple((int(point[0]), int(point[1])) for point in box)
        return CardDetection(
            center_x=float(center_x),
            center_y=float(center_y),
            width=long_side,
            height=short_side,
            angle_deg=float(angle_deg),
            area=area,
            rectangularity=rectangularity,
            aspect_ratio=aspect_ratio,
            vertical_error_deg=vertical_error_deg,
            score=score,
            box_points=box_points,
            connector=connector,
        )

    def _split_card_contour(self, mask: np.ndarray, contour) -> List[Tuple[np.ndarray, object]]:
        component_mask = np.zeros(mask.shape, dtype=np.uint8)
        cv2.drawContours(component_mask, [contour], -1, 255, thickness=-1)
        component_mask = cv2.bitwise_and(mask, component_mask)
        x, y, width, height = cv2.boundingRect(contour)
        segments = self._split_mask_roi(component_mask[y : y + height, x : x + width], x, y, mask.shape, 0)
        result = []
        for segment_mask in segments:
            contours, _ = cv2.findContours(segment_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for segment_contour in contours:
                if cv2.contourArea(segment_contour) >= self._min_card_area:
                    result.append((segment_mask, segment_contour))
        return result

    def _split_mask_roi(
        self,
        roi_mask: np.ndarray,
        offset_x: int,
        offset_y: int,
        full_shape: Tuple[int, int],
        depth: int,
    ) -> List[np.ndarray]:
        if not self._split_enabled or depth >= self._split_max_depth:
            return [self._offset_mask(roi_mask, offset_x, offset_y, full_shape)]

        split = self._find_best_split(roi_mask)
        if split is None:
            return [self._offset_mask(roi_mask, offset_x, offset_y, full_shape)]

        axis, start, end = split
        first_roi = roi_mask[:start, :] if axis == "y" else roi_mask[:, :start]
        second_roi = roi_mask[end:, :] if axis == "y" else roi_mask[:, end:]
        first_area = float(cv2.countNonZero(first_roi))
        second_area = float(cv2.countNonZero(second_roi))
        total_area = max(1.0, float(cv2.countNonZero(roi_mask)))
        min_child_area = max(self._min_card_area, total_area * self._split_min_child_area_ratio)
        if first_area < min_child_area or second_area < min_child_area:
            return [self._offset_mask(roi_mask, offset_x, offset_y, full_shape)]

        first_offset_x = offset_x
        first_offset_y = offset_y
        second_offset_x = offset_x if axis == "y" else offset_x + end
        second_offset_y = offset_y + end if axis == "y" else offset_y
        return (
            self._split_mask_roi(first_roi, first_offset_x, first_offset_y, full_shape, depth + 1)
            + self._split_mask_roi(second_roi, second_offset_x, second_offset_y, full_shape, depth + 1)
        )

    def _find_best_split(self, roi_mask: np.ndarray) -> Optional[Tuple[str, int, int]]:
        candidates = []
        for axis in ("y", "x"):
            projection = np.mean(roi_mask > 0, axis=1 if axis == "y" else 0)
            min_length = max(8, int(0.15 * projection.shape[0]))
            split = self._find_dark_gap(projection, min_length)
            if split is None:
                continue
            start, end, fill_ratio = split
            candidates.append((fill_ratio, axis, start, end))
        if not candidates:
            return None
        candidates.sort(key=lambda item: item[0])
        _, axis, start, end = candidates[0]
        return axis, start, end

    def _find_dark_gap(
        self,
        projection: np.ndarray,
        min_side_length: int,
    ) -> Optional[Tuple[int, int, float]]:
        is_gap = projection <= self._split_gap_fill_ratio
        best = None
        index = 0
        while index < len(is_gap):
            if not is_gap[index]:
                index += 1
                continue
            start = index
            while index < len(is_gap) and is_gap[index]:
                index += 1
            end = index
            if end - start < self._split_min_gap_px:
                continue
            if start < min_side_length or len(is_gap) - end < min_side_length:
                continue
            fill_ratio = float(np.mean(projection[start:end]))
            if best is None or fill_ratio < best[2]:
                best = (start, end, fill_ratio)
        return best

    @staticmethod
    def _offset_mask(
        roi_mask: np.ndarray,
        offset_x: int,
        offset_y: int,
        full_shape: Tuple[int, int],
    ) -> np.ndarray:
        points = cv2.findNonZero(roi_mask)
        if points is None:
            return np.zeros(full_shape, dtype=np.uint8)
        x, y, width, height = cv2.boundingRect(points)
        cropped = roi_mask[y : y + height, x : x + width]
        full = np.zeros(full_shape, dtype=np.uint8)
        full[offset_y + y : offset_y + y + height, offset_x + x : offset_x + x + width] = cropped
        return full

    def _detect_connector(
        self,
        bgr: np.ndarray,
        card_mask: np.ndarray,
        contour,
        card_rect_area: float,
        card_center_x: float,
        card_center_y: float,
    ) -> Optional[ConnectorDetection]:
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        inner_mask = card_mask.copy()
        erode_px = int(self._connector_mask_erode_px)
        if erode_px > 0:
            kernel = np.ones((erode_px, erode_px), dtype=np.uint8)
            inner_mask = cv2.erode(inner_mask, kernel, iterations=1)

        dark_mask = np.where((gray <= int(self._connector_max_value)) & (inner_mask > 0), 255, 0).astype(np.uint8)
        kernel_size = int(self._connector_morph_kernel_size)
        if kernel_size > 1:
            kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
            dark_mask = cv2.morphologyEx(dark_mask, cv2.MORPH_OPEN, kernel, iterations=1)
        contours, _ = cv2.findContours(dark_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        best = None
        best_score = -1.0
        max_connector_area = max(self._connector_min_area, card_rect_area * self._connector_max_area_ratio)
        for connector_contour in contours:
            area = float(cv2.contourArea(connector_contour))
            if area < self._connector_min_area or area > max_connector_area:
                continue
            x, y, width, height = cv2.boundingRect(connector_contour)
            if width <= 1 or height <= 1:
                continue
            aspect_ratio = max(float(width), float(height)) / max(1.0, min(float(width), float(height)))
            if aspect_ratio < self._connector_min_aspect_ratio or aspect_ratio > self._connector_max_aspect_ratio:
                continue

            moments = cv2.moments(connector_contour)
            if abs(moments["m00"]) <= 1e-9:
                center_x = x + width * 0.5
                center_y = y + height * 0.5
            else:
                center_x = float(moments["m10"] / moments["m00"])
                center_y = float(moments["m01"] / moments["m00"])

            _, _, card_width, card_height = cv2.boundingRect(contour)
            norm_x = (center_x - card_center_x) / max(1.0, 0.5 * float(card_width))
            norm_y = (center_y - card_center_y) / max(1.0, 0.5 * float(card_height))
            center_offset_norm = math.sqrt(norm_x * norm_x + norm_y * norm_y)
            if center_offset_norm > self._connector_max_center_offset_norm:
                continue

            score = area * (1.0 - min(0.95, center_offset_norm))
            if score > best_score:
                best_score = score
                best = ConnectorDetection(
                    center_x=center_x,
                    center_y=center_y,
                    area=area,
                    bbox=(int(x), int(y), int(width), int(height)),
                    center_offset_norm=center_offset_norm,
                )
        return best

    @staticmethod
    def _long_axis_vertical_error_deg(box_points: np.ndarray) -> float:
        longest_edge = (0.0, 0.0)
        longest_length = -1.0
        for index in range(4):
            first = box_points[index]
            second = box_points[(index + 1) % 4]
            dx = float(second[0] - first[0])
            dy = float(second[1] - first[1])
            length = math.sqrt(dx * dx + dy * dy)
            if length > longest_length:
                longest_length = length
                longest_edge = (dx, dy)
        if longest_length <= 1e-6:
            return 90.0
        angle_deg = abs(math.degrees(math.atan2(longest_edge[1], longest_edge[0])))
        if angle_deg > 90.0:
            angle_deg = 180.0 - angle_deg
        return abs(90.0 - angle_deg)

    def _publish_detections(
        self,
        msg: Image,
        bgr: np.ndarray,
        detections: List[CardDetection],
        card_group: Optional[CardGroupEstimate],
    ) -> None:
        payload = {
            "stamp": msg.header.stamp.to_sec() if msg.header.stamp else rospy.Time.now().to_sec(),
            "frame_id": msg.header.frame_id,
            "image_width": int(bgr.shape[1]),
            "image_height": int(bgr.shape[0]),
            "count": len(detections),
            "cards": [self._card_to_dict(index, detection) for index, detection in enumerate(detections)],
            "card_group": self._card_group_to_dict(card_group),
        }
        self._detections_publisher.publish(String(data=json.dumps(payload, sort_keys=True)))

    def _publish_debug_outputs(
        self,
        msg: Image,
        bgr: np.ndarray,
        mask: np.ndarray,
        detections: List[CardDetection],
        card_group: Optional[CardGroupEstimate],
    ) -> None:
        if not self._publish_debug_image and not self._publish_mask and not self._show_gui:
            return
        now = rospy.Time.now()
        if (now - self._last_debug_publish).to_sec() < 1.0 / self._debug_rate_hz:
            return
        self._last_debug_publish = now

        debug = None
        if self._publish_debug_image or self._show_gui:
            debug = self._draw_debug_image(bgr, mask, detections, card_group)
        if self._publish_debug_image and debug is not None:
            self._debug_image_publisher.publish(self._bgr_to_image_msg(debug, msg.header.frame_id))
        if self._show_gui and debug is not None:
            self._show_debug_window(debug)
        if self._publish_mask:
            mask_bgr = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
            self._mask_image_publisher.publish(self._bgr_to_image_msg(mask_bgr, msg.header.frame_id))

    def _draw_debug_image(
        self,
        bgr: np.ndarray,
        mask: np.ndarray,
        detections: List[CardDetection],
        card_group: Optional[CardGroupEstimate],
    ) -> np.ndarray:
        debug = bgr.copy()
        tint = np.zeros_like(debug)
        tint[:, :, 0] = mask
        tint[:, :, 1] = mask // 2
        debug = cv2.addWeighted(debug, 1.0 - self._mask_overlay_alpha, tint, self._mask_overlay_alpha, 0.0)

        for index, detection in enumerate(detections):
            box = np.array(detection.box_points, dtype=np.int32)
            cv2.polylines(debug, [box], isClosed=True, color=(0, 255, 255), thickness=2, lineType=cv2.LINE_AA)
            center = (int(round(detection.center_x)), int(round(detection.center_y)))
            cv2.drawMarker(
                debug,
                center,
                (0, 255, 255),
                markerType=cv2.MARKER_CROSS,
                markerSize=18,
                thickness=2,
                line_type=cv2.LINE_AA,
            )
            cv2.circle(debug, center, 5, (0, 255, 255), -1, cv2.LINE_AA)
            self._draw_label(
                debug,
                "card%d center=(%d,%d)" % (index + 1, center[0], center[1]),
                (center[0] + 8, max(18, center[1] - 24)),
                (0, 255, 255),
            )
            label = "box %.0fx%.0f ar=%.2f rect=%.2f" % (
                detection.width,
                detection.height,
                detection.aspect_ratio,
                detection.rectangularity,
            )
            self._draw_label(
                debug,
                "vertical_err=%.1fdeg" % detection.vertical_error_deg,
                (center[0] + 8, min(debug.shape[0] - 8, center[1] + 8)),
                (0, 255, 255),
            )
            self._draw_label(
                debug,
                label,
                (center[0] + 8, max(18, center[1] - 8)),
                (0, 255, 255),
            )

            if detection.connector is not None:
                connector = detection.connector
                x, y, width, height = connector.bbox
                cv2.rectangle(debug, (x, y), (x + width, y + height), (255, 0, 255), 2, cv2.LINE_AA)
                cv2.circle(
                    debug,
                    (int(round(connector.center_x)), int(round(connector.center_y))),
                    5,
                    (255, 0, 255),
                    -1,
                    cv2.LINE_AA,
                )
                cv2.line(
                    debug,
                    center,
                    (int(round(connector.center_x)), int(round(connector.center_y))),
                    (255, 0, 255),
                    1,
                    cv2.LINE_AA,
                )
                self._draw_label(
                    debug,
                    "port=(%d,%d)" % (int(round(connector.center_x)), int(round(connector.center_y))),
                    (x + width + 6, max(18, y + height // 2)),
                    (255, 0, 255),
                )
            else:
                self._draw_label(
                    debug,
                    "port not found",
                    (center[0] + 8, min(debug.shape[0] - 8, center[1] + 16)),
                    (80, 180, 255),
                )

        if card_group is not None:
            x, y, width, height = card_group.bbox
            cv2.rectangle(debug, (x, y), (x + width, y + height), (0, 180, 255), 2, cv2.LINE_AA)
            for slot_index, (slot_x, slot_y) in enumerate(card_group.slot_centers):
                center = (int(round(slot_x)), int(round(slot_y)))
                cv2.drawMarker(
                    debug,
                    center,
                    (0, 180, 255),
                    markerType=cv2.MARKER_TILTED_CROSS,
                    markerSize=14,
                    thickness=1,
                    line_type=cv2.LINE_AA,
                )
                if slot_index < 12:
                    self._draw_label(
                        debug,
                        "s%d" % (slot_index + 1),
                        (center[0] + 5, center[1] + 18),
                        (0, 180, 255),
                    )
            expected_text = str(card_group.expected_count) if card_group.expected_count > 0 else "auto"
            self._draw_label(
                debug,
                "group %d/%s complete=%s" % (
                    card_group.observed_count,
                    expected_text,
                    str(card_group.complete).lower(),
                ),
                (x + 6, max(18, y - 10)),
                (0, 180, 255),
            )

        status = "USB card detections: %d" % len(detections)
        cv2.rectangle(debug, (8, 8), (300, 36), (0, 0, 0), -1)
        cv2.putText(debug, status, (16, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
        return debug

    def _show_debug_window(self, debug: np.ndarray) -> None:
        try:
            cv2.imshow(self._debug_window_name, debug)
            cv2.waitKey(1)
        except cv2.error as exc:
            rospy.logwarn(
                "[usb_c_insertion] event=usb_card_detector_gui_failed error=%s disabling_gui=true",
                exc,
            )
            self._show_gui = False

    @staticmethod
    def _draw_label(image: np.ndarray, text: str, origin: Tuple[int, int], color: Tuple[int, int, int]) -> None:
        x = max(0, min(int(origin[0]), image.shape[1] - 1))
        y = max(14, min(int(origin[1]), image.shape[0] - 4))
        cv2.putText(image, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(image, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)

    def _handle_shutdown(self) -> None:
        if self._show_gui:
            try:
                cv2.destroyWindow(self._debug_window_name)
            except cv2.error:
                pass

    def _log_detections(
        self,
        detections: List[CardDetection],
        card_group: Optional[CardGroupEstimate],
    ) -> None:
        now = rospy.Time.now()
        if (now - self._last_log).to_sec() < 1.0 / self._log_rate_hz:
            return
        self._last_log = now
        summaries = []
        for index, detection in enumerate(detections):
            connector = "none"
            if detection.connector is not None:
                connector = "(%.1f,%.1f)" % (detection.connector.center_x, detection.connector.center_y)
            summaries.append(
                "card%d center=(%.1f,%.1f) size=(%.1f,%.1f) connector=%s"
                % (
                    index + 1,
                    detection.center_x,
                    detection.center_y,
                    detection.width,
                    detection.height,
                    connector,
                )
            )
        group_summary = ""
        if card_group is not None:
            expected = str(card_group.expected_count) if card_group.expected_count > 0 else "auto"
            group_summary = " group_observed=%d group_expected=%s group_complete=%s group_bbox=%s" % (
                card_group.observed_count,
                expected,
                str(card_group.complete).lower(),
                str(card_group.bbox),
            )
        rospy.loginfo(
            "[usb_c_insertion] event=usb_card_detector_result count=%d%s %s",
            len(detections),
            group_summary,
            " ".join(summaries),
        )

    def _card_to_dict(self, index: int, detection: CardDetection) -> dict:
        connector = None
        if detection.connector is not None:
            connector = {
                "center_x": round(detection.connector.center_x, 3),
                "center_y": round(detection.connector.center_y, 3),
                "area": round(detection.connector.area, 3),
                "bbox": list(detection.connector.bbox),
                "center_offset_norm": round(detection.connector.center_offset_norm, 4),
            }
        return {
            "index": index,
            "center_x": round(detection.center_x, 3),
            "center_y": round(detection.center_y, 3),
            "width": round(detection.width, 3),
            "height": round(detection.height, 3),
            "angle_deg": round(detection.angle_deg, 3),
            "area": round(detection.area, 3),
            "rectangularity": round(detection.rectangularity, 4),
            "aspect_ratio": round(detection.aspect_ratio, 4),
            "vertical_error_deg": round(detection.vertical_error_deg, 3),
            "score": round(detection.score, 3),
            "box_points": [list(point) for point in detection.box_points],
            "connector": connector,
        }

    @staticmethod
    def _card_group_to_dict(card_group: Optional[CardGroupEstimate]) -> Optional[dict]:
        if card_group is None:
            return None
        return {
            "observed_count": int(card_group.observed_count),
            "expected_count": int(card_group.expected_count),
            "complete": bool(card_group.complete),
            "bbox": list(card_group.bbox),
            "center_x": round(card_group.center_x, 3),
            "center_y": round(card_group.center_y, 3),
            "width": round(card_group.width, 3),
            "height": round(card_group.height, 3),
            "order_axis": card_group.order_axis,
            "order_direction": card_group.order_direction,
            "slot_centers": [
                {"index": index + 1, "center_x": round(center[0], 3), "center_y": round(center[1], 3)}
                for index, center in enumerate(card_group.slot_centers)
            ],
        }

    def _image_to_bgr(self, msg: Image) -> np.ndarray:
        encoding = msg.encoding.lower().strip()
        channels = 0
        conversion = None
        if encoding in ("bgr8", "rgb8"):
            channels = 3
        elif encoding in ("bgra8", "rgba8"):
            channels = 4
        elif encoding in ("mono8", "8uc1"):
            channels = 1
        else:
            raise ValueError("unsupported_image_encoding:%s" % msg.encoding)

        data = np.frombuffer(msg.data, dtype=np.uint8)
        expected = int(msg.height) * int(msg.step)
        if data.size < expected:
            raise ValueError("image_data_too_short")
        image = data[:expected].reshape((int(msg.height), int(msg.step)))
        image = image[:, : int(msg.width) * channels]
        image = image.reshape((int(msg.height), int(msg.width), channels))

        if encoding == "bgr8":
            return image.copy()
        if encoding == "rgb8":
            conversion = cv2.COLOR_RGB2BGR
        elif encoding == "bgra8":
            conversion = cv2.COLOR_BGRA2BGR
        elif encoding == "rgba8":
            conversion = cv2.COLOR_RGBA2BGR
        else:
            conversion = cv2.COLOR_GRAY2BGR
        return cv2.cvtColor(image, conversion)

    def _bgr_to_image_msg(self, bgr: np.ndarray, frame_id: str) -> Image:
        msg = Image()
        msg.header.stamp = rospy.Time.now()
        msg.header.frame_id = frame_id
        msg.height = int(bgr.shape[0])
        msg.width = int(bgr.shape[1])
        msg.encoding = "bgr8"
        msg.is_bigendian = False
        msg.step = int(bgr.shape[1] * 3)
        msg.data = bgr.astype(np.uint8).tobytes()
        return msg

    def _rotate_image_for_processing(self, bgr: np.ndarray) -> np.ndarray:
        if self._image_rotation_deg == 0.0:
            return bgr
        if self._image_rotation_deg == 90.0:
            return cv2.rotate(bgr, cv2.ROTATE_90_CLOCKWISE)
        if self._image_rotation_deg == 180.0:
            return cv2.rotate(bgr, cv2.ROTATE_180)
        return cv2.rotate(bgr, cv2.ROTATE_90_COUNTERCLOCKWISE)

    @staticmethod
    def _normalize_image_rotation_deg(rotation_deg: float) -> float:
        normalized = float(rotation_deg) % 360.0
        allowed = (0.0, 90.0, 180.0, 270.0)
        closest = min(allowed, key=lambda value: abs(value - normalized))
        if abs(closest - normalized) > 1e-3:
            rospy.logwarn(
                "[usb_c_insertion] event=usb_card_detector_rotation_rounded requested=%.1f used=%.1f",
                rotation_deg,
                closest,
            )
        return closest

    @staticmethod
    def _str_param(name: str, default: str) -> str:
        return str(get_param(name, default)).strip()

    @staticmethod
    def _float_param(name: str, default: float) -> float:
        return float(get_param(name, default))

    @staticmethod
    def _int_param(name: str, default: int) -> int:
        return int(get_param(name, default))

    @staticmethod
    def _bool_param(name: str, default: bool) -> bool:
        value = get_param(name, default)
        return UsbCardDetectorNode._coerce_bool(value)

    @staticmethod
    def _private_bool_override(private_name: str, config_name: str, default: bool) -> bool:
        if rospy.has_param(private_name):
            return UsbCardDetectorNode._coerce_bool(rospy.get_param(private_name))
        return UsbCardDetectorNode._bool_param(config_name, default)

    @staticmethod
    def _coerce_bool(value) -> bool:
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            normalized = value.strip().lower()
            if normalized in ("true", "1", "yes", "on"):
                return True
            if normalized in ("false", "0", "no", "off"):
                return False
        return bool(value)


def main() -> None:
    rospy.init_node("usb_c_insertion_usb_card_detector")
    UsbCardDetectorNode()
    rospy.spin()


if __name__ == "__main__":
    main()
