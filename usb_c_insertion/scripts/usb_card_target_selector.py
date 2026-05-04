#!/usr/bin/env python3

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from typing import Any, Dict, List, Optional, Tuple

import rospy

from param_utils import get_param


@dataclass(frozen=True)
class UsbCardTarget:
    stamp: rospy.Time
    image_width: int
    image_height: int
    found: bool
    center_x: float = 0.0
    center_y: float = 0.0
    area: float = 0.0
    aspect_ratio: float = 0.0
    message: str = ""
    card_index: int = 0
    source_card_index: int = 0
    target_kind: str = ""
    bbox: Optional[Tuple[int, int, int, int]] = None


class UsbCardTargetSelector:
    def __init__(
        self,
        target_card_index: int,
        target_point: str,
        require_connector: bool,
        order_axis: str,
        order_direction: str,
        expected_card_count: int = 0,
        estimated_slot_requires_complete: bool = True,
    ):
        self.target_card_index = max(1, int(target_card_index))
        self.target_point = str(target_point).strip().lower() or "connector"
        self.require_connector = bool(require_connector)
        self.order_axis = str(order_axis).strip().lower() or "x"
        self.order_direction = str(order_direction).strip().lower() or "ascending"
        self.expected_card_count = max(0, int(expected_card_count))
        self.estimated_slot_requires_complete = bool(estimated_slot_requires_complete)

    @classmethod
    def from_ros_params(cls, namespace: str):
        prefix = "~%s/" % namespace.strip("/")
        return cls(
            target_card_index=int(get_param(prefix + "target_card_index", get_param("~workflow/target_card_index", 1))),
            target_point=str(get_param(prefix + "usb_card_target_point", get_param("~workflow/usb_card_target_point", "connector"))),
            require_connector=cls._bool_param(
                prefix + "usb_card_require_connector",
                cls._bool_param("~workflow/usb_card_require_connector", True),
            ),
            order_axis=str(get_param(prefix + "usb_card_order_axis", get_param("~workflow/usb_card_order_axis", "x"))),
            order_direction=str(
                get_param(
                    prefix + "usb_card_order_direction",
                    get_param("~workflow/usb_card_order_direction", "ascending"),
                )
            ),
            expected_card_count=int(
                get_param(
                    prefix + "usb_card_total_count",
                    get_param(
                        "~workflow/usb_card_total_count",
                        get_param("~usb_card_detector/expected_card_count", 0),
                    ),
                )
            ),
            estimated_slot_requires_complete=cls._bool_param(
                prefix + "usb_card_estimated_slot_requires_complete",
                cls._bool_param("~workflow/usb_card_estimated_slot_requires_complete", True),
            ),
        )

    def select_from_json(self, data: str) -> UsbCardTarget:
        try:
            payload = json.loads(str(data))
        except (TypeError, ValueError) as exc:
            return self._not_found("invalid_usb_card_json:%s" % exc)
        return self.select_from_payload(payload)

    def select_from_payload(self, payload: Dict[str, Any]) -> UsbCardTarget:
        stamp = self._stamp_from_payload(payload)
        image_width = int(payload.get("image_width", 0) or 0)
        image_height = int(payload.get("image_height", 0) or 0)
        cards = payload.get("cards", [])

        if self._uses_estimated_slot():
            return self._select_estimated_slot(payload, stamp, image_width, image_height)

        if not isinstance(cards, list) or not cards:
            return self._not_found("usb_card_not_found", stamp, image_width, image_height)

        ordered_cards = self._ordered_cards(cards)
        selected_index = self.target_card_index - 1
        if selected_index < 0 or selected_index >= len(ordered_cards):
            return self._not_found(
                "usb_card_index_unavailable:%d/%d" % (self.target_card_index, len(ordered_cards)),
                stamp,
                image_width,
                image_height,
            )

        card = ordered_cards[selected_index]
        connector = card.get("connector")
        use_connector = self.target_point in ("connector", "port", "usb_port")
        if use_connector and connector is None and self.require_connector:
            return self._not_found(
                "usb_card_connector_missing:%d" % self.target_card_index,
                stamp,
                image_width,
                image_height,
                card,
            )

        target = connector if use_connector and connector is not None else card
        center_x = float(target.get("center_x", 0.0) or 0.0)
        center_y = float(target.get("center_y", 0.0) or 0.0)
        area = float(card.get("area", target.get("area", 0.0)) or 0.0)
        aspect_ratio = self._card_aspect_ratio(card)
        bbox = self._bbox_from_target(target)
        return UsbCardTarget(
            stamp=stamp,
            image_width=image_width,
            image_height=image_height,
            found=True,
            center_x=center_x,
            center_y=center_y,
            area=area,
            aspect_ratio=aspect_ratio,
            message="usb_card_target_found",
            card_index=self.target_card_index,
            source_card_index=int(card.get("index", selected_index) or 0),
            target_kind="connector" if target is connector else "card_center",
            bbox=bbox,
        )

    def _uses_estimated_slot(self) -> bool:
        return self.target_point in (
            "estimated_slot",
            "slot",
            "slot_center",
            "group_slot",
            "group_fraction",
            "estimated_card_center",
        )

    def with_target(self, target_point: str, require_connector: bool) -> "UsbCardTargetSelector":
        return UsbCardTargetSelector(
            target_card_index=self.target_card_index,
            target_point=target_point,
            require_connector=require_connector,
            order_axis=self.order_axis,
            order_direction=self.order_direction,
            expected_card_count=self.expected_card_count,
            estimated_slot_requires_complete=self.estimated_slot_requires_complete,
        )

    def _select_estimated_slot(
        self,
        payload: Dict[str, Any],
        stamp: rospy.Time,
        image_width: int,
        image_height: int,
    ) -> UsbCardTarget:
        card_group = payload.get("card_group")
        if not isinstance(card_group, dict):
            return self._not_found("usb_card_group_missing", stamp, image_width, image_height)

        observed_count = int(card_group.get("observed_count", 0) or 0)
        group_expected_count = int(card_group.get("expected_count", 0) or 0)
        expected_count = self.expected_card_count or group_expected_count
        complete = bool(card_group.get("complete", False))
        if self.estimated_slot_requires_complete and expected_count > 0:
            if observed_count < expected_count and not complete:
                return self._not_found(
                    "usb_card_group_incomplete:%d/%d" % (observed_count, expected_count),
                    stamp,
                    image_width,
                    image_height,
                )

        slot_centers = card_group.get("slot_centers", [])
        if not isinstance(slot_centers, list) or not slot_centers:
            return self._not_found("usb_card_group_slots_missing", stamp, image_width, image_height)

        selected_index = self.target_card_index - 1
        if selected_index < 0 or selected_index >= len(slot_centers):
            return self._not_found(
                "usb_card_group_slot_unavailable:%d/%d" % (self.target_card_index, len(slot_centers)),
                stamp,
                image_width,
                image_height,
            )

        slot = slot_centers[selected_index]
        if not isinstance(slot, dict):
            return self._not_found("usb_card_group_slot_invalid", stamp, image_width, image_height)
        center_x = float(slot.get("center_x", 0.0) or 0.0)
        center_y = float(slot.get("center_y", 0.0) or 0.0)
        width = float(card_group.get("width", 0.0) or 0.0)
        height = float(card_group.get("height", 0.0) or 0.0)
        slot_count = max(1, len(slot_centers))
        if card_group.get("order_axis", self.order_axis) == "y":
            slot_width = width
            slot_height = height / float(slot_count)
        else:
            slot_width = width / float(slot_count)
            slot_height = height
        area = max(1.0, slot_width * slot_height)
        aspect_ratio = max(slot_width, slot_height) / max(1e-6, min(slot_width, slot_height))
        return UsbCardTarget(
            stamp=stamp,
            image_width=image_width,
            image_height=image_height,
            found=True,
            center_x=center_x,
            center_y=center_y,
            area=area,
            aspect_ratio=aspect_ratio,
            message="usb_card_estimated_slot_found:%d/%d" % (self.target_card_index, slot_count),
            card_index=self.target_card_index,
            source_card_index=selected_index,
            target_kind="estimated_slot",
        )

    @staticmethod
    def _bbox_from_target(target: Dict[str, Any]) -> Optional[Tuple[int, int, int, int]]:
        bbox = target.get("bbox")
        if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
            return None
        try:
            x, y, width, height = (int(round(float(value))) for value in bbox)
        except (TypeError, ValueError):
            return None
        if width <= 0 or height <= 0:
            return None
        return (x, y, width, height)

    def _ordered_cards(self, cards: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        axis = self.order_axis
        reverse = self.order_direction in ("descending", "desc", "reverse")

        def key(card: Dict[str, Any]):
            if axis == "y":
                return float(card.get("center_y", 0.0) or 0.0)
            if axis == "score":
                return float(card.get("score", 0.0) or 0.0)
            if axis == "index":
                return int(card.get("index", 0) or 0)
            return float(card.get("center_x", 0.0) or 0.0)

        return sorted(cards, key=key, reverse=reverse)

    @staticmethod
    def _card_aspect_ratio(card: Dict[str, Any]) -> float:
        value = float(card.get("aspect_ratio", 0.0) or 0.0)
        if value > 0.0:
            return value
        width = float(card.get("width", 0.0) or 0.0)
        height = float(card.get("height", 0.0) or 0.0)
        if width <= 0.0 or height <= 0.0:
            return 0.0
        return max(width, height) / max(1e-6, min(width, height))

    @staticmethod
    def _stamp_from_payload(payload: Dict[str, Any]) -> rospy.Time:
        stamp = payload.get("stamp", None)
        try:
            if stamp is not None and math.isfinite(float(stamp)):
                return rospy.Time.from_sec(float(stamp))
        except (TypeError, ValueError):
            pass
        return rospy.Time.now()

    @staticmethod
    def _bool_param(name: str, default: bool) -> bool:
        value = get_param(name, default)
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            normalized = value.strip().lower()
            if normalized in ("true", "1", "yes", "on"):
                return True
            if normalized in ("false", "0", "no", "off"):
                return False
        return bool(value)

    @staticmethod
    def _not_found(
        message: str,
        stamp: Optional[rospy.Time] = None,
        image_width: int = 0,
        image_height: int = 0,
        card: Optional[Dict[str, Any]] = None,
    ) -> UsbCardTarget:
        center_x = float(card.get("center_x", 0.0) or 0.0) if card else 0.0
        center_y = float(card.get("center_y", 0.0) or 0.0) if card else 0.0
        area = float(card.get("area", 0.0) or 0.0) if card else 0.0
        return UsbCardTarget(
            stamp=stamp or rospy.Time.now(),
            image_width=int(image_width),
            image_height=int(image_height),
            found=False,
            center_x=center_x,
            center_y=center_y,
            area=area,
            aspect_ratio=UsbCardTargetSelector._card_aspect_ratio(card or {}),
            message=message,
        )
