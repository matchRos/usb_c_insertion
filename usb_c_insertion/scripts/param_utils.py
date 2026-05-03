#!/usr/bin/env python3

from __future__ import annotations

import rospy


GLOBAL_CONFIG_NAMESPACES = frozenset(
    (
        "frames",
        "topics",
        "motion",
        "micro_motion",
        "contact",
        "center_port",
        "looming",
        "housing_plane",
        "align_housing_yaw",
        "insert",
        "verify",
        "extract",
        "gripper",
        "precontact",
        "workflow",
        "photo_pose",
        "presentation_snapshots",
        "usb_card_detector",
        "insertion_workflow",
        "combined_workflow",
        "calibration",
        "move_to_pose_profiles",
        "pose_servo_profiles",
    )
)


def global_name_for_private(name: str) -> str:
    if not str(name).startswith("~"):
        return str(name)
    private_tail = str(name)[1:].lstrip("/")
    namespace = private_tail.split("/", 1)[0]
    if namespace not in GLOBAL_CONFIG_NAMESPACES:
        return ""
    return "/" + private_tail


def get_param(name: str, default=None):
    """
    Read package configuration from the global namespace first.

    Launch files load config YAMLs globally.  Keeping this helper global-first
    prevents stale private copies such as `/node/precontact/foo` from
    shadowing the single intended `/precontact/foo` value.
    """
    if str(name).startswith("~"):
        global_name = global_name_for_private(name)
        if global_name and rospy.has_param(global_name):
            return rospy.get_param(global_name)
    if rospy.has_param(name):
        return rospy.get_param(name)
    return default


def required_param(name: str):
    if str(name).startswith("~"):
        global_name = global_name_for_private(name)
        if global_name and rospy.has_param(global_name):
            return rospy.get_param(global_name)
    if rospy.has_param(name):
        return rospy.get_param(name)
    resolved_name = rospy.resolve_name(name)
    global_name = global_name_for_private(name) if str(name).startswith("~") else ""
    rospy.logerr(
        "[usb_c_insertion] event=missing_required_param param=%s resolved_param=%s global_param=%s",
        name,
        resolved_name,
        global_name,
    )
    raise RuntimeError("Missing required ROS parameter: %s (%s)" % (name, resolved_name))


def required_str_param(name: str) -> str:
    return str(required_param(name)).strip()


def required_float_param(name: str) -> float:
    return float(required_param(name))


def required_int_param(name: str) -> int:
    return int(required_param(name))


def required_bool_param(name: str) -> bool:
    value = required_param(name)
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in ("true", "1", "yes", "on"):
            return True
        if normalized in ("false", "0", "no", "off"):
            return False
        resolved_name = rospy.resolve_name(name)
        raise ValueError("Invalid boolean ROS parameter: %s (%s)=%r" % (name, resolved_name, value))
    return bool(value)


def required_vector_param(name: str, length: int = 3):
    value = required_param(name)
    if not isinstance(value, (list, tuple)) or len(value) != length:
        resolved_name = rospy.resolve_name(name)
        raise ValueError(
            "Invalid vector ROS parameter: %s (%s) expected length %d, got %r"
            % (name, resolved_name, length, value)
        )
    return tuple(float(component) for component in value)
