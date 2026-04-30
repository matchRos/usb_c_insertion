#!/usr/bin/env python3

from __future__ import annotations

import rospy


def required_param(name: str):
    if rospy.has_param(name):
        return rospy.get_param(name)
    resolved_name = rospy.resolve_name(name)
    rospy.logerr(
        "[usb_c_insertion] event=missing_required_param param=%s resolved_param=%s",
        name,
        resolved_name,
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

