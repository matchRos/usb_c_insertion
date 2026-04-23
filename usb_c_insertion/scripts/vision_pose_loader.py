#!/usr/bin/env python3

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from typing import Any, Dict

YAW_OFFSET_RAD = -0.5 * math.pi


@dataclass(frozen=True)
class VisionPose:
    x: float
    y: float
    z: float
    qx: float
    qy: float
    qz: float
    qw: float
    yaw_rad: float


def load_vision_pose_from_json(json_path: str) -> VisionPose:
    """
    Load a coarse port pose from the vision JSON output.

    Only the yaw around the base z-axis is used from the orientation because
    the PC is assumed to rest on the table. Roll and pitch are ignored.
    """
    with open(json_path, "r", encoding="utf-8") as handle:
        data = json.load(handle)

    position = _require_mapping(data, "position")
    yaw_rad = _normalize_angle(_extract_yaw_rad(data) + YAW_OFFSET_RAD)
    qx, qy, qz, qw = _quaternion_from_yaw(yaw_rad)
    transformed_x, transformed_y, transformed_z = _transform_position_axes(
        float(position["x"]),
        float(position["y"]),
        float(position["z"]),
    )
    return VisionPose(
        x=transformed_x,
        y=transformed_y,
        z=transformed_z,
        qx=qx,
        qy=qy,
        qz=qz,
        qw=qw,
        yaw_rad=yaw_rad,
    )


def _extract_yaw_rad(data: Dict[str, Any]) -> float:
    orientation = _require_mapping(data, "orientation")

    euler_xyz_deg = orientation.get("euler_xyz_deg")
    if isinstance(euler_xyz_deg, list) and len(euler_xyz_deg) >= 3:
        return math.radians(float(euler_xyz_deg[2]))

    quaternion_xyzw = orientation.get("quaternion_xyzw")
    if isinstance(quaternion_xyzw, list) and len(quaternion_xyzw) >= 4:
        return _yaw_from_quaternion(
            float(quaternion_xyzw[0]),
            float(quaternion_xyzw[1]),
            float(quaternion_xyzw[2]),
            float(quaternion_xyzw[3]),
        )

    raise ValueError("orientation must contain either euler_xyz_deg or quaternion_xyzw")


def _require_mapping(data: Dict[str, Any], key: str) -> Dict[str, Any]:
    value = data.get(key)
    if not isinstance(value, dict):
        raise ValueError("%s must be a mapping" % key)
    return value


def _transform_position_axes(x: float, y: float, z: float):
    """
    Transform the vision position into the robot-side convention.

    Required temporary mapping:
    - x stays unchanged
    - new y becomes old z
    - new z becomes negative old y
    """
    return (x, y, z)


def _quaternion_from_yaw(yaw_rad: float):
    half_yaw = 0.5 * yaw_rad
    return (0.0, 0.0, math.sin(half_yaw), math.cos(half_yaw))


def _yaw_from_quaternion(qx: float, qy: float, qz: float, qw: float) -> float:
    siny_cosp = 2.0 * (qw * qz + qx * qy)
    cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)
    return math.atan2(siny_cosp, cosy_cosp)


def _normalize_angle(angle_rad: float) -> float:
    while angle_rad > math.pi:
        angle_rad -= 2.0 * math.pi
    while angle_rad < -math.pi:
        angle_rad += 2.0 * math.pi
    return angle_rad
