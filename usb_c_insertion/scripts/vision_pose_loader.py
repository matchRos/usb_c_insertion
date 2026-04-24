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
    port_axis_xy = _extract_port_axis_xy(data)
    yaw_rad = math.atan2(port_axis_xy[1], port_axis_xy[0])
    qx, qy, qz, qw = _quaternion_from_port_axis(port_axis_xy)
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


def _extract_port_axis_xy(data: Dict[str, Any]) -> tuple[float, float]:
    plane = data.get("plane")
    if isinstance(plane, dict):
        normal = plane.get("normal")
        if isinstance(normal, dict):
            axis_xy = _axis_xy_from_plane_normal(normal)
            if axis_xy is not None:
                return axis_xy

    orientation = _require_mapping(data, "orientation")

    euler_xyz_deg = orientation.get("euler_xyz_deg")
    if isinstance(euler_xyz_deg, list) and len(euler_xyz_deg) >= 3:
        yaw_rad = math.radians(float(euler_xyz_deg[2]))
        return (math.cos(yaw_rad), math.sin(yaw_rad))

    quaternion_xyzw = orientation.get("quaternion_xyzw")
    if isinstance(quaternion_xyzw, list) and len(quaternion_xyzw) >= 4:
        yaw_rad = _yaw_from_quaternion(
            float(quaternion_xyzw[0]),
            float(quaternion_xyzw[1]),
            float(quaternion_xyzw[2]),
            float(quaternion_xyzw[3]),
        )
        return (math.cos(yaw_rad), math.sin(yaw_rad))

    raise ValueError("orientation must contain either euler_xyz_deg or quaternion_xyzw")


def _axis_xy_from_plane_normal(normal: Dict[str, Any]) -> tuple[float, float] | None:
    """
    Extract the projected case x-axis from the fitted plane normal.

    The perception frame may be arbitrarily rolled around that axis, so we use
    only the direction that points out of the case wall and ignore the other
    axes entirely.
    """
    try:
        normal_x = float(normal["x"])
        normal_y = float(normal["y"])
    except (KeyError, TypeError, ValueError):
        return None

    xy_norm = math.hypot(normal_x, normal_y)
    if xy_norm < 1e-6:
        return None

    return (normal_x / xy_norm, normal_y / xy_norm)


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


def _quaternion_from_port_axis(port_axis_xy: tuple[float, float]):
    """
    Build a case-frame quaternion from the horizontal port axis and world up.

    Convention:
    - x_case points toward the port
    - z_case points upward against gravity
    - y_case follows the right-handed frame
    """
    axis_x = (port_axis_xy[0], port_axis_xy[1], 0.0)
    axis_z = (0.0, 0.0, 1.0)
    axis_y = _normalize_vector(_cross(axis_z, axis_x))
    rotation_matrix = (
        (axis_x[0], axis_y[0], axis_z[0]),
        (axis_x[1], axis_y[1], axis_z[1]),
        (axis_x[2], axis_y[2], axis_z[2]),
    )
    return _quaternion_from_rotation_matrix(rotation_matrix)


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


def _cross(vector_a: tuple[float, float, float], vector_b: tuple[float, float, float]):
    return (
        vector_a[1] * vector_b[2] - vector_a[2] * vector_b[1],
        vector_a[2] * vector_b[0] - vector_a[0] * vector_b[2],
        vector_a[0] * vector_b[1] - vector_a[1] * vector_b[0],
    )


def _normalize_vector(vector_xyz: tuple[float, float, float]):
    magnitude = math.sqrt(sum(component * component for component in vector_xyz))
    if magnitude <= 1e-9:
        raise ValueError("vector must be non-zero")
    return tuple(component / magnitude for component in vector_xyz)


def _quaternion_from_rotation_matrix(matrix_rows):
    m00, m01, m02 = matrix_rows[0]
    m10, m11, m12 = matrix_rows[1]
    m20, m21, m22 = matrix_rows[2]
    trace = m00 + m11 + m22

    if trace > 0.0:
        s = math.sqrt(trace + 1.0) * 2.0
        qw = 0.25 * s
        qx = (m21 - m12) / s
        qy = (m02 - m20) / s
        qz = (m10 - m01) / s
    elif m00 > m11 and m00 > m22:
        s = math.sqrt(1.0 + m00 - m11 - m22) * 2.0
        qw = (m21 - m12) / s
        qx = 0.25 * s
        qy = (m01 + m10) / s
        qz = (m02 + m20) / s
    elif m11 > m22:
        s = math.sqrt(1.0 + m11 - m00 - m22) * 2.0
        qw = (m02 - m20) / s
        qx = (m01 + m10) / s
        qy = 0.25 * s
        qz = (m12 + m21) / s
    else:
        s = math.sqrt(1.0 + m22 - m00 - m11) * 2.0
        qw = (m10 - m01) / s
        qx = (m02 + m20) / s
        qy = (m12 + m21) / s
        qz = 0.25 * s

    norm = math.sqrt(qx * qx + qy * qy + qz * qz + qw * qw)
    if norm <= 1e-9:
        return (0.0, 0.0, 0.0, 1.0)
    return (qx / norm, qy / norm, qz / norm, qw / norm)
