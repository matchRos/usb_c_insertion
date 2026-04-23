#!/usr/bin/env python3

from __future__ import annotations

import math


def compute_port_frame_target(port_pose_xyzw, offset_xyz):
    px, py, pz, qx, qy, qz, qw = port_pose_xyzw
    ox, oy, oz = offset_xyz
    rx, ry, rz = rotate_vector_by_quaternion(ox, oy, oz, qx, qy, qz, qw)
    return (px + rx, py + ry, pz + rz)


def compute_tcp_target_orientation(current_tool_quaternion_xyzw, port_quaternion_xyzw):
    """
    Build the TCP yaw from the perceived case plane.

    The plane x-axis points out of the case wall. For the approach pose we use
    that projected direction directly as the TCP x-axis yaw reference in the
    robot XY plane. Only the yaw around robot z matters here.
    """
    target_yaw = compute_tcp_target_yaw(port_quaternion_xyzw)
    roll, pitch, _ = euler_from_quaternion(current_tool_quaternion_xyzw)
    return quaternion_from_euler(roll, pitch, target_yaw)


def compute_tcp_target_yaw(port_quaternion_xyzw):
    """
    Return only the desired TCP yaw derived from the perceived case plane.
    """
    plane_x_in_base = rotate_vector_by_quaternion(
        1.0,
        0.0,
        0.0,
        port_quaternion_xyzw[0],
        port_quaternion_xyzw[1],
        port_quaternion_xyzw[2],
        port_quaternion_xyzw[3],
    )
    plane_x_yaw = math.atan2(plane_x_in_base[1], plane_x_in_base[0])
    return normalize_angle(plane_x_yaw)


def rotate_vector_by_quaternion(
    vector_x: float,
    vector_y: float,
    vector_z: float,
    quat_x: float,
    quat_y: float,
    quat_z: float,
    quat_w: float,
):
    norm = math.sqrt(quat_x * quat_x + quat_y * quat_y + quat_z * quat_z + quat_w * quat_w)
    if norm <= 1e-9:
        raise ValueError("Port quaternion must be non-zero.")

    qx = quat_x / norm
    qy = quat_y / norm
    qz = quat_z / norm
    qw = quat_w / norm

    rotation_matrix = (
        (
            1.0 - 2.0 * (qy * qy + qz * qz),
            2.0 * (qx * qy - qz * qw),
            2.0 * (qx * qz + qy * qw),
        ),
        (
            2.0 * (qx * qy + qz * qw),
            1.0 - 2.0 * (qx * qx + qz * qz),
            2.0 * (qy * qz - qx * qw),
        ),
        (
            2.0 * (qx * qz - qy * qw),
            2.0 * (qy * qz + qx * qw),
            1.0 - 2.0 * (qx * qx + qy * qy),
        ),
    )

    return (
        rotation_matrix[0][0] * vector_x + rotation_matrix[0][1] * vector_y + rotation_matrix[0][2] * vector_z,
        rotation_matrix[1][0] * vector_x + rotation_matrix[1][1] * vector_y + rotation_matrix[1][2] * vector_z,
        rotation_matrix[2][0] * vector_x + rotation_matrix[2][1] * vector_y + rotation_matrix[2][2] * vector_z,
    )


def quaternion_multiply(first_xyzw, second_xyzw):
    x1, y1, z1, w1 = first_xyzw
    x2, y2, z2, w2 = second_xyzw
    return (
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
    )


def normalize_quaternion(quaternion_xyzw):
    norm = math.sqrt(sum(component * component for component in quaternion_xyzw))
    if norm <= 1e-9:
        return (0.0, 0.0, 0.0, 1.0)
    return tuple(component / norm for component in quaternion_xyzw)


def quaternion_from_yaw(yaw_rad: float):
    half_yaw = 0.5 * yaw_rad
    return (0.0, 0.0, math.sin(half_yaw), math.cos(half_yaw))


def quaternion_from_euler(roll_rad: float, pitch_rad: float, yaw_rad: float):
    half_roll = 0.5 * roll_rad
    half_pitch = 0.5 * pitch_rad
    half_yaw = 0.5 * yaw_rad

    sin_roll = math.sin(half_roll)
    cos_roll = math.cos(half_roll)
    sin_pitch = math.sin(half_pitch)
    cos_pitch = math.cos(half_pitch)
    sin_yaw = math.sin(half_yaw)
    cos_yaw = math.cos(half_yaw)

    qx = sin_roll * cos_pitch * cos_yaw - cos_roll * sin_pitch * sin_yaw
    qy = cos_roll * sin_pitch * cos_yaw + sin_roll * cos_pitch * sin_yaw
    qz = cos_roll * cos_pitch * sin_yaw - sin_roll * sin_pitch * cos_yaw
    qw = cos_roll * cos_pitch * cos_yaw + sin_roll * sin_pitch * sin_yaw
    return normalize_quaternion((qx, qy, qz, qw))


def euler_from_quaternion(quaternion_xyzw):
    qx, qy, qz, qw = normalize_quaternion(quaternion_xyzw)

    sinr_cosp = 2.0 * (qw * qx + qy * qz)
    cosr_cosp = 1.0 - 2.0 * (qx * qx + qy * qy)
    roll = math.atan2(sinr_cosp, cosr_cosp)

    sinp = 2.0 * (qw * qy - qz * qx)
    if abs(sinp) >= 1.0:
        pitch = math.copysign(0.5 * math.pi, sinp)
    else:
        pitch = math.asin(sinp)

    siny_cosp = 2.0 * (qw * qz + qx * qy)
    cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)
    yaw = math.atan2(siny_cosp, cosy_cosp)
    return (roll, pitch, yaw)


def quaternion_from_rotation_matrix(matrix_rows):
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

    return normalize_quaternion((qx, qy, qz, qw))


def normalize_angle(angle_rad: float):
    while angle_rad > math.pi:
        angle_rad -= 2.0 * math.pi
    while angle_rad < -math.pi:
        angle_rad += 2.0 * math.pi
    return angle_rad
