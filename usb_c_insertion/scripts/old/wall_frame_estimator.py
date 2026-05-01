#!/usr/bin/env python3

from __future__ import annotations

from dataclasses import dataclass
import math


@dataclass(frozen=True)
class WallYawEstimate:
    """
    Simple wall orientation estimate from two contact points.
    """

    wall_yaw: float
    wall_direction_x: float
    wall_direction_y: float
    wall_normal_x: float
    wall_normal_y: float


def estimate_wall_yaw(point_1, point_2) -> WallYawEstimate:
    """
    Estimate the wall tangent and normal in the base xy plane.

    The function is intentionally ROS-independent apart from expecting point-like
    objects with x and y attributes.
    """
    delta_x = point_2.x - point_1.x
    delta_y = point_2.y - point_1.y
    length = math.hypot(delta_x, delta_y)
    if length <= 1e-9:
        raise ValueError("Wall yaw cannot be estimated from identical probe points.")

    tangent_x = delta_x / length
    tangent_y = delta_y / length
    normal_x = -tangent_y
    normal_y = tangent_x
    wall_yaw = math.atan2(tangent_y, tangent_x)
    return WallYawEstimate(
        wall_yaw=wall_yaw,
        wall_direction_x=tangent_x,
        wall_direction_y=tangent_y,
        wall_normal_x=normal_x,
        wall_normal_y=normal_y,
    )
