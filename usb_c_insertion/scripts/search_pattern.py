#!/usr/bin/env python3

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import List


@dataclass(frozen=True)
class PlanarOffset:
    """
    Incremental Cartesian offset inside the wall plane.

    dx and dy represent incremental steps in a local 2D wall-aligned frame.
    The caller can map these axes to the chosen wall-plane directions.
    """

    dx: float
    dy: float


def generate_raster_pattern(step_x: float, step_y: float, width: float, height: float) -> List[PlanarOffset]:
    """
    Generate a simple boustrophedon raster as incremental offsets.

    The returned list contains step-to-step increments, not absolute positions.
    This makes the pattern easy to feed into a motion loop and easy to unit test.
    """
    _validate_inputs(step_x, step_y, width, height)

    x_positions = _symmetric_positions(width * 0.5, step_x)
    y_positions = _forward_positions(height, step_y)

    absolute_points = [(0.0, 0.0)]
    reverse_row = False
    for row_index, y_value in enumerate(y_positions):
        if row_index == 0:
            row_x_positions = [x for x in x_positions if x != 0.0]
        else:
            row_x_positions = list(reversed(x_positions)) if reverse_row else list(x_positions)
        for x_value in row_x_positions:
            absolute_points.append((x_value, y_value))
        reverse_row = not reverse_row

    return _to_incremental_offsets(absolute_points)


def generate_centered_raster_pattern(
    step_x: float,
    step_y: float,
    width: float,
    height: float,
    preferred_x_sign: float = 1.0,
    preferred_y_sign: float = 1.0,
    diagonal_first: bool = False,
) -> List[PlanarOffset]:
    """
    Generate a raster that starts at the estimated center and expands by rows.

    This keeps the first motions close to the estimated port position while
    still covering the configured rectangular search window deterministically.
    """
    _validate_inputs(step_x, step_y, width, height)

    preferred_x_sign = _normalize_sign(preferred_x_sign)
    preferred_y_sign = _normalize_sign(preferred_y_sign)

    half_width = width * 0.5
    half_height = height * 0.5
    x_positions = _symmetric_positions(half_width, step_x)
    y_positions = _preferred_center_out_positions(
        half_height,
        step_y,
        preferred_y_sign,
        start_at_preferred=diagonal_first,
    )

    absolute_points = [(0.0, 0.0)]
    visited_points = {(0.0, 0.0)}
    current_x = 0.0
    for row_index, y_value in enumerate(y_positions):
        if row_index == 0 and (not diagonal_first or abs(y_value) <= 1e-9):
            row_x_positions = _preferred_axis_positions(half_width, step_x, preferred_x_sign, include_zero=False)
        elif row_index == 0:
            row_x_positions = _preferred_axis_positions(half_width, step_x, preferred_x_sign, include_zero=False)
            if not row_x_positions:
                row_x_positions = [0.0]
        else:
            ascending = list(x_positions)
            descending = list(reversed(x_positions))
            row_x_positions = ascending if abs(ascending[0] - current_x) <= abs(descending[0] - current_x) else descending

        for x_value in row_x_positions:
            point = (x_value, y_value)
            if point in visited_points:
                continue
            absolute_points.append(point)
            visited_points.add(point)
            current_x = x_value

    return _to_incremental_offsets(absolute_points)


def generate_expanding_square_pattern(step: float, max_radius: float) -> List[PlanarOffset]:
    """
    Generate an expanding square spiral as incremental offsets.

    The pattern starts at the origin and grows outward in the local wall plane.
    """
    if step <= 0.0:
        raise ValueError("step must be positive")
    if max_radius < 0.0:
        raise ValueError("max_radius must be non-negative")

    if max_radius == 0.0:
        return []

    absolute_points = [(0.0, 0.0)]
    current_x = 0.0
    current_y = 0.0
    leg_length = 1
    radius_limit = max_radius + 1e-9

    while True:
        for direction_x, direction_y in ((1.0, 0.0), (0.0, 1.0), (-1.0, 0.0), (0.0, -1.0)):
            for _ in range(leg_length):
                next_x = current_x + direction_x * step
                next_y = current_y + direction_y * step
                if max(abs(next_x), abs(next_y)) > radius_limit:
                    return _to_incremental_offsets(absolute_points)
                absolute_points.append((next_x, next_y))
                current_x = next_x
                current_y = next_y
            if direction_y != 0.0:
                leg_length += 1


def generate_expanding_circle_pattern(
    radial_step: float,
    max_radius_x: float,
    max_radius_y: float,
) -> List[PlanarOffset]:
    """
    Generate an expanding circle-based pattern around the origin.

    The pattern approximates growing circles clipped to the requested search
    extents, which makes it more efficient than a raster for larger search
    regions while remaining deterministic and easy to test.
    """
    if radial_step <= 0.0:
        raise ValueError("radial_step must be positive")
    if max_radius_x < 0.0 or max_radius_y < 0.0:
        raise ValueError("max_radius_x and max_radius_y must be non-negative")
    if max_radius_x == 0.0 and max_radius_y == 0.0:
        return []

    absolute_points = [(0.0, 0.0)]
    max_radius = max(max_radius_x, max_radius_y)
    radius = radial_step
    while radius <= max_radius + 1e-9:
        circumference = max(2.0 * math.pi * radius, radial_step)
        point_count = max(12, int(math.ceil(circumference / radial_step)))
        for index in range(point_count):
            theta = (2.0 * math.pi * index) / float(point_count)
            point_x = radius * math.cos(theta)
            point_y = radius * math.sin(theta)
            if abs(point_x) <= max_radius_x + 1e-9 and abs(point_y) <= max_radius_y + 1e-9:
                rounded_point = (round(point_x, 10), round(point_y, 10))
                if rounded_point != absolute_points[-1]:
                    absolute_points.append(rounded_point)
        radius += radial_step

    return _to_incremental_offsets(absolute_points)


def _to_incremental_offsets(absolute_points) -> List[PlanarOffset]:
    offsets: List[PlanarOffset] = []
    for index in range(1, len(absolute_points)):
        previous_x, previous_y = absolute_points[index - 1]
        current_x, current_y = absolute_points[index]
        offsets.append(PlanarOffset(dx=current_x - previous_x, dy=current_y - previous_y))
    return offsets


def _symmetric_positions(half_extent: float, step: float) -> List[float]:
    positions = [0.0]
    current = step
    while current <= half_extent + 1e-9:
        positions.extend([current, -current])
        current += step
    return sorted(set(round(value, 10) for value in positions))


def _forward_positions(max_extent: float, step: float) -> List[float]:
    positions = [0.0]
    current = step
    while current <= max_extent + 1e-9:
        positions.append(round(current, 10))
        current += step
    return positions


def _center_out_positions(half_extent: float, step: float) -> List[float]:
    positions = [0.0]
    current = step
    while current <= half_extent + 1e-9:
        positions.extend([round(current, 10), round(-current, 10)])
        current += step
    return positions


def _preferred_center_out_positions(
    half_extent: float,
    step: float,
    preferred_sign: float,
    start_at_preferred: bool,
) -> List[float]:
    signed_positions = []
    current = step
    while current <= half_extent + 1e-9:
        signed_positions.extend(
            [
                round(preferred_sign * current, 10),
                round(-preferred_sign * current, 10),
            ]
        )
        current += step

    if not signed_positions:
        return [0.0]
    if start_at_preferred:
        return [signed_positions[0], 0.0] + signed_positions[1:]
    return [0.0] + signed_positions


def _preferred_axis_positions(
    half_extent: float,
    step: float,
    preferred_sign: float,
    include_zero: bool,
) -> List[float]:
    preferred_positions = []
    opposite_positions = []
    current = step
    while current <= half_extent + 1e-9:
        preferred_positions.append(round(preferred_sign * current, 10))
        opposite_positions.append(round(-preferred_sign * current, 10))
        current += step

    positions = []
    if include_zero:
        positions.append(0.0)
    positions.extend(preferred_positions)
    positions.extend(opposite_positions)
    return positions


def _normalize_sign(value: float) -> float:
    return -1.0 if float(value) < 0.0 else 1.0


def _validate_inputs(step_x: float, step_y: float, width: float, height: float) -> None:
    if step_x <= 0.0 or step_y <= 0.0:
        raise ValueError("step_x and step_y must be positive")
    if width < 0.0 or height < 0.0:
        raise ValueError("width and height must be non-negative")
