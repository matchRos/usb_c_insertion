#!/usr/bin/env python3

from __future__ import annotations

from dataclasses import dataclass
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


def _validate_inputs(step_x: float, step_y: float, width: float, height: float) -> None:
    if step_x <= 0.0 or step_y <= 0.0:
        raise ValueError("step_x and step_y must be positive")
    if width < 0.0 or height < 0.0:
        raise ValueError("width and height must be non-negative")
