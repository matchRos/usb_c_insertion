#!/usr/bin/env python3

from __future__ import annotations

import math
import os
import sys
from typing import Dict

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from ft_interface import FTInterface, WrenchData


class ContactDetector:
    """
    Contact detection logic based on force increase relative to a rolling baseline.

    The detector depends only on the force-torque interface and does not send any
    motion commands. This keeps sensing decisions separate from robot control.
    """

    def __init__(
        self,
        ft_interface: FTInterface,
        hysteresis: float = 0.0,
    ):
        self._ft_interface = ft_interface
        self._hysteresis = max(0.0, float(hysteresis))
        self._baseline = WrenchData(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        self._axis_contact_state: Dict[str, bool] = {"x": False, "y": False, "z": False}
        self._norm_contact_state = False

    def update_baseline(self) -> WrenchData:
        """
        Store a new baseline from the current filtered wrench estimate.

        The rolling average is computed inside FTInterface. This method only
        captures the current baseline snapshot used for contact comparison.
        """
        estimated = self._ft_interface.estimate_baseline()
        if estimated is not None:
            self._baseline = estimated
        return self._baseline

    def detect_contact_along_axis(self, axis_name: str, threshold: float) -> bool:
        """
        Detect contact from the force rise along one Cartesian axis.

        Supported axes are x, y, and z. The comparison uses the filtered wrench
        to stay conservative during noisy real robot experiments.
        """
        axis = axis_name.lower().strip()
        if axis not in self._axis_contact_state:
            raise ValueError("axis_name must be one of: x, y, z")

        filtered_wrench = self._ft_interface.get_filtered_wrench()
        force_delta = abs(self._get_force_component(filtered_wrench, axis) - self._get_force_component(self._baseline, axis))
        new_state = self._apply_hysteresis(
            previous_state=self._axis_contact_state[axis],
            signal_value=force_delta,
            threshold=float(threshold),
        )
        self._axis_contact_state[axis] = new_state
        return new_state

    def detect_contact_norm(self, threshold: float) -> bool:
        """
        Detect contact from the Euclidean norm of the force rise vector.
        """
        filtered_wrench = self._ft_interface.get_filtered_wrench()
        delta_x = filtered_wrench.force_x - self._baseline.force_x
        delta_y = filtered_wrench.force_y - self._baseline.force_y
        delta_z = filtered_wrench.force_z - self._baseline.force_z
        force_norm_delta = math.sqrt(delta_x * delta_x + delta_y * delta_y + delta_z * delta_z)

        self._norm_contact_state = self._apply_hysteresis(
            previous_state=self._norm_contact_state,
            signal_value=force_norm_delta,
            threshold=float(threshold),
        )
        return self._norm_contact_state

    @staticmethod
    def _get_force_component(wrench: WrenchData, axis_name: str) -> float:
        if axis_name == "x":
            return wrench.force_x
        if axis_name == "y":
            return wrench.force_y
        return wrench.force_z

    def _apply_hysteresis(self, previous_state: bool, signal_value: float, threshold: float) -> bool:
        """
        Apply a simple Schmitt-trigger style hysteresis.

        Once contact is active, the signal has to drop below threshold minus the
        hysteresis margin before the detector clears again.
        """
        threshold = max(0.0, threshold)
        if previous_state:
            release_threshold = max(0.0, threshold - self._hysteresis)
            return signal_value >= release_threshold
        return signal_value >= threshold
