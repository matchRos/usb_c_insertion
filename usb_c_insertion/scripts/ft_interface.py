#!/usr/bin/env python3

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Deque, Optional

import rospy
from geometry_msgs.msg import WrenchStamped
from std_srvs.srv import Trigger


@dataclass(frozen=True)
class WrenchData:
    force_x: float
    force_y: float
    force_z: float
    torque_x: float
    torque_y: float
    torque_z: float


class FTInterface:
    """
    Safe force-torque interface for ROS Noetic.

    This class keeps the latest wrench sample, exposes a moving-average filtered
    wrench, detects stale data, estimates a baseline, and optionally zeroes the
    hardware force-torque sensor through a ROS service.
    """

    def __init__(
        self,
        wrench_topic: str = "/wrench",
        filter_window_size: int = 20,
        wrench_timeout: float = 0.2,
        zero_service_name: str = "/ur_hardware_interface/zero_ftsensor",
    ):
        self._wrench_topic = wrench_topic
        self._filter_window_size = max(1, int(filter_window_size))
        self._wrench_timeout = rospy.Duration.from_sec(max(0.0, float(wrench_timeout)))
        self._zero_service_name = zero_service_name

        self._latest_wrench = WrenchData(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        self._latest_stamp: Optional[rospy.Time] = None
        self._baseline_wrench = WrenchData(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        self._wrench_window: Deque[WrenchData] = deque(maxlen=self._filter_window_size)

        self._subscriber = rospy.Subscriber(
            self._wrench_topic,
            WrenchStamped,
            self._wrench_callback,
            queue_size=10,
        )

    def _wrench_callback(self, msg: WrenchStamped) -> None:
        """
        Store the newest wrench sample and keep a bounded filter window.
        """
        stamp = msg.header.stamp
        if stamp == rospy.Time():
            stamp = rospy.Time.now()

        wrench = WrenchData(
            force_x=msg.wrench.force.x,
            force_y=msg.wrench.force.y,
            force_z=msg.wrench.force.z,
            torque_x=msg.wrench.torque.x,
            torque_y=msg.wrench.torque.y,
            torque_z=msg.wrench.torque.z,
        )

        self._latest_stamp = stamp
        self._latest_wrench = wrench
        self._wrench_window.append(wrench)

    def get_latest_wrench(self) -> WrenchData:
        """
        Return the most recent wrench sample, even if it may be stale.
        """
        return self._latest_wrench

    def get_filtered_wrench(self) -> WrenchData:
        """
        Return a moving-average filtered wrench.

        If no samples have been received yet, return a zero wrench.
        """
        if not self._wrench_window:
            return WrenchData(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

        sample_count = float(len(self._wrench_window))
        return WrenchData(
            force_x=sum(sample.force_x for sample in self._wrench_window) / sample_count,
            force_y=sum(sample.force_y for sample in self._wrench_window) / sample_count,
            force_z=sum(sample.force_z for sample in self._wrench_window) / sample_count,
            torque_x=sum(sample.torque_x for sample in self._wrench_window) / sample_count,
            torque_y=sum(sample.torque_y for sample in self._wrench_window) / sample_count,
            torque_z=sum(sample.torque_z for sample in self._wrench_window) / sample_count,
        )

    def is_wrench_stale(self) -> bool:
        """
        Report whether the newest wrench sample is older than the allowed timeout.
        """
        if self._latest_stamp is None:
            return True
        return (rospy.Time.now() - self._latest_stamp) > self._wrench_timeout

    def estimate_baseline(self) -> Optional[WrenchData]:
        """
        Estimate and store a baseline from the current moving-average window.

        Returning None makes it explicit that the caller does not yet have any
        valid data to rely on.
        """
        if not self._wrench_window:
            rospy.logwarn("Cannot estimate wrench baseline because no wrench data has been received yet.")
            return None

        self._baseline_wrench = self.get_filtered_wrench()
        return self._baseline_wrench

    def get_baseline_wrench(self) -> WrenchData:
        """
        Return the last stored baseline wrench.
        """
        return self._baseline_wrench

    def zero_sensor(self, service_timeout: float = 2.0) -> bool:
        """
        Request hardware FT zeroing through the configured ROS service.

        The call is wrapped with timeout handling so the robot does not block
        indefinitely when the service is unavailable.
        """
        try:
            rospy.wait_for_service(self._zero_service_name, timeout=service_timeout)
            zero_client = rospy.ServiceProxy(self._zero_service_name, Trigger)
            response = zero_client()
            if not response.success:
                rospy.logwarn("Force-torque sensor zeroing request was rejected: %s", response.message)
                return False
            rospy.loginfo("Force-torque sensor zeroing request sent successfully.")
            return True
        except (rospy.ROSException, rospy.ServiceException) as exc:
            rospy.logwarn("Failed to zero force-torque sensor: %s", exc)
            return False
