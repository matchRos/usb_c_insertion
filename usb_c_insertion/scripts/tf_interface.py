#!/usr/bin/env python3

from __future__ import annotations

from typing import Optional

import rospy
import tf2_ros
from geometry_msgs.msg import PoseStamped, TransformStamped


class TFInterface:
    """
    Small wrapper around tf2_ros lookups for the insertion pipeline.

    The class keeps frame naming and exception handling in one place so the
    state machine can depend on a simple, robot-focused API.
    """

    def __init__(self, tf_timeout: float = 0.2, cache_time: float = 5.0):
        self._base_frame = rospy.get_param("~frames/base_frame", "base_link")
        self._tool_frame = rospy.get_param("~frames/tool_frame", "tool0")
        self._tf_timeout = rospy.Duration.from_sec(max(0.0, float(tf_timeout)))

        self._buffer = tf2_ros.Buffer(cache_time=rospy.Duration.from_sec(max(0.1, float(cache_time))))
        self._listener = tf2_ros.TransformListener(self._buffer)

    def get_tool_transform(self) -> Optional[TransformStamped]:
        """
        Return the latest transform from base frame to tool frame.
        """
        return self.lookup_transform(self._base_frame, self._tool_frame)

    def get_tool_pose_in_base(self) -> Optional[PoseStamped]:
        """
        Return the tool pose expressed in the base frame.

        The pose contains the translation and orientation from the TF lookup
        and uses the transform timestamp when available.
        """
        transform = self.get_tool_transform()
        if transform is None:
            return None

        pose = PoseStamped()
        pose.header.stamp = transform.header.stamp
        pose.header.frame_id = transform.header.frame_id
        pose.pose.position.x = transform.transform.translation.x
        pose.pose.position.y = transform.transform.translation.y
        pose.pose.position.z = transform.transform.translation.z
        pose.pose.orientation = transform.transform.rotation
        return pose

    def lookup_transform(
        self,
        target_frame: str,
        source_frame: str,
    ) -> Optional[TransformStamped]:
        """
        Look up the latest available transform between two frames.

        Returning None keeps callers safe when TF is temporarily unavailable
        instead of forcing exception handling into every control module.
        """
        try:
            return self._buffer.lookup_transform(
                target_frame,
                source_frame,
                rospy.Time(0),
                self._tf_timeout,
            )
        except (
            tf2_ros.LookupException,
            tf2_ros.ConnectivityException,
            tf2_ros.ExtrapolationException,
            tf2_ros.TimeoutException,
        ) as exc:
            rospy.logwarn_throttle(
                2.0,
                "TF lookup failed from '%s' to '%s': %s",
                source_frame,
                target_frame,
                exc,
            )
            return None
