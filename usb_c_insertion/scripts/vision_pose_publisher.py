#!/usr/bin/env python3

from __future__ import annotations

import os
import sys

from geometry_msgs.msg import PoseStamped
import rospy

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from param_utils import get_param
from vision_pose_loader import load_vision_pose_from_json


def main() -> None:
    rospy.init_node("usb_c_insertion_vision_pose_publisher")

    json_path = str(get_param("~vision_pose_json_path", "")).strip()
    topic = str(get_param("~topics/vision_pose_debug", "/usb_c_insertion/vision_pose_debug"))
    base_frame = str(get_param("~frames/base_frame", "base_link"))
    publish_rate = float(get_param("~vision_pose_publish_rate", 2.0))

    publisher = rospy.Publisher(topic, PoseStamped, queue_size=1, latch=True)
    rate = rospy.Rate(max(0.2, publish_rate))

    while not rospy.is_shutdown():
        try:
            vision_pose = load_vision_pose_from_json(json_path)
        except (OSError, ValueError, KeyError, TypeError) as exc:
            rospy.logwarn_throttle(
                2.0,
                "[usb_c_insertion] event=vision_pose_publish_failed path=%s reason=%s",
                json_path,
                exc,
            )
            rate.sleep()
            continue

        msg = PoseStamped()
        msg.header.stamp = rospy.Time.now()
        msg.header.frame_id = base_frame
        msg.pose.position.x = vision_pose.x
        msg.pose.position.y = vision_pose.y
        msg.pose.position.z = vision_pose.z
        msg.pose.orientation.x = vision_pose.qx
        msg.pose.orientation.y = vision_pose.qy
        msg.pose.orientation.z = vision_pose.qz
        msg.pose.orientation.w = vision_pose.qw
        publisher.publish(msg)
        rospy.loginfo_throttle(
            2.0,
            "[usb_c_insertion] event=vision_pose_published topic=%s x=%.4f y=%.4f z=%.4f yaw_rad=%.4f",
            topic,
            vision_pose.x,
            vision_pose.y,
            vision_pose.z,
            vision_pose.yaw_rad,
        )
        rate.sleep()


if __name__ == "__main__":
    main()
