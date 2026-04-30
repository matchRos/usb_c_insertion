#!/usr/bin/env python3

from __future__ import annotations

import os
import sys

import rospy
from geometry_msgs.msg import PoseStamped

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from tf_interface import TFInterface
from param_utils import get_param


def main() -> None:
    rospy.init_node("usb_c_insertion_camera_pose_publisher")

    tf_interface = TFInterface()
    base_frame = str(get_param("~frames/base_frame", "base_link"))
    camera_frame = str(
        get_param("~camera_frame", "zedm_left_camera_optical_frame")
    ).strip()
    topic = str(
        get_param("~topics/camera_pose_debug", "/usb_c_insertion/camera_pose")
    ).strip()
    publish_rate = float(get_param("~camera_pose_publish_rate", 10.0))

    publisher = rospy.Publisher(topic, PoseStamped, queue_size=1)
    rate = rospy.Rate(max(0.5, publish_rate))

    while not rospy.is_shutdown():
        transform = tf_interface.lookup_transform(base_frame, camera_frame)
        if transform is None:
            rate.sleep()
            continue

        msg = PoseStamped()
        msg.header.stamp = transform.header.stamp if transform.header.stamp != rospy.Time() else rospy.Time.now()
        msg.header.frame_id = base_frame
        msg.pose.position.x = transform.transform.translation.x
        msg.pose.position.y = transform.transform.translation.y
        msg.pose.position.z = transform.transform.translation.z
        msg.pose.orientation = transform.transform.rotation

        publisher.publish(msg)
        rate.sleep()


if __name__ == "__main__":
    main()
