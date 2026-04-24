#!/usr/bin/env python3

from __future__ import annotations

import os
import sys

import rospy

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from extraction_controller import ExtractionController
from ft_interface import FTInterface
from robot_interface import RobotInterface
from tf_interface import TFInterface

def main() -> None:
    rospy.init_node("usb_c_insertion_extract_cable")

    robot = RobotInterface()
    tf_interface = TFInterface()
    ft_interface = FTInterface(
        wrench_topic=rospy.get_param("~topics/wrench", "/wrench"),
        filter_window_size=rospy.get_param("~contact/baseline_window", 20),
        wrench_timeout=rospy.get_param("~contact/wrench_timeout", 0.2),
    )
    controller = ExtractionController(robot, tf_interface, ft_interface)

    if not robot.wait_for_motion_pipeline(timeout=3.0, require_pose_servo=False):
        rospy.logerr("[usb_c_insertion] event=extract_cable_failed reason=motion_pipeline_not_ready")
        sys.exit(1)

    auto_zero_ft = bool(rospy.get_param("~extract/auto_zero_ft", True))
    if auto_zero_ft:
        if not ft_interface.zero_sensor():
            rospy.logerr("[usb_c_insertion] event=extract_cable_failed reason=zero_ft_failed")
            sys.exit(1)
        rospy.sleep(0.5)

    result = controller.extract()
    if result.success:
        rospy.loginfo(
            "[usb_c_insertion] event=extract_cable_complete extracted_distance=%.4f pull_force=%.3f lateral_force=%.3f torque_norm=%.3f",
            result.extracted_distance,
            result.pull_force,
            result.lateral_force,
            result.torque_norm,
        )
        return

    rospy.logerr(
        "[usb_c_insertion] event=extract_cable_failed reason=%s extracted_distance=%.4f pull_force=%.3f lateral_force=%.3f torque_norm=%.3f",
        result.reason,
        result.extracted_distance,
        result.pull_force,
        result.lateral_force,
        result.torque_norm,
    )
    sys.exit(1)


if __name__ == "__main__":
    main()
