#!/usr/bin/env python3

from __future__ import annotations

import os
import sys

import rospy

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from preinsert_alignment_workflow import PreinsertAlignmentWorkflow
from robot_interface import RobotInterface


class GraspInsertedCableWorkflow:
    """
    Reuse the pre-insertion alignment pipeline, then close the gripper.

    This is meant for the experiment where a USB-C cable is already seated in
    the port and the robot should approach the same precontact pose before
    trying to grasp the cable.
    """

    def __init__(self):
        self._preinsert_workflow = PreinsertAlignmentWorkflow()
        self._robot = RobotInterface()

    def run(self) -> bool:
        rospy.loginfo("[usb_c_insertion] event=grasp_inserted_cable_workflow_start")

        rospy.loginfo("[usb_c_insertion] event=grasp_inserted_cable_open_gripper_start")
        if not self._robot.open_gripper():
            rospy.logerr(
                "[usb_c_insertion] event=grasp_inserted_cable_workflow_failed reason=gripper_open_failed"
            )
            return False

        if not self._preinsert_workflow.run():
            rospy.logerr(
                "[usb_c_insertion] event=grasp_inserted_cable_workflow_failed reason=preinsert_alignment_failed"
            )
            return False

        rospy.loginfo("[usb_c_insertion] event=grasp_inserted_cable_close_gripper_start")
        if not self._robot.close_gripper():
            rospy.logerr(
                "[usb_c_insertion] event=grasp_inserted_cable_workflow_failed reason=gripper_close_failed"
            )
            return False

        rospy.loginfo("[usb_c_insertion] event=grasp_inserted_cable_workflow_complete success=true")
        return True


def main() -> None:
    rospy.init_node("usb_c_insertion_grasp_inserted_cable_workflow")
    success = GraspInsertedCableWorkflow().run()
    if not success:
        sys.exit(1)


if __name__ == "__main__":
    main()
