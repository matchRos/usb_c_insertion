#!/usr/bin/env python3

from __future__ import annotations

import os
import sys

from geometry_msgs.msg import PoseStamped
import rospy

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from preinsert_workflow_helpers import PreinsertWorkflowHelpers


class PreinsertAlignmentWorkflow:
    """
    High-level workflow up to the waiting pose in front of the USB-C port.

    This file intentionally contains the readable process. The ROS actions,
    TF math, retries, and goal construction live in PreinsertWorkflowHelpers.
    """

    def __init__(self):
        self._helpers = PreinsertWorkflowHelpers()
        self._updated_port_pose_topic = self._helpers._required_str_param("~workflow/updated_port_pose_topic")
        self._updated_port_pose_publisher = rospy.Publisher(
            self._updated_port_pose_topic,
            PoseStamped,
            queue_size=1,
            latch=True,
        )

    def run(self) -> bool:
        rospy.loginfo("[usb_c_insertion] event=preinsert_workflow_start")

        if not self._helpers.wait_for_dependencies():
            return False

        overview_pose = self._helpers.load_overview_pose()
        if self._helpers.move_to_pose(overview_pose, "overview_pose") is None:
            return False

        coarse_port_pose = self._helpers.run_overview_vision()
        if coarse_port_pose is None:
            return False

        camera_pose = self._helpers.plan_camera_pose_from_port(coarse_port_pose)
        if camera_pose is None:
            return False
        if self._helpers.move_to_pose(camera_pose, "camera_to_coarse_port") is None:
            return False

        if not self._helpers.align_housing_yaw():
            return False

        if self._helpers.center_port_in_image() is None:
            return False

        orientation_check = self._helpers.estimate_housing_plane("orientation_check")
        if orientation_check is None:
            return False
        if not self._helpers.validate_plane_quality(orientation_check, "orientation_check"):
            return False

        looming_result = self._helpers.verify_looming()
        if looming_result is None:
            return False

        final_plane = self._helpers.estimate_housing_plane("final_depth_update")
        if final_plane is None:
            return False
        if not self._helpers.validate_plane_quality(final_plane, "final_depth_update"):
            return False

        updated_port_pose = self._helpers.build_updated_port_pose(final_plane)
        if updated_port_pose is None:
            return False
        self._updated_port_pose_publisher.publish(updated_port_pose)

        tcp_precontact_pose = self._helpers.plan_tcp_precontact_pose(updated_port_pose)
        if tcp_precontact_pose is None:
            return False
        if self._helpers.move_to_pose(tcp_precontact_pose, "tcp_precontact_wait") is None:
            return False

        rospy.loginfo(
            "[usb_c_insertion] event=preinsert_workflow_complete updated_port_topic=%s port_x=%.4f port_y=%.4f port_z=%.4f",
            self._updated_port_pose_topic,
            updated_port_pose.pose.position.x,
            updated_port_pose.pose.position.y,
            updated_port_pose.pose.position.z,
        )
        return True


def main() -> None:
    rospy.init_node("usb_c_insertion_preinsert_alignment_workflow")
    success = PreinsertAlignmentWorkflow().run()
    if not success:
        sys.exit(1)


if __name__ == "__main__":
    main()
