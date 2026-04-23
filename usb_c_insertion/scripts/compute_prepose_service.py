#!/usr/bin/env python3

from __future__ import annotations

import os
import sys

from geometry_msgs.msg import PoseStamped
import rospy

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from prepose_planner import compute_port_frame_target, compute_tcp_target_orientation
from usb_c_insertion.srv import ComputePrePose, ComputePrePoseResponse


class ComputePrePoseServiceNode:
    def __init__(self):
        self._service_name = str(rospy.get_param("~service_name", "compute_prepose")).strip()
        self._base_frame = str(rospy.get_param("~frames/base_frame", "base_link"))
        self._offset_x = float(rospy.get_param("~state_machine/prepose_offset_port_x", -0.03))
        self._offset_y = float(rospy.get_param("~state_machine/prepose_offset_port_y", 0.0))
        self._offset_z = float(rospy.get_param("~state_machine/prepose_offset_port_z", 0.0))
        self._service = rospy.Service(self._service_name, ComputePrePose, self._handle_request)
        rospy.loginfo("[usb_c_insertion] event=compute_prepose_service_ready service=%s", self._service_name)

    def _handle_request(self, request) -> ComputePrePoseResponse:
        response = ComputePrePoseResponse()
        port_pose = request.port_pose
        frame_id = port_pose.header.frame_id.strip() or self._base_frame
        if frame_id != self._base_frame:
            response.success = False
            response.message = "unsupported_port_pose_frame"
            return response

        try:
            target_xyz = compute_port_frame_target(
                (
                    port_pose.pose.position.x,
                    port_pose.pose.position.y,
                    port_pose.pose.position.z,
                    port_pose.pose.orientation.x,
                    port_pose.pose.orientation.y,
                    port_pose.pose.orientation.z,
                    port_pose.pose.orientation.w,
                ),
                (self._offset_x, self._offset_y, self._offset_z),
            )
            target_quaternion = compute_tcp_target_orientation(
                (
                    port_pose.pose.orientation.x,
                    port_pose.pose.orientation.y,
                    port_pose.pose.orientation.z,
                    port_pose.pose.orientation.w,
                )
            )
        except ValueError as exc:
            response.success = False
            response.message = "compute_prepose_failed: %s" % exc
            return response

        pre_pose = PoseStamped()
        pre_pose.header.stamp = rospy.Time.now()
        pre_pose.header.frame_id = self._base_frame
        pre_pose.pose.position.x = target_xyz[0]
        pre_pose.pose.position.y = target_xyz[1]
        pre_pose.pose.position.z = target_xyz[2]
        pre_pose.pose.orientation.x = target_quaternion[0]
        pre_pose.pose.orientation.y = target_quaternion[1]
        pre_pose.pose.orientation.z = target_quaternion[2]
        pre_pose.pose.orientation.w = target_quaternion[3]

        response.success = True
        response.message = "prepose_ready"
        response.pre_pose = pre_pose
        rospy.loginfo(
            "[usb_c_insertion] event=prepose_computed x=%.4f y=%.4f z=%.4f",
            pre_pose.pose.position.x,
            pre_pose.pose.position.y,
            pre_pose.pose.position.z,
        )
        return response


def main() -> None:
    rospy.init_node("usb_c_insertion_compute_prepose_service")
    ComputePrePoseServiceNode()
    rospy.spin()


if __name__ == "__main__":
    main()
