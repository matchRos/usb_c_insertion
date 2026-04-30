#!/usr/bin/env python3

from __future__ import annotations

import os
import subprocess
import sys
import time

from geometry_msgs.msg import PoseStamped
import rospy

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from param_utils import required_float_param, required_str_param
from vision_pose_loader import load_vision_pose_from_json
from usb_c_insertion.srv import RunVision, RunVisionResponse


class RunVisionServiceNode:
    """
    Thin ROS service wrapper around the existing run_vision.sh script.

    The wrapper keeps the current shell-based integration intact while giving
    an external workflow a synchronous ROS interface with structured success,
    error handling, and the parsed pose estimate.
    """

    def __init__(self):
        self._service_name = "run_vision"
        self._script_path = required_str_param("~run_vision_script_path")
        self._json_path = required_str_param("~vision_pose_json_path")
        self._command_timeout = required_float_param("~workflow/run_vision_timeout")
        self._result_wait_timeout = required_float_param("~workflow/vision_result_wait_timeout")
        self._base_frame = required_str_param("~frames/base_frame")

        self._service = rospy.Service(self._service_name, RunVision, self._handle_request)
        rospy.loginfo(
            "[usb_c_insertion] event=run_vision_service_ready service=%s",
            self._service_name,
        )

    def _handle_request(self, request) -> RunVisionResponse:
        response = RunVisionResponse()

        if not self._script_path:
            response.success = False
            response.message = "run_vision_script_path_not_configured"
            return response

        if not os.path.isfile(self._script_path):
            response.success = False
            response.message = "run_vision_script_missing"
            return response

        if not self._json_path:
            response.success = False
            response.message = "vision_pose_json_path_not_configured"
            return response

        previous_mtime = os.path.getmtime(self._json_path) if os.path.isfile(self._json_path) else 0.0
        rospy.loginfo(
            "[usb_c_insertion] event=run_vision_start script=%s require_fresh_result=%s",
            self._script_path,
            str(bool(request.require_fresh_result)).lower(),
        )

        try:
            completed = subprocess.run(
                [self._script_path],
                check=False,
                timeout=max(1.0, self._command_timeout),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
        except subprocess.TimeoutExpired:
            response.success = False
            response.message = "run_vision_timeout"
            return response
        except OSError as exc:
            response.success = False
            response.message = "run_vision_launch_failed: %s" % exc
            return response

        if completed.returncode != 0:
            response.success = False
            response.message = "run_vision_failed: %s" % completed.stderr.strip()
            return response

        if request.require_fresh_result and not self._wait_for_fresh_result(previous_mtime):
            response.success = False
            response.message = "vision_result_not_updated"
            return response

        try:
            vision_pose = load_vision_pose_from_json(self._json_path)
        except (OSError, ValueError, KeyError, TypeError) as exc:
            response.success = False
            response.message = "vision_pose_load_failed: %s" % exc
            return response

        pose = PoseStamped()
        pose.header.stamp = rospy.Time.now()
        pose.header.frame_id = self._base_frame
        pose.pose.position.x = vision_pose.x
        pose.pose.position.y = vision_pose.y
        pose.pose.position.z = vision_pose.z
        pose.pose.orientation.x = vision_pose.qx
        pose.pose.orientation.y = vision_pose.qy
        pose.pose.orientation.z = vision_pose.qz
        pose.pose.orientation.w = vision_pose.qw

        response.success = True
        response.message = "vision_result_ready"
        response.json_path = self._json_path
        response.port_pose = pose
        rospy.loginfo(
            "[usb_c_insertion] event=run_vision_complete path=%s x=%.4f y=%.4f z=%.4f",
            self._json_path,
            vision_pose.x,
            vision_pose.y,
            vision_pose.z,
        )
        return response

    def _wait_for_fresh_result(self, previous_mtime: float) -> bool:
        deadline = time.time() + max(0.0, self._result_wait_timeout)
        while not rospy.is_shutdown() and time.time() <= deadline:
            if os.path.isfile(self._json_path) and os.path.getmtime(self._json_path) > previous_mtime:
                return True
            rospy.sleep(0.1)
        return False


def main() -> None:
    rospy.init_node("usb_c_insertion_run_vision_service")
    RunVisionServiceNode()
    rospy.spin()


if __name__ == "__main__":
    main()
