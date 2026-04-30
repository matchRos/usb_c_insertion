#!/usr/bin/env python3

from __future__ import annotations

import os
import sys

from geometry_msgs.msg import PoseStamped, Twist
import rospy
from std_msgs.msg import Bool, String

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from param_utils import required_bool_param, required_int_param, required_str_param


class RobotInterface:
    """
    Small client wrapper for the motion pipeline nodes.

    Motion-producing modules should talk only to this interface so every robot
    command flows through the same smoothing and watchdog path.
    """

    def __init__(self, queue_size: int = 10):
        self._raw_twist_topic = required_str_param("~topics/raw_twist_cmd")
        self._pose_target_topic = required_str_param("~topics/pose_target")
        self._pose_servo_enable_topic = required_str_param("~topics/pose_servo_enable")
        self._script_command_topic = required_str_param("~topics/script_command")
        self._open_via_script_command = required_bool_param("~gripper/open_via_script_command")
        self._open_script_command = required_str_param("~gripper/open_script_command")
        self._io_service_name = required_str_param("~gripper/io_service_name")
        self._fallback_digital_output_pin = required_int_param("~gripper/fallback_digital_output_pin")
        self._fallback_digital_output_state = required_bool_param("~gripper/fallback_digital_output_state")
        self._stop_repeat_count = required_int_param("~motion/stop_repeat_count")

        self._raw_twist_publisher = rospy.Publisher(self._raw_twist_topic, Twist, queue_size=queue_size)
        self._pose_target_publisher = rospy.Publisher(
            self._pose_target_topic,
            PoseStamped,
            queue_size=1,
            latch=True,
        )
        self._pose_servo_enable_publisher = rospy.Publisher(
            self._pose_servo_enable_topic,
            Bool,
            queue_size=1,
            latch=True,
        )
        self._script_command_publisher = rospy.Publisher(
            self._script_command_topic,
            String,
            queue_size=1,
        )

    def send_twist(self, vx: float, vy: float, vz: float, wx: float, wy: float, wz: float) -> Twist:
        """
        Send a raw Cartesian Twist request into the smoothing node.
        """
        twist = Twist()
        twist.linear.x = float(vx)
        twist.linear.y = float(vy)
        twist.linear.z = float(vz)
        twist.angular.x = float(wx)
        twist.angular.y = float(wy)
        twist.angular.z = float(wz)
        self._raw_twist_publisher.publish(twist)
        return twist

    def send_zero_twist(self) -> Twist:
        """
        Send an explicit zero Twist request into the smoothing node.
        """
        return self.send_twist(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

    def stop_motion(self) -> None:
        """
        Disable pose servoing and publish repeated zero Twist commands.
        """
        self.enable_pose_servo(False)
        for _ in range(max(1, self._stop_repeat_count)):
            self.send_zero_twist()

    def enable_pose_servo(self, enabled: bool) -> None:
        """
        Enable or disable the dedicated pose servo node.
        """
        self._pose_servo_enable_publisher.publish(Bool(data=bool(enabled)))

    def send_pose_target(
        self,
        x: float,
        y: float,
        z: float,
        qx: float = 0.0,
        qy: float = 0.0,
        qz: float = 0.0,
        qw: float = 1.0,
        frame_id: str = "",
    ) -> PoseStamped:
        """
        Publish a target pose for the pose servo node.
        """
        pose = PoseStamped()
        pose.header.stamp = rospy.Time.now()
        pose.header.frame_id = frame_id or required_str_param("~frames/base_frame")
        pose.pose.position.x = float(x)
        pose.pose.position.y = float(y)
        pose.pose.position.z = float(z)
        pose.pose.orientation.x = float(qx)
        pose.pose.orientation.y = float(qy)
        pose.pose.orientation.z = float(qz)
        pose.pose.orientation.w = float(qw)
        self._pose_target_publisher.publish(pose)
        return pose

    def open_gripper(self) -> bool:
        """
        Open the gripper using a configured script command or digital output fallback.
        """
        if self._open_via_script_command and self._open_script_command:
            self._script_command_publisher.publish(String(data=self._open_script_command))
            rospy.loginfo("[usb_c_insertion] event=gripper_open_command mode=script_command")
            return True

        if self.set_digital_output(self._fallback_digital_output_pin, self._fallback_digital_output_state):
            rospy.loginfo(
                "[usb_c_insertion] event=gripper_open_command mode=digital_output pin=%d state=%s",
                self._fallback_digital_output_pin,
                str(self._fallback_digital_output_state).lower(),
            )
            return True

        rospy.logerr("[usb_c_insertion] event=gripper_open_command_failed")
        return False

    def wait_for_motion_pipeline(self, timeout: float = 2.0, require_pose_servo: bool = False) -> bool:
        """
        Wait until the motion pipeline subscribers are connected.
        """
        deadline = rospy.Time.now() + rospy.Duration.from_sec(max(0.0, float(timeout)))
        rate = rospy.Rate(20.0)
        while not rospy.is_shutdown():
            raw_twist_ready = self._raw_twist_publisher.get_num_connections() > 0
            pose_target_ready = self._pose_target_publisher.get_num_connections() > 0
            pose_enable_ready = self._pose_servo_enable_publisher.get_num_connections() > 0
            pose_ready = pose_target_ready and pose_enable_ready
            if raw_twist_ready and (pose_ready or not require_pose_servo):
                return True
            if rospy.Time.now() > deadline:
                return False
            rate.sleep()
        return False

    def set_digital_output(self, pin: int, state: bool) -> bool:
        """
        Set a UR digital output through the driver service.
        """
        try:
            from ur_msgs.srv import SetIO, SetIORequest
        except ImportError as exc:
            rospy.logerr("[usb_c_insertion] event=set_digital_output_import_failed error=%s", exc)
            return False

        try:
            rospy.wait_for_service(self._io_service_name, timeout=2.0)
            client = rospy.ServiceProxy(self._io_service_name, SetIO)
            request = SetIORequest()
            request.fun = SetIORequest.FUN_SET_DIGITAL_OUT
            request.pin = int(pin)
            request.state = 1.0 if state else 0.0
            response = client(request)
            return bool(response.success)
        except (rospy.ROSException, rospy.ServiceException) as exc:
            rospy.logerr("[usb_c_insertion] event=set_digital_output_failed pin=%d error=%s", int(pin), exc)
            return False
