#!/usr/bin/env python3

from __future__ import annotations

from geometry_msgs.msg import PoseStamped, Twist
import rospy
from std_msgs.msg import Bool


class RobotInterface:
    """
    Small client wrapper for the motion pipeline nodes.

    Motion-producing modules should talk only to this interface so every robot
    command flows through the same smoothing and watchdog path.
    """

    def __init__(self, queue_size: int = 10):
        self._raw_twist_topic = rospy.get_param("~topics/raw_twist_cmd", "/usb_c_insertion/raw_twist_cmd")
        self._pose_target_topic = rospy.get_param("~topics/pose_target", "/usb_c_insertion/pose_target")
        self._pose_servo_enable_topic = rospy.get_param("~topics/pose_servo_enable", "/usb_c_insertion/pose_servo_enable")
        self._stop_repeat_count = int(rospy.get_param("~motion/stop_repeat_count", 3))

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
        pose.header.frame_id = frame_id or rospy.get_param("~frames/base_frame", "base_link")
        pose.pose.position.x = float(x)
        pose.pose.position.y = float(y)
        pose.pose.position.z = float(z)
        pose.pose.orientation.x = float(qx)
        pose.pose.orientation.y = float(qy)
        pose.pose.orientation.z = float(qz)
        pose.pose.orientation.w = float(qw)
        self._pose_target_publisher.publish(pose)
        return pose
