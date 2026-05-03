#!/usr/bin/env python3

from __future__ import annotations

import json
import os
import tempfile
from typing import Optional

from geometry_msgs.msg import PoseStamped
import rospy


def save_pose_stamped(pose: PoseStamped, path: str) -> bool:
    path = str(path or "").strip()
    if not path:
        return False
    payload = {
        "frame_id": pose.header.frame_id,
        "stamp": pose.header.stamp.to_sec() if pose.header.stamp else rospy.Time.now().to_sec(),
        "position": {
            "x": float(pose.pose.position.x),
            "y": float(pose.pose.position.y),
            "z": float(pose.pose.position.z),
        },
        "orientation": {
            "x": float(pose.pose.orientation.x),
            "y": float(pose.pose.orientation.y),
            "z": float(pose.pose.orientation.z),
            "w": float(pose.pose.orientation.w),
        },
    }
    temp_path = ""
    try:
        directory = os.path.dirname(path) or "."
        os.makedirs(directory, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=directory,
            prefix="%s." % os.path.basename(path),
            suffix=".tmp",
            delete=False,
        ) as handle:
            temp_path = handle.name
            json.dump(payload, handle, sort_keys=True)
            handle.write("\n")
        os.replace(temp_path, path)
        rospy.loginfo("[usb_c_insertion] event=pose_saved path=%s frame=%s", path, pose.header.frame_id)
        return True
    except OSError as exc:
        if temp_path:
            try:
                os.unlink(temp_path)
            except OSError:
                pass
        rospy.logwarn("[usb_c_insertion] event=pose_save_failed path=%s error=%s", path, exc)
        return False


def load_pose_stamped(path: str) -> Optional[PoseStamped]:
    path = str(path or "").strip()
    if not path:
        return None
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, ValueError) as exc:
        rospy.logwarn("[usb_c_insertion] event=pose_load_failed path=%s error=%s", path, exc)
        return None

    try:
        pose = PoseStamped()
        pose.header.frame_id = str(payload["frame_id"])
        pose.header.stamp = rospy.Time.from_sec(float(payload.get("stamp", 0.0)))
        position = payload["position"]
        orientation = payload["orientation"]
        pose.pose.position.x = float(position["x"])
        pose.pose.position.y = float(position["y"])
        pose.pose.position.z = float(position["z"])
        pose.pose.orientation.x = float(orientation["x"])
        pose.pose.orientation.y = float(orientation["y"])
        pose.pose.orientation.z = float(orientation["z"])
        pose.pose.orientation.w = float(orientation["w"])
        return pose
    except (KeyError, TypeError, ValueError) as exc:
        rospy.logwarn("[usb_c_insertion] event=pose_load_invalid path=%s error=%s", path, exc)
        return None
