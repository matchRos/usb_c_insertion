#!/usr/bin/env python3

from __future__ import annotations

import json
import math
import os
import sys
import threading
import time
import tkinter as tk
from tkinter import ttk
from typing import Dict, Optional, Tuple

import actionlib
from geometry_msgs.msg import PoseStamped
import rospy

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from param_utils import get_param, required_float_param, required_str_param
from pose_persistence import load_pose_stamped, save_pose_stamped
from prepose_planner import normalize_quaternion, rotate_vector_by_quaternion
from robot_interface import RobotInterface
from tf_interface import TFInterface
from usb_c_insertion.msg import MoveToPoseAction, MoveToPoseGoal


class PortCalibrationGui:
    """
    Small interactive tool for collecting vision-estimated and manually aligned port poses.
    """

    def __init__(self):
        self._base_frame = required_str_param("~frames/base_frame")
        self._tool_frame = required_str_param("~frames/tool_frame")
        self._updated_port_pose_topic = str(
            get_param(
                "~calibration/updated_port_pose_topic",
                get_param("~workflow/updated_port_pose_topic", "/usb_c_insertion/updated_port_pose"),
            )
        ).strip()
        self._output_path = str(
            get_param("~calibration/output_path", "/tmp/usb_c_insertion_port_calibration_samples.jsonl")
        ).strip()
        self._latest_estimate_path = str(
            get_param(
                "~calibration/latest_estimate_path",
                get_param("~workflow/updated_port_pose_path", "/tmp/usb_c_insertion_latest_port_pose.json"),
            )
        ).strip()
        self._move_action_name = str(
            get_param("~calibration/move_action_name", get_param("~workflow/accurate_move_action_name", "move_to_pose_accurate"))
        ).strip()
        self._move_timeout = float(get_param("~calibration/move_timeout", get_param("~workflow/accurate_move_timeout", 90.0)))
        self._move_settle_time = float(
            get_param("~calibration/move_settle_time", get_param("~workflow/accurate_move_settle_time", 0.35))
        )
        self._jog_linear_speed = float(get_param("~calibration/jog_linear_speed", 0.0015))
        self._jog_angular_speed = float(get_param("~calibration/jog_angular_speed", 0.015))
        self._jog_linear_speed_min = float(get_param("~calibration/jog_linear_speed_min", 0.0002))
        self._jog_linear_speed_max = float(get_param("~calibration/jog_linear_speed_max", 0.0060))
        self._jog_angular_speed_min = float(get_param("~calibration/jog_angular_speed_min", 0.003))
        self._jog_angular_speed_max = float(get_param("~calibration/jog_angular_speed_max", 0.06))
        self._jog_speed_step_factor = max(1.01, float(get_param("~calibration/jog_speed_step_factor", 1.5)))
        self._jog_rate_hz = max(5.0, float(get_param("~calibration/jog_rate_hz", 30.0)))
        self._target_card_index = int(get_param("~workflow/target_card_index", 1))

        self._target_offset_tool_x = required_float_param("~precontact/target_offset_tool_x")
        self._target_offset_tool_y = required_float_param("~precontact/target_offset_tool_y")
        self._precontact_offset_tool_x = required_float_param("~precontact/precontact_offset_tool_x")
        self._precontact_offset_tool_y = required_float_param("~precontact/precontact_offset_tool_y")
        self._precontact_offset_tool_z = required_float_param("~precontact/precontact_offset_tool_z")
        self._precontact_offset_tool = (
            self._target_offset_tool_x + self._precontact_offset_tool_x,
            self._target_offset_tool_y + self._precontact_offset_tool_y,
            self._precontact_offset_tool_z,
        )

        self._tf = TFInterface()
        self._robot = RobotInterface()
        self._move_client = actionlib.SimpleActionClient(self._move_action_name, MoveToPoseAction)

        self._latest_estimated_port_pose: Optional[PoseStamped] = load_pose_stamped(self._latest_estimate_path)
        self._estimated_precontact_pose: Optional[PoseStamped] = None
        self._active_linear_tool = (0.0, 0.0, 0.0)
        self._active_angular_tool = (0.0, 0.0, 0.0)
        self._pressed_keys = set()
        self._keyboard_bindings = self._build_keyboard_bindings()
        self._last_sample_error_base = None
        self._sample_count = self._count_existing_samples()
        self._closed = False

        os.makedirs(os.path.dirname(self._output_path) or ".", exist_ok=True)
        self._subscriber = rospy.Subscriber(
            self._updated_port_pose_topic,
            PoseStamped,
            self._updated_port_pose_callback,
            queue_size=1,
        )

        self._root = tk.Tk()
        self._root.title("USB-C Port Calibration")
        self._root.geometry("1040x720")
        self._root.protocol("WM_DELETE_WINDOW", self._handle_close)
        self._build_ui()
        self._bind_keyboard_controls()
        if self._latest_estimated_port_pose is not None:
            self._status.configure(text="loaded saved port estimate")
        self._root.after(100, self._refresh_ui)
        self._root.after(int(1000.0 / self._jog_rate_hz), self._send_active_jog)

    def run(self) -> None:
        self._root.mainloop()

    def _set_status(self, text: str) -> None:
        if self._closed:
            return
        try:
            self._root.after(0, lambda: self._status.configure(text=text))
        except RuntimeError:
            pass

    def _build_ui(self) -> None:
        self._root.columnconfigure(0, weight=1)
        self._root.rowconfigure(1, weight=1)

        header = ttk.Frame(self._root, padding=(10, 8, 10, 4))
        header.grid(row=0, column=0, sticky="ew")
        header.columnconfigure(1, weight=1)
        ttk.Label(header, text="USB-C Port Calibration", font=("TkDefaultFont", 15, "bold")).grid(row=0, column=0, sticky="w")
        self._status = ttk.Label(header, text="waiting for port estimate")
        self._status.grid(row=0, column=1, sticky="e")

        main = ttk.Frame(self._root, padding=(10, 6, 10, 10))
        main.grid(row=1, column=0, sticky="nsew")
        main.columnconfigure(0, weight=1)
        main.columnconfigure(1, weight=1)
        main.rowconfigure(0, weight=1)

        left = ttk.LabelFrame(main, text="Pose Data", padding=8)
        left.grid(row=0, column=0, sticky="nsew", padx=(0, 6))
        left.columnconfigure(0, weight=1)
        left.rowconfigure(0, weight=1)
        self._pose_text = tk.Text(left, height=22, wrap="none")
        self._pose_text.configure(state="disabled")
        self._pose_text.grid(row=0, column=0, sticky="nsew")

        right = ttk.Frame(main)
        right.grid(row=0, column=1, sticky="nsew", padx=(6, 0))
        right.columnconfigure(0, weight=1)

        actions = ttk.LabelFrame(right, text="Actions", padding=8)
        actions.grid(row=0, column=0, sticky="ew")
        actions.columnconfigure(0, weight=1)
        actions.columnconfigure(1, weight=1)
        ttk.Button(actions, text="Use Latest Estimate", command=self._use_latest_estimate).grid(row=0, column=0, sticky="ew", padx=3, pady=3)
        ttk.Button(actions, text="Move To Estimated Precontact", command=self._start_move_to_precontact).grid(row=0, column=1, sticky="ew", padx=3, pady=3)
        ttk.Button(actions, text="Stop Motion", command=self._stop_motion).grid(row=1, column=0, sticky="ew", padx=3, pady=3)
        ttk.Button(actions, text="Save Sample", command=self._save_sample).grid(row=1, column=1, sticky="ew", padx=3, pady=3)

        jog = ttk.LabelFrame(right, text="Jog In Tool Frame", padding=8)
        jog.grid(row=1, column=0, sticky="ew", pady=(8, 0))
        for column in range(3):
            jog.columnconfigure(column, weight=1)
        self._jog_speed_label = ttk.Label(jog, text=self._jog_speed_text())
        self._jog_speed_label.grid(row=0, column=0, columnspan=3, sticky="w", pady=(0, 6))
        self._jog_button(jog, "-X", (-1.0, 0.0, 0.0), (0.0, 0.0, 0.0), 1, 0)
        self._jog_button(jog, "+X", (1.0, 0.0, 0.0), (0.0, 0.0, 0.0), 1, 1)
        self._jog_button(jog, "-Y", (0.0, -1.0, 0.0), (0.0, 0.0, 0.0), 2, 0)
        self._jog_button(jog, "+Y", (0.0, 1.0, 0.0), (0.0, 0.0, 0.0), 2, 1)
        self._jog_button(jog, "-Z", (0.0, 0.0, -1.0), (0.0, 0.0, 0.0), 3, 0)
        self._jog_button(jog, "+Z", (0.0, 0.0, 1.0), (0.0, 0.0, 0.0), 3, 1)
        self._jog_button(jog, "-Rx", (0.0, 0.0, 0.0), (-1.0, 0.0, 0.0), 4, 0)
        self._jog_button(jog, "+Rx", (0.0, 0.0, 0.0), (1.0, 0.0, 0.0), 4, 1)
        self._jog_button(jog, "-Ry", (0.0, 0.0, 0.0), (0.0, -1.0, 0.0), 5, 0)
        self._jog_button(jog, "+Ry", (0.0, 0.0, 0.0), (0.0, 1.0, 0.0), 5, 1)
        self._jog_button(jog, "-Rz", (0.0, 0.0, 0.0), (0.0, 0.0, -1.0), 6, 0)
        self._jog_button(jog, "+Rz", (0.0, 0.0, 0.0), (0.0, 0.0, 1.0), 6, 1)
        ttk.Label(
            jog,
            text=(
                "Keyboard: arrows=X/Y, PageUp/PageDown=Z, "
                "W/S=Rx, A/D=Ry, Q/E=Rz, +/-=speed, Space/Esc=stop"
            ),
        ).grid(row=7, column=0, columnspan=3, sticky="w", pady=(6, 0))

        metadata = ttk.LabelFrame(right, text="Sample Metadata", padding=8)
        metadata.grid(row=2, column=0, sticky="ew", pady=(8, 0))
        metadata.columnconfigure(1, weight=1)
        ttk.Label(metadata, text="Case/Port note").grid(row=0, column=0, sticky="w", padx=(0, 6))
        self._note_var = tk.StringVar()
        ttk.Entry(metadata, textvariable=self._note_var).grid(row=0, column=1, sticky="ew")
        self._output_label = ttk.Label(metadata, text=self._output_path)
        self._output_label.grid(row=1, column=0, columnspan=2, sticky="w", pady=(6, 0))

    def _jog_button(self, parent, label, linear, angular, row, column) -> None:
        button = ttk.Button(parent, text=label)
        button.grid(row=row, column=column, sticky="ew", padx=3, pady=3)
        button.bind("<ButtonPress-1>", lambda _event: self._start_jog(linear, angular))
        button.bind("<ButtonRelease-1>", lambda _event: self._stop_jog())
        button.bind("<Leave>", lambda _event: self._stop_jog())

    def _build_keyboard_bindings(self) -> Dict[str, Tuple[Tuple[float, float, float], Tuple[float, float, float]]]:
        return {
            "Left": ((-1.0, 0.0, 0.0), (0.0, 0.0, 0.0)),
            "Right": ((1.0, 0.0, 0.0), (0.0, 0.0, 0.0)),
            "Down": ((0.0, -1.0, 0.0), (0.0, 0.0, 0.0)),
            "Up": ((0.0, 1.0, 0.0), (0.0, 0.0, 0.0)),
            "Next": ((0.0, 0.0, -1.0), (0.0, 0.0, 0.0)),
            "Prior": ((0.0, 0.0, 1.0), (0.0, 0.0, 0.0)),
            "s": ((0.0, 0.0, 0.0), (-1.0, 0.0, 0.0)),
            "w": ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0)),
            "a": ((0.0, 0.0, 0.0), (0.0, -1.0, 0.0)),
            "d": ((0.0, 0.0, 0.0), (0.0, 1.0, 0.0)),
            "q": ((0.0, 0.0, 0.0), (0.0, 0.0, -1.0)),
            "e": ((0.0, 0.0, 0.0), (0.0, 0.0, 1.0)),
        }

    def _bind_keyboard_controls(self) -> None:
        self._root.bind("<KeyPress>", self._handle_key_press)
        self._root.bind("<KeyRelease>", self._handle_key_release)
        self._root.focus_set()

    def _handle_key_press(self, event) -> None:
        key = self._normalize_key(event.keysym)
        if key in ("space", "Escape"):
            self._pressed_keys.clear()
            self._stop_motion()
            return
        if key in ("plus", "minus"):
            self._adjust_jog_speed(1 if key == "plus" else -1)
            return
        if key not in self._keyboard_bindings:
            return
        self._pressed_keys.add(key)
        self._update_keyboard_jog()

    def _handle_key_release(self, event) -> None:
        key = self._normalize_key(event.keysym)
        if key not in self._keyboard_bindings:
            return
        self._pressed_keys.discard(key)
        self._update_keyboard_jog()

    def _normalize_key(self, key: str) -> str:
        key = str(key)
        if len(key) == 1:
            if key in ("+", "="):
                return "plus"
            if key in ("-", "_"):
                return "minus"
            return key.lower()
        if key in ("KP_Add", "plus"):
            return "plus"
        if key in ("KP_Subtract", "minus"):
            return "minus"
        return key

    def _update_keyboard_jog(self) -> None:
        linear = [0.0, 0.0, 0.0]
        angular = [0.0, 0.0, 0.0]
        for key in self._pressed_keys:
            linear_direction, angular_direction = self._keyboard_bindings[key]
            for index in range(3):
                linear[index] += linear_direction[index]
                angular[index] += angular_direction[index]
        linear = self._normalize_direction(linear)
        angular = self._normalize_direction(angular)
        self._active_linear_tool = tuple(component * self._jog_linear_speed for component in linear)
        self._active_angular_tool = tuple(component * self._jog_angular_speed for component in angular)
        if not self._pressed_keys:
            self._robot.send_zero_twist()

    def _adjust_jog_speed(self, direction: int) -> None:
        if direction > 0:
            factor = self._jog_speed_step_factor
        else:
            factor = 1.0 / self._jog_speed_step_factor
        self._jog_linear_speed = self._clamp(
            self._jog_linear_speed * factor,
            self._jog_linear_speed_min,
            self._jog_linear_speed_max,
        )
        self._jog_angular_speed = self._clamp(
            self._jog_angular_speed * factor,
            self._jog_angular_speed_min,
            self._jog_angular_speed_max,
        )
        self._update_keyboard_jog()
        self._jog_speed_label.configure(text=self._jog_speed_text())
        self._set_status(
            "jog speed: %.2f mm/s, %.2f deg/s" % (
                self._jog_linear_speed * 1000.0,
                math.degrees(self._jog_angular_speed),
            )
        )

    def _jog_speed_text(self) -> str:
        return (
            "Hold buttons/keys to move. Linear %.2f mm/s, angular %.2f deg/s"
            % (self._jog_linear_speed * 1000.0, math.degrees(self._jog_angular_speed))
        )

    def _updated_port_pose_callback(self, msg: PoseStamped) -> None:
        self._latest_estimated_port_pose = msg
        self._estimated_precontact_pose = None
        save_pose_stamped(msg, self._latest_estimate_path)
        self._set_status("received port estimate")

    def _use_latest_estimate(self) -> None:
        if self._latest_estimated_port_pose is None:
            self._latest_estimated_port_pose = load_pose_stamped(self._latest_estimate_path)
        if self._latest_estimated_port_pose is None:
            self._set_status("no estimated port pose available")
            return
        self._estimated_precontact_pose = self._plan_precontact_pose(self._latest_estimated_port_pose)
        if self._estimated_precontact_pose is None:
            self._set_status("failed to plan precontact pose")
            return
        self._set_status("loaded latest estimated port pose")

    def _start_move_to_precontact(self) -> None:
        if self._estimated_precontact_pose is None:
            self._use_latest_estimate()
        if self._estimated_precontact_pose is None:
            return
        thread = threading.Thread(target=self._move_to_precontact, daemon=True)
        thread.start()

    def _move_to_precontact(self) -> None:
        self._set_status("waiting for %s" % self._move_action_name)
        if not self._move_client.wait_for_server(rospy.Duration.from_sec(5.0)):
            self._set_status("move action unavailable")
            return
        goal = MoveToPoseGoal()
        goal.target_pose = self._estimated_precontact_pose
        goal.timeout = self._move_timeout
        goal.settle_time = self._move_settle_time
        self._move_client.send_goal(goal)
        self._set_status("moving to estimated precontact")
        finished = self._move_client.wait_for_result(rospy.Duration.from_sec(max(1.0, self._move_timeout + 2.0)))
        if not finished:
            self._move_client.cancel_goal()
            self._robot.stop_motion()
            self._set_status("move timed out")
            return
        result = self._move_client.get_result()
        if result is not None and result.success:
            self._set_status("at estimated precontact, jog manually")
        else:
            message = result.message if result is not None else "no result"
            self._set_status("move failed: %s" % message)

    def _start_jog(self, linear_direction, angular_direction) -> None:
        self._active_linear_tool = tuple(component * self._jog_linear_speed for component in linear_direction)
        self._active_angular_tool = tuple(component * self._jog_angular_speed for component in angular_direction)

    def _stop_jog(self) -> None:
        self._pressed_keys.clear()
        self._active_linear_tool = (0.0, 0.0, 0.0)
        self._active_angular_tool = (0.0, 0.0, 0.0)
        self._robot.send_zero_twist()

    def _send_active_jog(self) -> None:
        if self._closed or rospy.is_shutdown():
            return
        if any(abs(value) > 1e-12 for value in self._active_linear_tool + self._active_angular_tool):
            current_pose = self._tf.get_tool_pose_in_base()
            if current_pose is not None:
                q = self._pose_quaternion(current_pose)
                linear_base = rotate_vector_by_quaternion(*self._active_linear_tool, *q)
                angular_base = rotate_vector_by_quaternion(*self._active_angular_tool, *q)
                self._robot.send_twist(
                    linear_base[0],
                    linear_base[1],
                    linear_base[2],
                    angular_base[0],
                    angular_base[1],
                    angular_base[2],
                )
        self._root.after(int(1000.0 / self._jog_rate_hz), self._send_active_jog)

    def _stop_motion(self) -> None:
        self._stop_jog()
        self._robot.stop_motion()
        self._set_status("motion stopped")

    def _save_sample(self) -> None:
        if self._latest_estimated_port_pose is None:
            self._latest_estimated_port_pose = load_pose_stamped(self._latest_estimate_path)
        if self._latest_estimated_port_pose is None:
            self._set_status("cannot save: no estimated port pose")
            return
        current_tcp = self._tf.get_tool_pose_in_base()
        if current_tcp is None:
            self._set_status("cannot save: no current TCP pose")
            return
        self._estimated_precontact_pose = self._plan_precontact_pose(self._latest_estimated_port_pose)
        actual_port_pose = self._actual_port_pose_from_tcp(current_tcp)
        error_base = self._pose_delta(actual_port_pose, self._latest_estimated_port_pose)
        error_tool = self._rotate_base_vector_to_tool(error_base, current_tcp)
        self._last_sample_error_base = error_base
        self._sample_count += 1
        sample = {
            "sample_id": self._sample_count,
            "stamp": rospy.Time.now().to_sec(),
            "wall_time": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "target_card_index": self._target_card_index,
            "note": self._note_var.get(),
            "base_frame": self._base_frame,
            "tool_frame": self._tool_frame,
            "primary_error_frame": "tool",
            "updated_port_pose_topic": self._updated_port_pose_topic,
            "precontact_offset_tool": {
                "x": self._precontact_offset_tool[0],
                "y": self._precontact_offset_tool[1],
                "z": self._precontact_offset_tool[2],
            },
            "estimated_port_pose": self._pose_to_dict(self._latest_estimated_port_pose),
            "estimated_precontact_pose": self._pose_to_dict(self._estimated_precontact_pose),
            "actual_tcp_pose": self._pose_to_dict(current_tcp),
            "actual_port_pose": self._pose_to_dict(actual_port_pose),
            "error": self._vector_to_dict(error_tool),
            "error_base": self._vector_to_dict(error_base),
            "error_tool": self._vector_to_dict(error_tool),
            "error_norm_m": math.sqrt(sum(component * component for component in error_base)),
        }
        with open(self._output_path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(sample, sort_keys=True) + "\n")
        self._set_status(
            "saved sample %d, tool_error=(%.2f, %.2f, %.2f) mm" % (
                self._sample_count,
                error_tool[0] * 1000.0,
                error_tool[1] * 1000.0,
                error_tool[2] * 1000.0,
            )
        )

    def _plan_precontact_pose(self, port_pose: PoseStamped) -> Optional[PoseStamped]:
        current_pose = self._tf.get_tool_pose_in_base()
        if current_pose is None:
            return None
        q = self._pose_quaternion(current_pose)
        offset_base = rotate_vector_by_quaternion(*self._precontact_offset_tool, *q)
        target = PoseStamped()
        target.header.stamp = rospy.Time.now()
        target.header.frame_id = self._base_frame
        target.pose.position.x = port_pose.pose.position.x + offset_base[0]
        target.pose.position.y = port_pose.pose.position.y + offset_base[1]
        target.pose.position.z = port_pose.pose.position.z + offset_base[2]
        target.pose.orientation = current_pose.pose.orientation
        return target

    def _actual_port_pose_from_tcp(self, tcp_pose: PoseStamped) -> PoseStamped:
        q = self._pose_quaternion(tcp_pose)
        offset_base = rotate_vector_by_quaternion(*self._precontact_offset_tool, *q)
        actual = PoseStamped()
        actual.header.stamp = rospy.Time.now()
        actual.header.frame_id = self._base_frame
        actual.pose.position.x = tcp_pose.pose.position.x - offset_base[0]
        actual.pose.position.y = tcp_pose.pose.position.y - offset_base[1]
        actual.pose.position.z = tcp_pose.pose.position.z - offset_base[2]
        actual.pose.orientation = tcp_pose.pose.orientation
        return actual

    def _refresh_ui(self) -> None:
        if self._closed:
            return
        current_tcp = self._tf.get_tool_pose_in_base()
        actual_port = self._actual_port_pose_from_tcp(current_tcp) if current_tcp is not None else None
        error_base = None
        error_tool = None
        if actual_port is not None and self._latest_estimated_port_pose is not None:
            error_base = self._pose_delta(actual_port, self._latest_estimated_port_pose)
            error_tool = self._rotate_base_vector_to_tool(error_base, current_tcp)
        lines = [
            "updated_port_pose_topic: %s" % self._updated_port_pose_topic,
            "latest_estimate_path: %s" % self._latest_estimate_path,
            "output_path: %s" % self._output_path,
            "sample_count: %d" % self._sample_count,
            "primary_error_frame: tool (%s)" % self._tool_frame,
            "",
            "estimated_port_pose:",
            self._format_pose(self._latest_estimated_port_pose),
            "",
            "estimated_precontact_pose:",
            self._format_pose(self._estimated_precontact_pose),
            "",
            "current_tcp_pose:",
            self._format_pose(current_tcp),
            "",
            "actual_port_pose_from_current_tcp:",
            self._format_pose(actual_port),
            "",
            "current_tool_error actual_port - estimated_port:",
            self._format_vector(error_tool),
            "",
            "current_base_error actual_port - estimated_port:",
            self._format_vector(error_base),
        ]
        self._pose_text.configure(state="normal")
        self._pose_text.delete("1.0", tk.END)
        self._pose_text.insert("1.0", "\n".join(lines))
        self._pose_text.configure(state="disabled")
        self._root.after(250, self._refresh_ui)

    def _handle_close(self) -> None:
        self._closed = True
        self._robot.stop_motion()
        self._root.destroy()

    @staticmethod
    def _normalize_direction(vector) -> Tuple[float, float, float]:
        norm = math.sqrt(sum(float(component) * float(component) for component in vector))
        if norm <= 1e-9:
            return (0.0, 0.0, 0.0)
        return tuple(float(component) / norm for component in vector)

    @staticmethod
    def _clamp(value: float, min_value: float, max_value: float) -> float:
        lower = min(float(min_value), float(max_value))
        upper = max(float(min_value), float(max_value))
        return max(lower, min(upper, float(value)))

    def _count_existing_samples(self) -> int:
        try:
            with open(self._output_path, "r", encoding="utf-8") as handle:
                return sum(1 for line in handle if line.strip())
        except OSError:
            return 0

    @staticmethod
    def _pose_quaternion(pose: PoseStamped) -> Tuple[float, float, float, float]:
        return normalize_quaternion(
            (
                pose.pose.orientation.x,
                pose.pose.orientation.y,
                pose.pose.orientation.z,
                pose.pose.orientation.w,
            )
        )

    @staticmethod
    def _pose_delta(first: PoseStamped, second: PoseStamped) -> Tuple[float, float, float]:
        return (
            first.pose.position.x - second.pose.position.x,
            first.pose.position.y - second.pose.position.y,
            first.pose.position.z - second.pose.position.z,
        )

    def _rotate_base_vector_to_tool(self, vector_base: Tuple[float, float, float], tool_pose: PoseStamped) -> Tuple[float, float, float]:
        qx, qy, qz, qw = self._pose_quaternion(tool_pose)
        return rotate_vector_by_quaternion(vector_base[0], vector_base[1], vector_base[2], -qx, -qy, -qz, qw)

    @staticmethod
    def _pose_to_dict(pose: Optional[PoseStamped]) -> Optional[Dict]:
        if pose is None:
            return None
        return {
            "frame_id": pose.header.frame_id,
            "stamp": pose.header.stamp.to_sec() if pose.header.stamp else 0.0,
            "position": {
                "x": pose.pose.position.x,
                "y": pose.pose.position.y,
                "z": pose.pose.position.z,
            },
            "orientation": {
                "x": pose.pose.orientation.x,
                "y": pose.pose.orientation.y,
                "z": pose.pose.orientation.z,
                "w": pose.pose.orientation.w,
            },
        }

    @staticmethod
    def _vector_to_dict(vector: Tuple[float, float, float]) -> Dict:
        return {"x": vector[0], "y": vector[1], "z": vector[2]}

    @staticmethod
    def _format_pose(pose: Optional[PoseStamped]) -> str:
        if pose is None:
            return "  unavailable"
        return (
            "  frame=%s\n"
            "  p=(%.5f, %.5f, %.5f)\n"
            "  q=(%.5f, %.5f, %.5f, %.5f)"
            % (
                pose.header.frame_id,
                pose.pose.position.x,
                pose.pose.position.y,
                pose.pose.position.z,
                pose.pose.orientation.x,
                pose.pose.orientation.y,
                pose.pose.orientation.z,
                pose.pose.orientation.w,
            )
        )

    @staticmethod
    def _format_vector(vector: Optional[Tuple[float, float, float]]) -> str:
        if vector is None:
            return "  unavailable"
        return "  m=(%.5f, %.5f, %.5f)\n  mm=(%.2f, %.2f, %.2f)" % (
            vector[0],
            vector[1],
            vector[2],
            vector[0] * 1000.0,
            vector[1] * 1000.0,
            vector[2] * 1000.0,
        )


def main() -> None:
    rospy.init_node("usb_c_insertion_port_calibration_gui")
    PortCalibrationGui().run()


if __name__ == "__main__":
    main()
