#!/usr/bin/env python3

from __future__ import annotations

import json
import os
import queue
import sys
import time
import tkinter as tk
from tkinter import ttk

import rospy
from std_msgs.msg import String

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from param_utils import get_param


class WorkflowGui:
    """
    Small read-only monitor for the combined insertion workflow.
    """

    def __init__(self):
        self._status_topic = get_param(
            "~status_topic",
            get_param("~combined_workflow/status_topic", "/usb_c_insertion/combined_workflow/status"),
        )
        self._queue = queue.Queue()
        self._rows = {}
        self._payloads = {}
        self._latest_payload = None

        self._root = tk.Tk()
        self._root.title("USB-C Insertion Workflow")
        self._root.geometry("1180x680")
        self._root.protocol("WM_DELETE_WINDOW", self._handle_close)

        self._build_ui()
        self._subscriber = rospy.Subscriber(self._status_topic, String, self._status_callback, queue_size=100)
        self._root.after(100, self._poll_queue)

    def _build_ui(self) -> None:
        self._root.columnconfigure(0, weight=1)
        self._root.rowconfigure(1, weight=1)
        self._root.rowconfigure(3, weight=1)

        header = ttk.Frame(self._root, padding=(10, 8, 10, 4))
        header.grid(row=0, column=0, sticky="ew")
        header.columnconfigure(1, weight=1)

        ttk.Label(header, text="USB-C Insertion Workflow", font=("TkDefaultFont", 14, "bold")).grid(
            row=0,
            column=0,
            sticky="w",
        )
        self._topic_label = ttk.Label(header, text=self._status_topic)
        self._topic_label.grid(row=0, column=1, sticky="e")

        columns = ("status", "success", "message", "values", "updated")
        self._tree = ttk.Treeview(self._root, columns=columns, show="tree headings", selectmode="browse")
        self._tree.heading("#0", text="Step")
        self._tree.heading("status", text="Status")
        self._tree.heading("success", text="Success")
        self._tree.heading("message", text="Message")
        self._tree.heading("values", text="Return Values")
        self._tree.heading("updated", text="Updated")

        self._tree.column("#0", width=210, minwidth=160, stretch=False)
        self._tree.column("status", width=100, minwidth=90, stretch=False)
        self._tree.column("success", width=80, minwidth=70, stretch=False)
        self._tree.column("message", width=220, minwidth=120, stretch=True)
        self._tree.column("values", width=420, minwidth=220, stretch=True)
        self._tree.column("updated", width=110, minwidth=90, stretch=False)

        self._tree.tag_configure("pending", foreground="#666666")
        self._tree.tag_configure("running", foreground="#005a9c")
        self._tree.tag_configure("success", foreground="#176b2c")
        self._tree.tag_configure("failed", foreground="#a40000")
        self._tree.tag_configure("skipped", foreground="#666666")
        self._tree.bind("<<TreeviewSelect>>", self._handle_selection)
        self._tree.grid(row=1, column=0, sticky="nsew", padx=10, pady=(0, 6))

        status_frame = ttk.Frame(self._root, padding=(10, 0, 10, 6))
        status_frame.grid(row=2, column=0, sticky="ew")
        status_frame.columnconfigure(1, weight=1)
        ttk.Label(status_frame, text="Last event:").grid(row=0, column=0, sticky="w")
        self._last_event_label = ttk.Label(status_frame, text="waiting")
        self._last_event_label.grid(row=0, column=1, sticky="w", padx=(6, 0))

        detail_frame = ttk.Frame(self._root, padding=(10, 0, 10, 10))
        detail_frame.grid(row=3, column=0, sticky="nsew")
        detail_frame.columnconfigure(0, weight=1)
        detail_frame.rowconfigure(1, weight=1)
        ttk.Label(detail_frame, text="Details").grid(row=0, column=0, sticky="w")

        self._details = tk.Text(detail_frame, height=10, wrap="word")
        self._details.configure(state="disabled")
        self._details.grid(row=1, column=0, sticky="nsew")

    def _status_callback(self, msg: String) -> None:
        try:
            payload = json.loads(msg.data)
        except ValueError:
            payload = {
                "stage_id": "invalid",
                "label": "Invalid Message",
                "status": "failed",
                "success": False,
                "message": msg.data,
                "values": {},
                "stamp": time.time(),
            }
        self._queue.put(payload)

    def _poll_queue(self) -> None:
        while not self._queue.empty():
            payload = self._queue.get()
            self._apply_payload(payload)
        if not rospy.is_shutdown():
            self._root.after(100, self._poll_queue)
        else:
            self._root.after(100, self._handle_close)

    def _apply_payload(self, payload) -> None:
        stage_id = str(payload.get("stage_id", "unknown"))
        label = str(payload.get("label", stage_id))
        status = str(payload.get("status", "unknown"))
        success = payload.get("success", None)
        message = str(payload.get("message", ""))
        values = payload.get("values", {})
        updated = self._format_time(payload.get("stamp"))

        row_values = (
            status,
            "" if success is None else str(bool(success)).lower(),
            message,
            self._format_values(values),
            updated,
        )
        tag = status if status in ("pending", "running", "success", "failed", "skipped") else ""
        if stage_id in self._rows:
            self._tree.item(stage_id, text=label, values=row_values, tags=(tag,))
        else:
            self._tree.insert("", "end", iid=stage_id, text=label, values=row_values, tags=(tag,))
            self._rows[stage_id] = True

        self._payloads[stage_id] = payload
        self._latest_payload = payload
        self._last_event_label.configure(text="%s: %s" % (label, status))
        selected = self._tree.selection()
        if not selected or selected[0] == stage_id:
            self._tree.selection_set(stage_id)
            self._update_details(payload)

    def _handle_selection(self, _event=None) -> None:
        selected = self._tree.selection()
        if not selected:
            return
        selected_stage = selected[0]
        payload = self._payloads.get(selected_stage)
        if payload is not None:
            self._update_details(payload)

    def _update_details(self, payload) -> None:
        pretty = json.dumps(payload, indent=2, sort_keys=True)
        self._details.configure(state="normal")
        self._details.delete("1.0", tk.END)
        self._details.insert("1.0", pretty)
        self._details.configure(state="disabled")

    @staticmethod
    def _format_values(values) -> str:
        if not isinstance(values, dict) or not values:
            return ""
        parts = []
        for key in sorted(values.keys()):
            value = values[key]
            if isinstance(value, dict):
                nested = ",".join("%s=%s" % (nested_key, value[nested_key]) for nested_key in sorted(value.keys()))
                parts.append("%s:{%s}" % (key, nested))
            else:
                parts.append("%s=%s" % (key, value))
        text = "  ".join(parts)
        if len(text) > 220:
            return text[:217] + "..."
        return text

    @staticmethod
    def _format_time(stamp) -> str:
        try:
            return time.strftime("%H:%M:%S", time.localtime(float(stamp)))
        except (TypeError, ValueError):
            return ""

    def _handle_close(self) -> None:
        self._root.destroy()

    def run(self) -> None:
        self._root.mainloop()


def main() -> None:
    rospy.init_node("usb_c_insertion_workflow_gui", disable_signals=True)
    WorkflowGui().run()


if __name__ == "__main__":
    main()
