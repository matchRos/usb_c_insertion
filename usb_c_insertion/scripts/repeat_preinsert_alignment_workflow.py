#!/usr/bin/env python3

from __future__ import annotations

import json
import math
import os
import sys
import time
from typing import Dict, List, Optional, Tuple

import rospy

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from param_utils import get_param
from pose_persistence import load_pose_stamped
from preinsert_alignment_workflow import PreinsertAlignmentWorkflow


PHOTO_POSE_PARAMS = (
    "/photo_pose/x",
    "/photo_pose/y",
    "/photo_pose/z",
    "/photo_pose/qx",
    "/photo_pose/qy",
    "/photo_pose/qz",
    "/photo_pose/qw",
)


class RepeatPreinsertAlignmentWorkflow:
    def __init__(self):
        self._iterations = max(1, int(get_param("~repeat_preinsert/iterations", 10)))
        self._output_path = str(
            get_param(
                "~repeat_preinsert/output_path",
                "/tmp/usb_c_insertion_preinsert_repeat_estimates.jsonl",
            )
        ).strip()
        self._latest_pose_path = str(
            get_param("~workflow/updated_port_pose_path", "/tmp/usb_c_insertion_latest_port_pose.json")
        ).strip()
        self._vary_photo_pose = self._bool_param("~repeat_preinsert/vary_photo_pose", True)
        self._variation_x = abs(float(get_param("~repeat_preinsert/photo_pose_variation_x", 0.05)))
        self._variation_y = abs(float(get_param("~repeat_preinsert/photo_pose_variation_y", 0.05)))
        self._settle_between_runs = max(0.0, float(get_param("~repeat_preinsert/settle_between_runs", 1.0)))
        self._stop_on_failure = self._bool_param("~repeat_preinsert/stop_on_failure", True)
        self._snapshot_subdirs = self._bool_param("~repeat_preinsert/snapshot_subdirs", True)
        self._snapshot_base_dir = str(
            get_param(
                "~repeat_preinsert/snapshot_base_dir",
                get_param("~presentation_snapshots/output_dir", "/tmp/usb_c_insertion_presentation_snapshots"),
            )
        ).strip()
        self._target_card_index = int(get_param("~workflow/target_card_index", 1))
        self._base_photo_pose = self._read_photo_pose_params()

    def run(self) -> bool:
        rospy.loginfo(
            "[usb_c_insertion] event=repeat_preinsert_start iterations=%d target_card_index=%d "
            "vary_photo_pose=%s variation_x=%.3f variation_y=%.3f output_path=%s",
            self._iterations,
            self._target_card_index,
            str(self._vary_photo_pose).lower(),
            self._variation_x,
            self._variation_y,
            self._output_path,
        )

        offsets = self._photo_pose_offsets(self._iterations)
        records: List[Dict] = []
        success = True
        try:
            self._prepare_output_file()
            for run_index, (offset_x, offset_y) in enumerate(offsets, start=1):
                if rospy.is_shutdown():
                    success = False
                    break
                record = self._run_once(run_index, offset_x, offset_y)
                records.append(record)
                self._append_record(record)
                if not record["success"]:
                    success = False
                    if self._stop_on_failure:
                        break
                if run_index < len(offsets) and self._settle_between_runs > 0.0:
                    rospy.sleep(self._settle_between_runs)
        finally:
            self._restore_photo_pose_params()

        self._log_summary(records)
        rospy.loginfo(
            "[usb_c_insertion] event=repeat_preinsert_complete success=%s completed=%d requested=%d output_path=%s",
            str(success).lower(),
            len(records),
            self._iterations,
            self._output_path,
        )
        return success

    def _run_once(self, run_index: int, offset_x: float, offset_y: float) -> Dict:
        run_started = rospy.Time.now()
        applied_x = float(self._base_photo_pose["/photo_pose/x"] + offset_x)
        applied_y = float(self._base_photo_pose["/photo_pose/y"] + offset_y)
        if self._vary_photo_pose:
            rospy.set_param("/photo_pose/x", applied_x)
            rospy.set_param("/photo_pose/y", applied_y)
        else:
            offset_x = 0.0
            offset_y = 0.0
            applied_x = float(self._base_photo_pose["/photo_pose/x"])
            applied_y = float(self._base_photo_pose["/photo_pose/y"])

        if self._snapshot_subdirs and self._snapshot_base_dir:
            snapshot_dir = os.path.join(self._snapshot_base_dir, "repeat_%02d" % run_index)
            rospy.set_param("/presentation_snapshots/output_dir", snapshot_dir)
        else:
            snapshot_dir = str(get_param("~presentation_snapshots/output_dir", ""))

        rospy.loginfo(
            "[usb_c_insertion] event=repeat_preinsert_iteration_start index=%d/%d "
            "photo_pose=(%.4f,%.4f) offset=(%.4f,%.4f) snapshot_dir=%s",
            run_index,
            self._iterations,
            applied_x,
            applied_y,
            offset_x,
            offset_y,
            snapshot_dir,
        )

        workflow = PreinsertAlignmentWorkflow()
        success = bool(workflow.run())
        pose = self._load_fresh_estimate(run_started)
        record = {
            "run_index": run_index,
            "success": success,
            "stamp": rospy.Time.now().to_sec(),
            "wall_time": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "target_card_index": self._target_card_index,
            "photo_pose": {
                "x": applied_x,
                "y": applied_y,
                "z": float(self._base_photo_pose["/photo_pose/z"]),
            },
            "photo_pose_offset": {"x": float(offset_x), "y": float(offset_y)},
            "latest_estimate_path": self._latest_pose_path,
            "snapshot_dir": snapshot_dir,
            "estimated_port_pose": self._pose_to_dict(pose),
        }
        rospy.loginfo(
            "[usb_c_insertion] event=repeat_preinsert_iteration_complete index=%d success=%s estimate=%s",
            run_index,
            str(success).lower(),
            self._pose_summary(pose),
        )
        return record

    def _load_fresh_estimate(self, run_started: rospy.Time):
        pose = load_pose_stamped(self._latest_pose_path)
        if pose is None:
            return None
        if pose.header.stamp and pose.header.stamp < run_started:
            rospy.logwarn(
                "[usb_c_insertion] event=repeat_preinsert_stale_estimate path=%s pose_stamp=%.3f run_started=%.3f",
                self._latest_pose_path,
                pose.header.stamp.to_sec(),
                run_started.to_sec(),
            )
            return None
        return pose

    def _read_photo_pose_params(self) -> Dict[str, float]:
        values = {}
        for name in PHOTO_POSE_PARAMS:
            if not rospy.has_param(name):
                raise RuntimeError("Missing required ROS parameter: %s" % name)
            values[name] = float(rospy.get_param(name))
        return values

    def _restore_photo_pose_params(self) -> None:
        for name, value in self._base_photo_pose.items():
            rospy.set_param(name, value)
        rospy.loginfo("[usb_c_insertion] event=repeat_preinsert_photo_pose_restored")

    def _photo_pose_offsets(self, count: int) -> List[Tuple[float, float]]:
        if not self._vary_photo_pose:
            return [(0.0, 0.0)] * count
        x = self._variation_x
        y = self._variation_y
        pattern = [
            (0.0, 0.0),
            (-x, 0.0),
            (x, 0.0),
            (0.0, -y),
            (0.0, y),
            (-x, -y),
            (-x, y),
            (x, -y),
            (x, y),
            (-0.5 * x, 0.5 * y),
            (0.5 * x, -0.5 * y),
            (-0.5 * x, -0.5 * y),
            (0.5 * x, 0.5 * y),
        ]
        offsets = []
        while len(offsets) < count:
            offsets.extend(pattern)
        return offsets[:count]

    def _prepare_output_file(self) -> None:
        if not self._output_path:
            return
        directory = os.path.dirname(self._output_path)
        if directory:
            os.makedirs(directory, exist_ok=True)
        with open(self._output_path, "w", encoding="utf-8") as handle:
            handle.write("")

    def _append_record(self, record: Dict) -> None:
        if not self._output_path:
            return
        with open(self._output_path, "a", encoding="utf-8") as handle:
            json.dump(record, handle, sort_keys=True)
            handle.write("\n")

    @staticmethod
    def _pose_to_dict(pose) -> Optional[Dict]:
        if pose is None:
            return None
        return {
            "frame_id": pose.header.frame_id,
            "stamp": pose.header.stamp.to_sec() if pose.header.stamp else 0.0,
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

    @staticmethod
    def _pose_summary(pose) -> str:
        if pose is None:
            return "none"
        return "(%.4f,%.4f,%.4f)" % (
            pose.pose.position.x,
            pose.pose.position.y,
            pose.pose.position.z,
        )

    def _log_summary(self, records: List[Dict]) -> None:
        positions = []
        for record in records:
            pose = record.get("estimated_port_pose")
            if not pose:
                continue
            position = pose["position"]
            positions.append((float(position["x"]), float(position["y"]), float(position["z"])))
        if not positions:
            rospy.logwarn("[usb_c_insertion] event=repeat_preinsert_summary no_estimates=true")
            return
        mean = tuple(sum(position[i] for position in positions) / float(len(positions)) for i in range(3))
        std = tuple(
            math.sqrt(sum((position[i] - mean[i]) ** 2 for position in positions) / float(len(positions)))
            for i in range(3)
        )
        max_xy = max(math.hypot(position[0] - mean[0], position[1] - mean[1]) for position in positions)
        rospy.loginfo(
            "[usb_c_insertion] event=repeat_preinsert_summary estimates=%d "
            "mean=(%.4f,%.4f,%.4f) std_mm=(%.2f,%.2f,%.2f) max_xy_from_mean_mm=%.2f",
            len(positions),
            mean[0],
            mean[1],
            mean[2],
            std[0] * 1000.0,
            std[1] * 1000.0,
            std[2] * 1000.0,
            max_xy * 1000.0,
        )

    @staticmethod
    def _bool_param(name: str, default: bool) -> bool:
        value = get_param(name, default)
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            return value.strip().lower() in ("true", "1", "yes", "on")
        return bool(value)


def main() -> None:
    rospy.init_node("usb_c_insertion_repeat_preinsert_alignment_workflow")
    try:
        workflow = RepeatPreinsertAlignmentWorkflow()
        ok = workflow.run()
    except Exception as exc:
        rospy.logerr("[usb_c_insertion] event=repeat_preinsert_exception error=%s", exc)
        ok = False
    if not ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
