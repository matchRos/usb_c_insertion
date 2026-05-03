#!/usr/bin/env python3

from __future__ import annotations

import os
import sys

from geometry_msgs.msg import PoseStamped
import rospy

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from param_utils import get_param
from pose_persistence import save_pose_stamped
from preinsert_workflow_helpers import PreinsertWorkflowHelpers
from presentation_snapshot_recorder import PresentationSnapshotRecorder


class PreinsertAlignmentWorkflow:
    """
    High-level workflow up to the waiting pose in front of the USB-C port.

    This file intentionally contains the readable process. The ROS actions,
    TF math, retries, and goal construction live in PreinsertWorkflowHelpers.
    """

    def __init__(self):
        self._helpers = PreinsertWorkflowHelpers()
        self._snapshots = PresentationSnapshotRecorder()
        self._updated_port_pose_topic = self._helpers._required_str_param("~workflow/updated_port_pose_topic")
        self._updated_port_pose_publisher = rospy.Publisher(
            self._updated_port_pose_topic,
            PoseStamped,
            queue_size=1,
            latch=True,
        )
        self._updated_port_pose_path = str(
            get_param("~workflow/updated_port_pose_path", "/tmp/usb_c_insertion_latest_port_pose.json")
        ).strip()

    def run(self) -> bool:
        rospy.loginfo(
            "[usb_c_insertion] event=preinsert_workflow_start overview_vision_mode=%s target_card_index=%d",
            self._helpers.overview_vision_mode(),
            self._helpers.target_card_index(),
        )

        if not self._helpers.wait_for_dependencies():
            return False

        overview_pose = self._helpers.load_overview_pose()
        if self._helpers.move_to_pose(overview_pose, "overview_pose") is None:
            return False

        if self._helpers.uses_legacy_overview_vision():
            coarse_port_pose = self._helpers.run_overview_vision()
            if coarse_port_pose is None:
                return False
            self._snapshots.capture_port_pose_axes(
                "01_overview_initial_port_estimate.png",
                "01 Overview position",
                coarse_port_pose,
            )

            camera_pose = self._helpers.plan_camera_pose_from_port(coarse_port_pose)
            if camera_pose is None:
                return False
            if self._helpers.move_to_pose(camera_pose, "camera_to_coarse_port") is None:
                return False
        else:
            rospy.loginfo(
                "[usb_c_insertion] event=preinsert_overview_vision_skipped mode=%s target_card_index=%d",
                self._helpers.overview_vision_mode(),
                self._helpers.target_card_index(),
            )
        self._snapshots.capture_current_view(
            "02_camera_at_initial_estimate.png",
            "02 Camera at initial estimate",
        )

        if (
            self._helpers.uses_usb_card_overview_vision()
            and self._helpers.usb_card_overview_center_before_yaw_enabled()
        ):
            if self._helpers.center_port_in_image(
                usb_card_target_point=self._helpers.usb_card_overview_center_target_point(),
                usb_card_require_connector=self._helpers.usb_card_overview_center_require_connector(),
                label="overview_card_centering",
            ) is None:
                return False
            self._snapshots.capture_marker_alignment(
                "03_centered_on_usb_card_overview.png",
                "03 Centered on USB card overview",
                fallback_marker_center=PresentationSnapshotRecorder.center_from_center_result(
                    self._helpers.latest_center_port_result()
                ),
            )

        if not self._helpers.align_housing_yaw():
            return False
        self._snapshots.capture_marker_alignment(
            "04_after_yaw_alignment_before_centering.png",
            "04 After yaw alignment",
        )

        if (
            self._helpers.uses_usb_card_overview_vision()
            and self._helpers.usb_card_coarse_camera_move_enabled()
        ):
            coarse_plane = self._helpers.estimate_housing_plane(
                "coarse_card_depth_update",
                usb_card_target_point=self._helpers.usb_card_coarse_target_point(),
                usb_card_require_connector=self._helpers.usb_card_coarse_require_connector(),
            )
            if coarse_plane is None:
                return False
            if not self._helpers.validate_plane_quality(coarse_plane, "coarse_card_depth_update"):
                return False

            camera_pose = self._helpers.plan_camera_pose_from_plane(
                coarse_plane,
                "coarse_card_depth_update",
            )
            if camera_pose is None:
                return False
            if self._helpers.move_to_pose(camera_pose, "camera_to_usb_card_coarse", accurate=True) is None:
                return False
            self._snapshots.capture_current_view(
                "05_camera_at_usb_card_coarse.png",
                "05 Camera at USB card coarse estimate",
            )

            if self._helpers.usb_card_refine_yaw_after_coarse_move():
                if not self._helpers.align_housing_yaw():
                    return False
                self._snapshots.capture_marker_alignment(
                    "06_after_usb_card_refine_yaw.png",
                    "06 After USB card refine yaw",
                )

        if self._helpers.center_port_in_image() is None:
            return False
        self._snapshots.capture_marker_alignment(
            "07_centered_over_port.png",
            "07 Camera over port center",
            fallback_marker_center=PresentationSnapshotRecorder.center_from_center_result(
                self._helpers.latest_center_port_result()
            ),
        )

        orientation_check = self._helpers.estimate_housing_plane(
            "orientation_check",
            usb_card_target_point=self._helpers.usb_card_final_target_point(),
            usb_card_require_connector=self._helpers.usb_card_final_require_connector(),
        )
        if orientation_check is None:
            return False
        if not self._helpers.validate_plane_quality(orientation_check, "orientation_check"):
            return False

        looming_result = self._helpers.verify_looming()
        if looming_result is None:
            return False
        recenter_ok, recenter_result = self._helpers.recenter_after_looming_if_needed(looming_result)
        if not recenter_ok:
            return False
        marker_center = PresentationSnapshotRecorder.center_from_center_result(recenter_result)
        if marker_center is None:
            marker_center = PresentationSnapshotRecorder.center_from_looming_result(looming_result)
        self._snapshots.capture_marker_alignment(
            "08_after_verify_looming.png",
            "08 After looming verification",
            fallback_marker_center=marker_center,
        )

        final_plane = self._helpers.estimate_housing_plane(
            "final_depth_update",
            usb_card_target_point=self._helpers.usb_card_final_target_point(),
            usb_card_require_connector=self._helpers.usb_card_final_require_connector(),
        )
        if final_plane is None:
            return False
        if not self._helpers.validate_plane_quality(final_plane, "final_depth_update"):
            return False

        updated_port_pose = self._helpers.build_updated_port_pose(final_plane)
        if updated_port_pose is None:
            return False
        self._updated_port_pose_publisher.publish(updated_port_pose)
        save_pose_stamped(updated_port_pose, self._updated_port_pose_path)

        tcp_precontact_pose = self._helpers.plan_tcp_precontact_pose(updated_port_pose)
        if tcp_precontact_pose is None:
            return False
        if self._helpers.move_to_pose(tcp_precontact_pose, "tcp_precontact_wait", accurate=True) is None:
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
