#!/usr/bin/env python3

import rospy
import actionlib

from param_utils import get_param
from ur_dashboard_msgs.msg import SetModeAction, SetModeGoal, RobotMode
from std_srvs.srv import Trigger, TriggerRequest
from controller_manager_msgs.srv import (
    LoadController,
    LoadControllerRequest,
    SwitchController,
    SwitchControllerRequest,
    ListControllers,
    ListControllersRequest,
)

class UREnable:
    def __init__(self):
        self.ur_hardware_interface_topic = get_param(
            '~ur_hardware_interface_topic',
            '/ur_hardware_interface'
        )

        # Name of the twist controller to load/start
        self.twist_controller_name = get_param(
            '~twist_controller_name',
            'twist_controller'
        )

        # Controllers that are allowed to keep running
        self.allowed_running_controllers = {
            'joint_state_controller',
            'speed_scaling_state_controller',
            'force_torque_sensor_controller',
            self.twist_controller_name,
        }

        # Controller manager namespace
        self.controller_manager_ns = get_param(
            '~controller_manager_ns',
            '/controller_manager'
        )

        self.startup_timeout = get_param('~startup_timeout', 0.1)

    def stop_dashboard_server(self):
        rospy.loginfo("Stopping dashboard server")
        trigger_client = rospy.ServiceProxy(
            self.ur_hardware_interface_topic + '/dashboard/stop',
            Trigger
        )
        rospy.loginfo("Waiting for dashboard stop service")
        trigger_client.wait_for_service()

        trigger_request = TriggerRequest()
        response = trigger_client(trigger_request)

        if not response.success:
            rospy.logwarn("Dashboard stop service returned: %s", response.message)
        else:
            rospy.loginfo("Dashboard server stopped")

    def enable_robot(self):
        rospy.loginfo("Waiting for set_mode action server")
        client = actionlib.SimpleActionClient(
            self.ur_hardware_interface_topic + "/set_mode",
            SetModeAction
        )
        client.wait_for_server()

        set_mode_goal = SetModeGoal()
        robot_mode = RobotMode()
        robot_mode.mode = 7  # RUNNING

        set_mode_goal.target_robot_mode = robot_mode
        set_mode_goal.play_program = True

        rospy.loginfo("Sending enable request")
        client.send_goal(set_mode_goal)
        client.wait_for_result()

        result = client.get_result()
        rospy.loginfo("Robot enabled")
        return result

    def get_controller_services(self):
        load_srv_name = self.controller_manager_ns + '/load_controller'
        switch_srv_name = self.controller_manager_ns + '/switch_controller'
        list_srv_name = self.controller_manager_ns + '/list_controllers'

        rospy.loginfo("Waiting for controller_manager services")
        rospy.wait_for_service(load_srv_name)
        rospy.wait_for_service(switch_srv_name)
        rospy.wait_for_service(list_srv_name)

        load_srv = rospy.ServiceProxy(load_srv_name, LoadController)
        switch_srv = rospy.ServiceProxy(switch_srv_name, SwitchController)
        list_srv = rospy.ServiceProxy(list_srv_name, ListControllers)

        return load_srv, switch_srv, list_srv

    def load_twist_controller(self, load_srv):
        rospy.loginfo("Loading twist controller: %s", self.twist_controller_name)

        req = LoadControllerRequest()
        req.name = self.twist_controller_name
        resp = load_srv(req)

        if resp.ok:
            rospy.loginfo("Twist controller loaded or already available")
        else:
            rospy.logwarn(
                "Could not load controller '%s'. It may already be loaded or not configured.",
                self.twist_controller_name
            )

    def get_running_controllers_to_stop(self, list_srv):
        req = ListControllersRequest()
        resp = list_srv(req)

        controllers_to_stop = []
        for ctrl in resp.controller:
            if ctrl.state == "running" and ctrl.name not in self.allowed_running_controllers:
                controllers_to_stop.append(ctrl.name)

        return controllers_to_stop

    def switch_to_twist_controller(self, switch_srv, controllers_to_stop):
        rospy.loginfo("Controllers to stop: %s", controllers_to_stop)
        rospy.loginfo("Controller to start: %s", self.twist_controller_name)

        req = SwitchControllerRequest()
        req.stop_controllers = controllers_to_stop
        req.start_controllers = [self.twist_controller_name]
        req.strictness = SwitchControllerRequest.STRICT
        req.start_asap = True
        req.timeout = 5.0

        resp = switch_srv(req)

        if not resp.ok:
            rospy.logerr("Failed to switch controllers")
            return False

        rospy.loginfo("Successfully switched to twist controller")
        return True

    def main(self):
        rospy.sleep(self.startup_timeout)  # Allow some time for the robot to initialize
        self.stop_dashboard_server()
        self.enable_robot()

        rospy.sleep(1.0)  # Short delay to ensure robot is in running mode before managing controllers
        load_srv, switch_srv, list_srv = self.get_controller_services()

        self.load_twist_controller(load_srv)
        controllers_to_stop = self.get_running_controllers_to_stop(list_srv)

        success = self.switch_to_twist_controller(switch_srv, controllers_to_stop)

        if success:
            rospy.loginfo("Robot enabled and twist controller is running")
        else:
            rospy.logerr("Robot enabled, but controller switch failed")

        rospy.signal_shutdown("Done")

if __name__ == '__main__':
    rospy.init_node('UR_enable')
    node = UREnable()
    node.main()
