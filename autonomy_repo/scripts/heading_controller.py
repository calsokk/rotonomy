#!/usr/bin/env python3 

import numpy as np
import rclpy 

from asl_tb3_lib.control import BaseHeadingController 
from asl_tb3_lib.math_utils import wrap_angle 
from asl_tb3_msgs.msg import TurtleBotControl, TurtleBotState

class HeadingController(BaseHeadingController):
    def __init__(self, node_name: str = "heading_control") -> None:
        super().__init__(node_name)
        self.declare_parameter("kp", 2.0)

    @property
    def kp(self) -> float: 
        # returns real time kp (gain)
        return self.get_parameter("kp").value

    def compute_control_with_goal(self, cur_state: TurtleBotState, desired_state: TurtleBotState):
        desired_theta = desired_state.theta 
        cur_theta = cur_state.theta

        heading_error = wrap_angle(desired_theta - cur_theta)

        omega = self.kp * heading_error
        new_control = TurtleBotControl()
        new_control.v = 0.0
        new_control.omega = omega

        return new_control
    

if __name__ == "__main__":
    rclpy.init()
    node = HeadingController()
    rclpy.spin(node) 
    rclpy.shutdown()


        
