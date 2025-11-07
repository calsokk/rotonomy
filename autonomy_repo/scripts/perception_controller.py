#!/usr/bin/env python3

import numpy as np
import rclpy

from asl_tb3_lib.control import BaseController
from asl_tb3_msgs.msg import TurtleBotControl

class PerceptionController(BaseController):
    def __init__(self, node_name: str = "perception_controller") -> None:
        super().__init__(node_name)
        self.declare_parameter("active", True)
        self.declare_parameter("start_time", 0.0)
        self.declare_parameter("prev_active", True)

    @property
    def active(self) -> bool:
        # returns real time kp (gain)
        return self.get_parameter("active").value

    @property
    def start_time(self) -> float:
        return self.get_parameter("start_time").value

    @property
    def prev_active(self) -> bool:
        return self.get_parameter("prev_active").value

    def compute_control(self):
        new_control = TurtleBotControl()
        temp = self.active
        if self.active == True:
            new_control.omega = 0.5
        else:
            new_control.omega = 0.0
            if self.prev_active == True:
                self.set_parameters([rclpy.Parameter("start_time", value=self.get_clock().now().nanoseconds/1e9)])
            start_time = self.start_time
            cur_time = self.get_clock().now().nanoseconds/1e9
            time_passed = cur_time - start_time
            if time_passed >= 5.0:
                self.set_parameters([rclpy.Parameter("active", value=True)])
        self.set_parameters([rclpy.Parameter("prev_active", value=temp)])
        return new_control


if __name__ == "__main__":
    rclpy.init()
    node = PerceptionController()
    rclpy.spin(node)
    rclpy.shutdown()




