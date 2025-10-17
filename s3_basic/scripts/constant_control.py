#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from std_msgs.msg import String, Bool, Int64
from geometry_msgs.msg import Twist

# import the message type to use


class Publisher(Node):
    def __init__(self) -> None:
        # initialize base class (must happen before everything else)
        super().__init__("publisher")
        self.get_logger().info("sending constant control...")

        # a timer
        # self.totalTime = 0

        # create publisher with: self.create_publisher(<msg type>, <topic>, <qos>)
        self.publisher = self.create_publisher(Twist, "/cmd_vel", 10)
        #self.kill_publisher = self.create_publisher(Bool, "/kill", 10)

        # create a timer with: self.create_timer(<second>, <callback>)
        self.timer = self.create_timer(0.2, self.timer_callback)
        self.kill_subscriber = self.create_subscription(Bool, "/kill", self.kill_callback, 10)

    def timer_callback(self) -> None:
        """
        Heartbeat callback triggered by the timer
        """
        # construct heartbeat message
        msg = Twist()
        msg.linear.x = -2.0
        msg.angular.z = 0.0
        self.get_logger().info(f"{msg}\n")
        #msg.data = "sending constant control..."


        # publish heartbeat counter
        self.publisher.publish(msg)

        # increment counter
        # self.hb_counter += 1

    def kill_callback(self, msg: Bool) -> None:
        """
        Sensor health callback triggered by subscription
        """
        if msg.data == True:
            velocity_msg = Twist()
            velocity_msg.linear.x = 0.0
            velocity_msg.linear.y = 0.0
            velocity_msg.linear.z = 0.0
            velocity_msg.angular.x = 0.0
            velocity_msg.angular.y = 0.0
            velocity_msg.angular.z = 0.0
            self.publisher.publish(velocity_msg)

            self.get_logger().fatal("Kill Received")
            self.timer.cancel()

            kill_msg = 0
            self.get_logger().info(f"{kill_msg}")
            self.publisher.publish(kill_msg)



if __name__ == "__main__":
    rclpy.init()        # initialize ROS2 context (must run before any other rclpy call)
    node = Publisher()  # instantiate the heartbeat node
    rclpy.spin(node)    # Use ROS2 built-in schedular for executing the node
    rclpy.shutdown()    # cleanly shutdown ROS2 context
