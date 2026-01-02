#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from std_msgs.msg import Bool
import math

class OdomIDS(Node):
    def __init__(self):
        super().__init__('odom_ids')

        self.declare_parameter('sigma', 0.2)
        self.declare_parameter('threshold', 9.21)
        self.declare_parameter('consecutive', 5)

        self.sigma = self.get_parameter('sigma').value
        self.threshold = self.get_parameter('threshold').value
        self.consecutive = self.get_parameter('consecutive').value

        self.legit = None
        self.bad_count = 0

        self.sub_legit = self.create_subscription(
            Odometry, '/odom', self.cb_legit, 50)
        self.sub_obs = self.create_subscription(
            Odometry, '/odom_attacked', self.cb_obs, 50)

        self.pub_alert = self.create_publisher(Bool, '/security/alert', 10)

        self.get_logger().info("IDS started")

    def cb_legit(self, msg):
        self.legit = msg

    def cb_obs(self, msg):
        if self.legit is None:
            return

        dx = msg.pose.pose.position.x - self.legit.pose.pose.position.x
        dy = msg.pose.pose.position.y - self.legit.pose.pose.position.y

        d2 = (dx*dx + dy*dy) / (self.sigma*self.sigma)

        if d2 > self.threshold:
            self.bad_count += 1
        else:
            self.bad_count = max(0, self.bad_count - 1)

        alert = Bool()
        alert.data = self.bad_count >= self.consecutive
        self.pub_alert.publish(alert)

        if alert.data:
            self.get_logger().warn(f"ATTACK DETECTED d2={d2:.2f}")

def main():
    rclpy.init()
    node = OdomIDS()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
