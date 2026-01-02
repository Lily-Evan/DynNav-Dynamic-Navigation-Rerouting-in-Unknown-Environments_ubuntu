#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
import random
from collections import deque

class OdomAttacker(Node):
    def __init__(self):
        super().__init__('odom_attacker')

        self.declare_parameter('mode', 'spoof_bias')
        self.declare_parameter('in_topic', '/odom')
        self.declare_parameter('out_topic', '/odom_attacked')
        self.declare_parameter('bias_x', 1.0)
        self.declare_parameter('bias_y', 0.0)
        self.declare_parameter('replay_delay', 40)

        self.mode = self.get_parameter('mode').value
        self.in_topic = self.get_parameter('in_topic').value
        self.out_topic = self.get_parameter('out_topic').value
        self.bias_x = float(self.get_parameter('bias_x').value)
        self.bias_y = float(self.get_parameter('bias_y').value)
        self.replay_delay = int(self.get_parameter('replay_delay').value)

        self.buffer = deque(maxlen=200)

        self.sub = self.create_subscription(Odometry, self.in_topic, self.cb, 50)
        self.pub = self.create_publisher(Odometry, self.out_topic, 50)

        self.get_logger().info(f"Attacker running in mode={self.mode}")

    def cb(self, msg):
        self.buffer.append(msg)

        if self.mode == 'replay' and len(self.buffer) > self.replay_delay:
            out = self.buffer[-self.replay_delay]
        else:
            out = msg
            out.pose.pose.position.x += self.bias_x
            out.pose.pose.position.y += self.bias_y

        self.pub.publish(out)

def main():
    rclpy.init()
    node = OdomAttacker()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
