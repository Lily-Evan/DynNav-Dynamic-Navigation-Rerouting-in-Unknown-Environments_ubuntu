#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
import numpy as np

class NCEMonitor(Node):
    def __init__(self):
        super().__init__('nce_monitor')

        self.create_subscription(Odometry, '/ekf_pose', self.pose_callback, 10)

        self.coverage = 0.0
        self.path_length = 0.0
        self.last_pos = None

        self.timer = self.create_timer(1.0, self.report_nce)

    def pose_callback(self, msg):
        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y
        pos = np.array([x, y])

        # Update path length
        if self.last_pos is not None:
            self.path_length += np.linalg.norm(pos - self.last_pos)

        self.last_pos = pos

        # Fake demo coverage (replace with real grid coverage)
        self.coverage = min(1.0, self.coverage + 0.0005)

    def report_nce(self):
        if self.path_length > 0:
            nce = self.coverage / self.path_length
            self.get_logger().info(f"NCE: {nce:.6f}")

def main(args=None):
    rclpy.init(args=args)
    node = NCEMonitor()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()

