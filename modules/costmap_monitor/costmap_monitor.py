#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from nav_msgs.msg import OccupancyGrid
import numpy as np

class CostmapMonitor(Node):
    def __init__(self):
        super().__init__('costmap_monitor')

        self.create_subscription(
            OccupancyGrid,
            '/global_costmap/costmap',
            self.costmap_callback,
            10
        )

        self.costmap = None
        self.threshold = 80  # danger threshold

    def costmap_callback(self, msg):
        # Convert costmap data → 2D array
        w = msg.info.width
        h = msg.info.height
        data = np.array(msg.data).reshape((h, w))

        self.costmap = data

        # Count dangerous cells
        dangerous = np.sum(data > self.threshold)

        self.get_logger().info(
            f"Dangerous cells: {dangerous}"
        )

def main(args=None):
    rclpy.init(args=args)
    node = CostmapMonitor()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
