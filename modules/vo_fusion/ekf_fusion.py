#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
import numpy as np

from nav_msgs.msg import Odometry
from sensor_msgs.msg import Imu

class EKFFusion(Node):
    def __init__(self):
        super().__init__('ekf_fusion')

        # EKF state: [x, y, yaw]
        self.x = np.zeros((3,1))
        self.P = np.eye(3) * 0.1  # covariance

        # Subscribers
        self.create_subscription(Odometry, '/wheel/odom', self.odom_callback, 10)
        self.create_subscription(Odometry, '/vo', self.vo_callback, 10)
        self.create_subscription(Imu, '/imu', self.imu_callback, 10)

        # Publisher
        self.pub = self.create_publisher(Odometry, '/ekf_pose', 10)

    def odom_callback(self, msg):
        # Prediction step
        vx = msg.twist.twist.linear.x
        wyaw = msg.twist.twist.angular.z
        dt = 0.05

        F = np.eye(3)
        B = np.array([[dt, 0],[0, dt],[0, dt]])

        u = np.array([[vx],[0]])

        self.x = self.x + B @ u
        self.P = F @ self.P @ F.T + np.eye(3)*0.01

    def vo_callback(self, msg):
        # VO measurement
        z = np.array([[msg.pose.pose.position.x],
                      [msg.pose.pose.position.y]])

        H = np.array([[1,0,0],
                      [0,1,0]])

        R = np.eye(2)*0.02

        # Kalman gain
        K = self.P @ H.T @ np.linalg.inv(H @ self.P @ H.T + R)
        y = z - (H @ self.x)

        self.x = self.x + K @ y
        self.P = (np.eye(3) - K @ H) @ self.P

        self.publish_ekf(msg)

    def imu_callback(self, msg):
        # Yaw only (2D robot)
        yaw = msg.orientation.z
        self.x[2,0] = yaw

    def publish_ekf(self, source_msg):
        odom = Odometry()
        odom.header = source_msg.header
        odom.pose.pose.position.x = float(self.x[0])
        odom.pose.pose.position.y = float(self.x[1])
        odom.pose.pose.orientation.z = float(self.x[2])

        self.pub.publish(odom)

def main(args=None):
    rclpy.init(args=args)
    node = EKFFusion()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()

