from setuptools import setup, find_packages

package_name = 'cybersecurity_ros2'

setup(
    name=package_name,
    version='0.0.1',
    packages=find_packages(),
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='panagiota',
    maintainer_email='panagiota@example.com',
    description='ROS2 cybersecurity nodes (attacker + IDS)',
    license='MIT',
    entry_points={
        'console_scripts': [
            'odom_attacker = cybersecurity_ros2.nodes.odom_attacker:main',
            'odom_ids = cybersecurity_ros2.nodes.odom_ids:main',
        ],
    },
)
