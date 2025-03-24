from setuptools import setup, find_packages

package_name = 'robot_navigation_yolo'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', ['launch/robot_yolo_explorator.py']),
    ],
    install_requires=[
        'setuptools',
        'launch',
        'launch_ros',
        'rclpy',
    ],
    zip_safe=True,
    maintainer='gabriel',
    maintainer_email='gabriel@todo.todo',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'robot_vision = robot_navigation_yolo.yolo_detector_node:main',
        ],
    },
)


