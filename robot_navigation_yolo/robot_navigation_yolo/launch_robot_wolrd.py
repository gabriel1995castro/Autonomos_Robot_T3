#!/usr/bin/env python3
import rclpy
import random
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, LogInfo, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

def generate_random_values(context):
    # Gera valores aleatórios para x, y e yaw
    x = random.uniform(-10, 10)  
    y = random.uniform(-10, 10)  
    yaw = random.uniform(-3.14, 3.14) 

    return [
        LogInfo(
            condition=None,
            msg=f'Launching with random values: x={x}, y={y}, yaw={yaw}'
        ),
        Node(
            package='clearpath_gz',
            executable='simulation.launch.py',
            name='simulation_node',
            output='screen',
            parameters=[{'x': x, 'y': y, 'yaw': yaw}]
        )
    ]

def generate_launch_description():
    return LaunchDescription([
        OpaqueFunction(function=generate_random_values)
    ])
