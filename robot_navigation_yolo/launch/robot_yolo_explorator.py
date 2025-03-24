import launch
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, LogInfo
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        # Declaração de parâmetros (caso necessário)
        DeclareLaunchArgument('use_sim_time', default_value='false', description='Use simulation time if true'),

        # Nó de navegação
        Node(
            package='robot_navigation_yolo', 
            executable='navigation_node',  
            name='navigation_node',  
            output='screen',
            
        ),

        # Nó de exploração
        Node(
            package='robot_navigation_yolo',  
            executable='fontrier_based_exploratin',  
            name='exploration_node',  
            output='screen',
            
        ),

        # Nó do detector YOLO
        Node(
            package='robot_navigation_yolo',  
            executable='yolo_detector_node',  
            name='yolo_detector_node',  
            output='screen',
            
        ),
    ])

