from launch import LaunchDescription
from launch_ros.actions import Node
import os
from ament_index_python.packages import get_package_share_directory

def generate_launch_description():
    param_path = os.path.join(
        get_package_share_directory('node_state_monitor'),
        'config',
        'node_monitor_params.yaml'
    )

    return LaunchDescription([
        Node(
            package='node_state_monitor',
            executable='node_monitor_node',
            name='node_monitor_node',
            parameters=[param_path],
            output='screen'
        )
    ])