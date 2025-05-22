import launch
from launch_ros.actions import ComposableNodeContainer
from launch_ros.descriptions import ComposableNode


# def generate_launch_description():
#     """Generate launch description with multiple components."""
#     container = ComposableNodeContainer(
#             name='image_container',
#             namespace='',
#             package='rclcpp_components',
#             executable='component_container',
#             composable_node_descriptions=[
#                 ComposableNode(
#                     package='image_tools',
#                     plugin='image_tools::Cam2Image',
#                     name='cam2image',
#                     remappings=[('/image', '/burgerimage')],
#                     parameters=[{'width': 320, 'height': 240, 'burger_mode': True, 'history': 'keep_last'}],
#                     extra_arguments=[{'use_intra_process_comms': True}]
#                     )
#             ],
#             output='both',
#             exec_name = "talk_1"
#     )

#     return launch.LaunchDescription([container])


# import launch
# from launch_ros.actions import ComposableNodeContainer
# from launch_ros.descriptions import ComposableNode


# def generate_launch_description():
#     """Generate launch description with multiple components."""
#     container = ComposableNodeContainer(
#             name='image_container',
#             namespace='',
#             package='rclcpp_components',
#             executable='component_container',
#             composable_node_descriptions=[
#                 ComposableNode(
#                     package='image_tools',
#                     plugin='image_tools::Cam2Image',
#                     name='cam2image',
#                     remappings=[('/image', '/burgerimage')],
#                     parameters=[{'width': 320, 'height': 240, 'burger_mode': True, 'history': 'keep_last'}],
#                     extra_arguments=[{'use_intra_process_comms': True}]),
#                 ComposableNode(
#                     package='image_tools',
#                     plugin='image_tools::ShowImage',
#                     name='showimage',
#                     remappings=[('/image', '/burgerimage')],
#                     parameters=[{'history': 'keep_last'}],
#                     extra_arguments=[{'use_intra_process_comms': True}])
#             ],
#             output='both',
#             exec_name = "talk_1"
#     )

#     return launch.LaunchDescription([container])


import launch
from launch_ros.actions import ComposableNodeContainer
from launch_ros.descriptions import ComposableNode
from launch import LaunchDescription
from launch_ros.actions import LoadComposableNodes, Node
from launch_ros.descriptions import ComposableNode


def generate_launch_description():
    """Generate launch description with multiple components."""
    container = ComposableNodeContainer(
            name='my_container',
            namespace='',
            package='rclcpp_components',
            executable='component_container',
            composable_node_descriptions=[
                ComposableNode(
                    package='composition',
                    plugin='composition::Talker',
                    name='talker'),
                ComposableNode(
                    package='composition',
                    plugin='composition::Listener',
                    name='listener')
            ],
            output='screen',
            exec_name = "test1"
    )

    load_composable_nodes = LoadComposableNodes(
        target_container='my_container',
        composable_node_descriptions=[
           ComposableNode(
                    package='composition',
                    plugin='composition::Talker',
                    name='talker2'),
                ComposableNode(
                    package='composition',
                    plugin='composition::Listener',
                    name='listener2')
        ],
    )

    return launch.LaunchDescription([container, load_composable_nodes])

# from launch import LaunchDescription
# from launch_ros.actions import LoadComposableNodes, Node
# from launch_ros.descriptions import ComposableNode

# def generate_launch_description():
#     container = Node(
#         name='my_container',
#         package='rclcpp_components',
#         executable='component_container',
#         output='both',
#         exec_name = "talk_1"
#     )

#     load_composable_nodes = LoadComposableNodes(
#         target_container='my_container',
#         composable_node_descriptions=[
#            ComposableNode(
#                     package='composition',
#                     plugin='composition::Talker',
#                     name='talker'),
#                 ComposableNode(
#                     package='composition',
#                     plugin='composition::Listener',
#                     name='listener')
#         ],
#     )

#     return LaunchDescription([container, load_composable_nodes])
