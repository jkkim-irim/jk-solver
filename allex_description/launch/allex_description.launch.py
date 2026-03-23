import os
from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from ament_index_python.packages import get_package_share_directory
from launch.conditions import IfCondition
from launch_ros.parameter_descriptions import ParameterValue

def load_file(package_name, file_path):
    package_path = get_package_share_directory(package_name)
    absolute_file_path = os.path.join(package_path, file_path)

    try:
        with open(absolute_file_path, 'r') as file:
            return file.read()
    except EnvironmentError: # parent of IOError, OSError *and* WindowsError where available
        return None

def generate_launch_description():
    pkg_allex_description = get_package_share_directory('allex_description')
    
    use_gui_arg = DeclareLaunchArgument(
        name='use_gui',
        default_value='true',  
        choices=['true', 'false'],
        description='Flag to enable joint_state_publisher_gui'
    )
    
    urdf_file_path = os.path.join(pkg_allex_description, 'urdf', 'ALLEX.urdf')
    with open(urdf_file_path, 'r') as infp:
        robot_desc = infp.read()
    
    robot_description = {'robot_description': robot_desc}
    
    # robot_state_publisher 
    robot_state_publisher_node = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        output='screen',
        parameters=[robot_description] 
    )

    # joint_state_publisher_gui 
    joint_state_publisher_gui_node = Node(
        package='joint_state_publisher_gui',
        executable='joint_state_publisher_gui',
        name='joint_state_publisher_gui',
        condition=IfCondition(LaunchConfiguration('use_gui'))
    )

    # Static transform: world -> Base_Link (z=0.685)
    # This allows base_link to be at z=0 in URDF while appearing at z=0.685 in RViz
    world_to_base_transform = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        name='world_to_base_link',
        arguments=['0', '0', '0.685', '0', '0', '0', 'world', 'Base_Link'],
        output='screen'
    )

    # RViz 
    rviz_config_file = os.path.join(pkg_allex_description, 'rviz', 'default.rviz')  
    rviz_node = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        output='log',
        arguments=['-d', rviz_config_file]
    )

    return LaunchDescription([
        use_gui_arg,
        world_to_base_transform,
        robot_state_publisher_node,
        joint_state_publisher_gui_node,
        rviz_node,
    ])