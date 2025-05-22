source install/setup.bash

echo "start camera 0"
ros2 launch v4l2_camera v4l2_camera.launch_dev0.py &
sleep 10

echo "start camera 1"
ros2 launch v4l2_camera v4l2_camera.launch_dev1.py &
sleep 10

echo "start camera 2"
ros2 launch v4l2_camera v4l2_camera.launch_dev2.py &
sleep 10

echo "start camera 3"
ros2 launch v4l2_camera v4l2_camera.launch_dev3.py &
sleep 10

echo "start camera 4"
ros2 launch v4l2_camera v4l2_camera.launch_dev4.py &
sleep 10

echo "start camera 5"
ros2 launch v4l2_camera v4l2_camera.launch_dev5.py &