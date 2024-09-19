# Feedback report of set command limits for steering system

std_msgs/Header header

float64 pos_cmd_limit             # Value of limit imposed on position command in rad
float64 limited_pos_cmd           # Value of position command constrained by pos__cmd_limit in rad
float64 rotation_rate_cmd_limit   # Value of limit imposed on rotation rate command in rad/s
float64 limited_rotation_rate_cmd # Value of rotation rate command constrained by rotatio_rate_cmd_limit in rad/s
