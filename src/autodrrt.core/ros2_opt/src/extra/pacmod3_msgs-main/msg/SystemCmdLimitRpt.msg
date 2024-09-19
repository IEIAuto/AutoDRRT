# Feedback report of set command limits for accel and brake systems

std_msgs/Header header

float64 sys_cmd_limit   # Value of limit imposed on the command for the respective system in %
float64 limited_sys_cmd # Value of command for the respective system constrained by sys_cmd_limit in %
