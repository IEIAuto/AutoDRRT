# User command to control steering system

std_msgs/Header header

bool enable             # Set true to enable steering system
bool ignore_overrides   # Set true for steering system to ignore steering overrides
bool clear_override     # Set true to clear steering override flag

float64 command         # Set steering command in rad in the range specified in user manual
float64 rotation_rate   # Set rotation rate in rad/sec in the range specified in user manual
