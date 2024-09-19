# User command to control float systems
# Set command in % for accel and brake system.

std_msgs/Header header

bool enable             # Set true to enable the respective system
bool ignore_overrides   # Set true for respective system to ignore overrides
bool clear_override     # Set true to clear a system override flag

float64 command         # Set command in % for respective system
