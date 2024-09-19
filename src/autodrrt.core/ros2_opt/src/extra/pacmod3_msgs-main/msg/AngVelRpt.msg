# Angular velocity feedback as reported by vehicle.

std_msgs/Header header

bool pitch_new_data_rx  # True if new data is received
bool roll_new_data_rx   # True if new data is received
bool yaw_new_data_rx    # True if new data is received
bool pitch_valid        # True if pitch information is valid
bool roll_valid         # True if roll information is valid
bool yaw_valid          # True if yaw information is valid
float64 pitch_vel       # Pitch rate value in rad/s
float64 roll_vel        # Roll rate value in rad/s
float64 yaw_vel         # Yaw rate value in rad/s
