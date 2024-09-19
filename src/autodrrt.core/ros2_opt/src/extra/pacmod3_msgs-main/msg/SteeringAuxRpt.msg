# Auxiliary feedback data related to steering system.

std_msgs/Header header

float64 steering_torque                # Measured torque in N-m as reported by vehicle
float64 rotation_rate                  # Measured rotation rate in rad/s as reported by vehicle
bool operator_interaction              # True if driver physically interacted with related vehicle control
bool rotation_rate_sign                # True if sign is negative i.e. rotation is clockwise
bool vehicle_angle_calib_status        # True if steering angle is calibrated on Vehicle ECU
bool steering_limiting_active          # True if limiting function is reducing an excessive command
bool steering_torque_avail             # True if this information is available on the platform.
                                       # Does not change during operation.
bool rotation_rate_avail               # True if this information is available on the platform.
                                       # Does not change during operation.
bool operator_interaction_avail        # True if this information is available on the platform.
                                       # Does not change during operation.
bool rotation_rate_sign_avail          # True if this information is available on the platform.
                                       # Does not change during operation.
bool vehicle_angle_calib_status_avail  # True if this information is available on the platform.
                                       # Does not change during operation.
bool steering_limiting_active_avail    # True if steering limiting is available on the platform.
                                       # Does not change during operation.
