# Engine feedback report as reported by vehicle

std_msgs/Header header

float64 engine_speed                # Measured speed in rpm as reported by vehicle
float64 engine_torque               # Measured engine torque in N-m as reported by vehicle
int16 engine_coolant_temp           # Measured coolant temperature in deg C in the range [-40, 210] as reported by vehicle
bool engine_speed_avail             # True if this information is available on the platform.
                                    # Does not change during operation.
bool engine_torque_avail            # True if this information is available on the platform.
                                    # Does not change during operation.
bool engine_coolant_temp_avail      # True if this information is available on the platform.
                                    # Does not change during operation.
bool fuel_level_avail               # True if this information is available on the platform.
                                    # Does not change during operation.
float64 fuel_level                  # Measured fuel level in % as reported by vehicle
