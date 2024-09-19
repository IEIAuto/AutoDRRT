# Linear acceleration feedback as reported by vehicle

std_msgs/Header header

bool lateral_new_data_rx       # True if new data is received
bool longitudinal_new_data_rx  # True if new data is received
bool vertical_new_data_rx      # True if new data is received
bool lateral_valid             # True if lateral information is valid
bool longitudinal_valid        # True if longitudinal information is valid
bool vertical_valid            # True if vertical information is valid
float64 lateral_accel          # Lateral acceleration in m/s^2 as reported by vehicle
float64 longitudinal_accel     # Longitudinal acceleration in m/s^2 as reported by vehicle
float64 vertical_accel         # Vertical acceleration in m/s^2 as reported by vehicle
