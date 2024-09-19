# Feedback report for float systems
# For accel and brake system, unit used is "%".
# For steering system, unit used is "rad" in the range defined in platform user manual.
# For brake deccel system, unit used is "m/s^2" in the range [0,10.0]
# For cabin temperature system, unit used is deg_c in the range [10.0,35.0]

std_msgs/Header header

bool enabled                  # True if system is enabled
bool override_active          # True if override has been triggered
bool command_output_fault     # True if fault is active
bool input_output_fault       # True if fault is active
bool output_reported_fault    # True if fault is active
bool pacmod_fault             # True if fault is active
bool vehicle_fault            # True if fault is active
bool command_timeout          # True if timeout has occurred on command message while enabled
float64 manual_input          # Measured value on vehicle stalk/sensor in unit defined according to system
float64 command               # Echoed user command value in unit defined according to system
float64 output                # Measured value of PACMod output in unit defined according to system
