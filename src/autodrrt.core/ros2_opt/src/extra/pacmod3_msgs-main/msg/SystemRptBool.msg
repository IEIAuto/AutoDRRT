# Feedback report for bool systems

# Hazard Light Command constants
bool HAZARDLIGHTS_OFF = 0
bool HAZARDLIGHTS_ON = 1

# Horn Command Constants
bool HORN_OFF = 0
bool HORN_ON = 1

# Marker Lamp Command Constants
bool MARKERLAMP_OFF = 0
bool MARKERLAMP_ON = 1

# Parking Brake Command Constants
bool PARKING_BRAKE_OFF = 0
bool PARKING_BRAKE_ON = 1

# Sprayer Command Constants
bool SPRAY_OFF = 0
bool SPRAY_ON = 1

std_msgs/Header header

bool enabled                  # True if system is enabled
bool override_active          # True if override has been triggered
bool command_output_fault     # True if fault is active
bool input_output_fault       # True if fault is active
bool output_reported_fault    # True if fault is active
bool pacmod_fault             # True if fault is active
bool vehicle_fault            # True if fault is active
bool command_timeout          # True if timeout has occurred on command message while enabled
bool manual_input             # Measured value on vehicle stalk/sensor. Values defined above
bool command                  # Echoed user command value. Values defined above
bool output                   # Measured value of PACMod output. Values defined above
