# User command to control bool systems

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

bool enable             # Set true to enable the respective system
bool ignore_overrides   # Set true for respective system to ignore overrides
bool clear_override     # Set true to clear a system override flag

bool command            # Set the commands to respective system constants defined above.
