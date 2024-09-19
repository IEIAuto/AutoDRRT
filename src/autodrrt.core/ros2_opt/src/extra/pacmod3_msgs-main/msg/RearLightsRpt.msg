# Rear lights status as reported by vehicle.

std_msgs/Header header

bool brake_lights_on         # True if ON
bool reverse_lights_on       # True if ON
bool brake_lights_on_avail   # True if this information is available on the platform.
                             # Does not change during operation.
bool reverse_lights_on_avail # True if this information is available on the platform.
                             # Does not change during operation.
