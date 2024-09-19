# Parking brake status feedback

# Parking Brake Status Constants
uint16 RELEASED = 0
uint16 APPLIED = 1
uint16 BETWEEN = 2

std_msgs/Header header

uint8 parking_brake_status         # Values defined above
bool parking_brake_status_avail    # True if status is available on the platform.
                                   # Does not change during operation.
