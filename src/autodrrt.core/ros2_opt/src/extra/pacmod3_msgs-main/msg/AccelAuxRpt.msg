# Auxiliary feedback data related to accel system

std_msgs/Header header

bool operator_interaction              # True if driver physically interacted with related vehicle control
bool accel_limiting_active             # True if limiting function is reducing an excessive command
bool park_brake_interlock_active       # True if parkin brake is active and is preventing system from commanding
bool brake_interlock_active            # True if interlock is active
bool operator_interaction_avail        # True if operator interaction is available on the platform.
                                       # Does not change during operation.
bool accel_limiting_active_avail       # True if accel limiting is available on the platform.
                                       # Does not change during operation.
bool park_brake_interlock_active_avail # True if park brake interlock is available on the platform.
                                       # Does not change during operation.
bool brake_interlock_active_avail      # True if this information is available on the platform.
                                       # Does not change during operation.
