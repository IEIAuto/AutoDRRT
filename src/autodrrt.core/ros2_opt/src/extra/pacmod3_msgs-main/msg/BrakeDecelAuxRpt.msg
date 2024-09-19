# Auxiliary feedback data related to brake deccel system

# XBR Active Control Mode Constants
uint16 NO_BRAKE_DEMAND = 0
uint16 DRIVERS_BRAKE_DEMAND = 1
uint16 ADDITION_MODE_OF_XBR_ACCELERATION_CONTROL = 2
uint16 MAXIMUM_MODE_OF_XBR_ACCELERATION_CONTROL = 3

# XBR System State Constants
uint16 ANY_EXTERNAL_BRAKE_DEMAND_WILL_BE_ACCEPTED = 0
uint16 NO_EXTERNAL_BRAKE_DEMAND_WILL_BE_ACCEPTED = 2

# Foundation Brake Use Constants
uint16 FOUNDATION_BRAKES_NOT_IN_USE = 0
uint16 FOUNDATION_BRAKES_IN_USE = 1

# Hill Holder Mode Constants
uint16 INACTIVE = 0
uint16 ACTIVE = 1
uint16 ACTIVE_BUT_INACTIVE_SOON = 2
uint16 ERROR = 6

std_msgs/Header header

uint16 xbr_active_control_mode     # Status as reported. Values defined above.
uint16 xbr_system_state            # Status as reported. Values defined above.
uint16 foundation_brake_use        # Status as reported. Values defined above.
uint16 hill_holder_mode            # Status as reported. Values defined above.
bool xbr_active_control_mode_avail # True if this information is available on the platform.
                                   # Does not change during operation.
bool xbr_system_state_avail        # True if this information is available on the platform.
                                   # Does not change during operation.
bool foundation_brake_use_avail    # True if this information is available on the platform.
                                   # Does not change during operation.
bool hill_holder_mode_avail        # True if this information is available on the platform.
                                   # Does not change during operation.
