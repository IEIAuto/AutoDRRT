# Door status as reported by vehicle.

std_msgs/Header header

bool driver_door_open               # True if the door is open
bool passenger_door_open            # True if the door is open
bool rear_driver_door_open          # True if the door is open
bool rear_passenger_door_open       # True if the door is open
bool hood_open                      # True if open
bool trunk_open                     # True if open
bool fuel_door_open                 # True if open
bool driver_door_open_avail         # True if this information is available on the platform.
                                    # Does not change during operation.
bool passenger_door_open_avail      # True if this information is available on the platform.
                                    # Does not change during operation.
bool rear_driver_door_open_avail    # True if this information is available on the platform.
                                    # Does not change during operation.
bool rear_passenger_door_open_avail # True if this information is available on the platform.
                                    # Does not change during operation.
bool hood_open_avail                # True if this information is available on the platform.
                                    # Does not change during operation.
bool trunk_open_avail               # True if this information is available on the platform.
                                    # Does not change during operation.
bool fuel_door_open_avail           # True if this information is available on the platform.
                                    # Does not change during operation.
