# Auxiliary feedback data related to headlight system

# Headlight System Mode Constants
uint8 HEADLIGHTS_OFF = 0
uint8 PARKING_LIGHTS_ONLY = 1
uint8 HEADLIGHTS_ON_MANUAL_MODE = 2
uint8 HEADLIGHTS_ON_AUTO_MODE = 3

std_msgs/Header header

bool headlights_on                  # True if ON
bool headlights_on_bright           # True if ON
bool fog_lights_on                  # True if ON
uint8 headlights_mode               # Headlight system mode status as reported by vehicle. Values defined above.
bool headlights_on_avail            # True if this information is available on the platform.
                                    # Does not change during operation.
bool headlights_on_bright_avail     # True if this information is available on the platform.
                                    # Does not change during operation.
bool fog_lights_on_avail            # True if this information is available on the platform.
                                    # Does not change during operation.
bool headlights_mode_avail          # True if this information is available on the platform.
                                    # Does not change during operation.
