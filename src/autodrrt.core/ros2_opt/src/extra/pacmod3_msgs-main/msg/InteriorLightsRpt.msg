# Interior lights status as reported by vehicle

# Interior lights dim level constants
uint8 DIM_LEVEL_MIN = 0
uint8 DIM_LEVEL_1 = 1
uint8 DIM_LEVEL_2 = 2
uint8 DIM_LEVEL_3 = 3
uint8 DIM_LEVEL_4 = 4
uint8 DIM_LEVEL_5 = 5
uint8 DIM_LEVEL_6 = 6
uint8 DIM_LEVEL_7 = 7
uint8 DIM_LEVEL_8 = 8
uint8 DIM_LEVEL_9 = 9
uint8 DIM_LEVEL_10 = 10
uint8 DIM_LEVEL_11 = 11
uint8 DIM_LEVEL_MAX = 12
uint8 DIM_LEVEL_12 = 13
uint8 DIM_LEVEL_13 = 14
uint8 DIM_LEVEL_14 = 15
uint8 DIM_LEVEL_15 = 16
uint8 DIM_LEVEL_16 = 17
uint8 DIM_LEVEL_17 = 18
uint8 DIM_LEVEL_18 = 19
uint8 DIM_LEVEL_19 = 20
uint8 DIM_LEVEL_20 = 21
uint8 DIM_LEVEL_21 = 22
uint8 DIM_LEVEL_22 = 23
uint8 DIM_LEVEL_23 = 24

std_msgs/Header header

bool front_dome_lights_on           # True if ON
bool rear_dome_lights_on            # True if ON
bool mood_lights_on                 # True if ON
bool ambient_light_sensor           # True if bright
uint8 dim_level                     # Dim level status as reported. Values defined above.
bool front_dome_lights_on_avail     # True if this information is available on the platform.
                                    # Does not change during operation.
bool rear_dome_lights_on_avail      # True if this information is available on the platform.
                                    # Does not change during operation.
bool mood_lights_on_avail           # True if this information is available on the platform.
                                    # Does not change during operation.
bool dim_level_avail                # True if this information is available on the platform.
                                    # Does not change during operation.
bool ambient_light_sensor_avail     # True if this information is available on the platform.
                                    # Does not change during operation.
