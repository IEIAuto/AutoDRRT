# User command to control int systems

# Cabin Fan Speed Constants
uint16 FAN_OFF = 0
uint16 FAN_SPEED_1 = 1
uint16 FAN_SPEED_2 = 2
uint16 FAN_SPEED_3 = 3
uint16 FAN_SPEED_4 = 4
uint16 FAN_SPEED_5 = 5
uint16 FAN_SPEED_6 = 6
uint16 FAN_SPEED_7 = 7
uint16 FAN_SPEED_8 = 8
uint16 FAN_SPEED_9 = 9
uint16 FAN_SPEED_10 = 10

# Cruise Control Button Command Constants
uint16 CRUISE_CONTROL_NONE = 0
uint16 CRUISE_CONTROL_CNCL = 1
uint16 CRUISE_CONTROL_ACC_FURTHER = 2
uint16 CRUISE_CONTROL_ACC_CLOSER = 3
uint16 CRUISE_CONTROL_SET_DEC = 4
uint16 CRUISE_CONTROL_RES_INC = 5
uint16 CRUISE_CONTROL_ON_OFF = 6

# Dash Control Left/Right Command Constants
uint16 DASH_CONTROL_NONE = 0
uint16 DASH_CONTROL_OK = 1
uint16 DASH_CONTROL_LEFT = 2
uint16 DASH_CONTROL_RIGHT = 3
uint16 DASH_CONTROL_UP = 4
uint16 DASH_CONTROL_DOWN = 5

# Engine Brake Command Constants
uint16 ENGINE_BRAKE_OFF = 0
uint16 ENGINE_BRAKE_LOW = 1
uint16 ENGINE_BRAKE_MEDIUM = 2
uint16 ENGINE_BRAKE_HIGH = 3

# Headlight Command Constants
uint16 HEADLIGHTS_OFF = 0
uint16 HEADLIGHTS_LOW = 1
uint16 HEADLIGHTS_HIGH = 2

# Media Control Command Constants
uint16 MEDIA_CONTROL_NONE = 0
uint16 MEDIA_CONTROL_VOICE_COMMAND = 1
uint16 MEDIA_CONTROL_MUTE = 2
uint16 MEDIA_CONTROL_PREV_TRACK_ANSWER = 3
uint16 MEDIA_CONTROL_NEXT_TRACK_HANG_UP = 4
uint16 MEDIA_CONTROL_VOL_UP = 5
uint16 MEDIA_CONTROL_VOL_DOWN = 6

# Rear Pass Door Command Constants
uint16 DOOR_NEUTRAL = 0
uint16 DOOR_OPEN = 1
uint16 DOOR_CLOSE = 2

# Shift Command Constants
uint16 SHIFT_PARK = 0
uint16 SHIFT_REVERSE = 1
uint16 SHIFT_NEUTRAL = 2
uint16 SHIFT_HIGH = 3
uint16 SHIFT_FORWARD = 3
uint16 SHIFT_LOW = 4
uint16 SHIFT_NONE = 7

# Turn Command Constants
uint16 TURN_RIGHT = 0
uint16 TURN_NONE = 1
uint16 TURN_LEFT = 2
uint16 TURN_HAZARDS = 3

# Wiper Command Constants
uint16 WIPERS_OFF = 0
uint16 WIPERS_INT_1 = 1
uint16 WIPERS_INT_2 = 2
uint16 WIPERS_INT_3 = 3
uint16 WIPERS_INT_4 = 4
uint16 WIPERS_INT_5 = 5
uint16 WIPERS_INT_6 = 6
uint16 WIPERS_INT_7 = 7
uint16 WIPERS_INT_8 = 8
uint16 WIPERS_INT_9 = 9
uint16 WIPERS_INT_10 = 10
uint16 WIPERS_LOW = 253
uint16 WIPERS_MEDIUM = 254
uint16 WIPERS_HIGH = 255


std_msgs/Header header

bool enable             # Set true to enable the respective system
bool ignore_overrides   # Set true for respective system to ignore overrides
bool clear_override     # Set true to clear a system override flag

uint16 command          # Set the commands to respective system constants defined above.

