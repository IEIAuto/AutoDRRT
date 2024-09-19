# Safety system feedback

# Safety Function Command Constants
uint16 CMD_NONE = 0
uint16 CMD_AUTO_READY = 1
uint16 CMD_AUTO = 2
uint16 CMD_MANUAL_READY = 3
uint16 CMD_CRITICAL_STOP1 = 4
uint16 CMD_CRITICAL_STOP2 = 5

# Safety Function State Constants
uint16 STATE_MANUAL_BRAKED = 0
uint16 STATE_MANUAL_UNBRAKED = 1
uint16 STATE_AUTO_READY = 2
uint16 STATE_AUTO_INACTIVE = 3
uint16 STATE_AUTO_ACTIVE_BRAKED = 4
uint16 STATE_AUTO_ACTIVE_UNBRAKED = 5
uint16 STATE_MANUAL_READY = 6
uint16 STATE_CRITICAL_STOP1 = 7
uint16 STATE_CRITICAL_STOP2 = 8
uint16 STATE_STARTUP = 9

# Auto Manual Operator Control Constants
uint16 AUTO_MAN_INVALID = 0
uint16 AUTO_MAN_MANUAL = 1
uint16 AUTO_MAN_AUTO = 2

# Cabin Safety Brake Operator Control Constants
uint16 CABIN_BRAKE_INVALID = 0
uint16 CABIN_BRAKE_APPLIED = 1
uint16 CABIN_BRAKE_UNAPPLIED = 2

# Remote Stop Status Constants
uint16 REMOTE_STOP_STATE_INVALID = 0
uint16 REMOTE_STOP_STATE_GO = 1
uint16 REMOTE_STOP_STATE_STOP = 2

# User PC, PACMod, Vehicle Fault Constants
uint16 OKAY = 0
uint16 FAULT = 1
uint16 TIMEOUT = 2

std_msgs/Header header

uint16 commanded_val                   # Echoed command. Values defined above in Safety Function Command Constants
uint16 state                           # Current state of safety system. Values defined above in Safety Function State Constants
uint16 automanual_opctrl               # Values defined above in Auto Manual Operator Control Constants
uint16 cabin_safety_brake_opctrl       # Values defined above in Cabin E-Brake Operator Control Constants
uint16 remote_stop_status              # Values defined above in Remote Stop Status Constants
bool engine_status                     # True if running
bool pacmod_system_status              # True if enabled
uint16 user_pc_fault                   # Values defined above in User PC, PACMod, Vehicle Fault Constants
uint16 pacmod_system_fault             # Values defined above in User PC, PACMod, Vehicle Fault Constants

bool manual_state_obtainable           # True if obtainable
bool auto_ready_state_obtainable       # True if obtainable
bool auto_state_obtainable             # True if obtainable
bool manual_ready_state_obtainable     # True if obtainable
bool critical_stop1_state_obtainable   # True if obtainable
bool critical_stop2_state_obtainable   # True if obtainable
