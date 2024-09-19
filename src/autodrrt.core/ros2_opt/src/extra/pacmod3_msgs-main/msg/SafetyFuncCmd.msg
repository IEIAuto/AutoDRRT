# User command to set the safety function for the platform

# Safety Function Command Constants
uint16 CMD_NONE = 0
uint16 CMD_AUTO_READY = 1
uint16 CMD_AUTO = 2
uint16 CMD_MANUAL_READY = 3
uint16 CMD_CRITICAL_STOP1 = 4
uint16 CMD_CRITICAL_STOP2 = 5

std_msgs/Header header

uint16 command    # Set the commands to respective system constants defined above.
