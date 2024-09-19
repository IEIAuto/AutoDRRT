# User command to control cabin climate system

std_msgs/Header header

bool enable                   # Set true to enable the respective system
bool ignore_overrides         # Set true for respective system to ignore overrides
bool clear_override           # Set true to clear a system override flag
uint16 cmd_ac_off_on          # Set true to set AC ON
uint16 cmd_max_ac_off_on      # Set true to set AC max ON
uint16 cmd_defrost_off_on     # Set true to set defrost ON
uint16 cmd_max_defrost_off_on # Set true to set defrost max ON
uint16 cmd_dir_up_off_on      # Set true to set direction up
uint16 cmd_dir_down_off_on    # Set true to set direction down
