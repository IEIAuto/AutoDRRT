# This message type is deprecated. However, this message type is maintained
# because autoware_hmi_msgs (https://github.com/tier4/AutowareArchitectureProposal_msgs/tree/main/autoware_hmi_msgs)
# depends on this message type.

# constants
uint32 SUCCESS=1
uint32 IGNORED=2
uint32 WARN=3
uint32 ERROR=4

# fields
uint32 code
string message
