# Component level report

# Component types in PACMod system
uint8 COMPONENT_TYPE_PACMOD = 0
uint8 COMPONENT_TYPE_PACMINI = 1
uint8 COMPONENT_TYPE_PACMICRO = 2

std_msgs/Header header

uint8 component_type  # Component type generating the report. Values defined above.

# For following fields, value is true if the system is present on the component
bool accel
bool brake
bool cruise_control_buttons
bool dash_controls_left
bool dash_controls_right
bool hazard_lights
bool headlight
bool horn
bool media_controls
bool parking_brake
bool shift
bool sprayer
bool steering
bool turn
bool wiper
bool watchdog
bool brake_deccel
bool rear_pass_door
bool engine_brake
bool marker_lamp
bool cabin_climate
bool cabin_fan_speed
bool cabin_temp

uint8 counter                      # Message transmission counter which increases by 1 with every
                                   # subsequent transmission up to and including 0xF. After this value
                                   # is reached, the pattern restarts from 0.
uint8 complement                   # Complement of counter. For example, if counter is 0x01, then the
                                   # complement is 0x10.

bool config_fault                  # True if the CONFIG.TXT file is absent or is read incorrectly.
bool can_timeout_fault             # True if a timeout has occurred in any of the connected CAN interfaces.
bool internal_supply_voltage_fault # True if any fault has occurred in power supply of the component.

bool supervisory_timeout           # True if timeout occurred on SUPERVISORY_CTRL message when supervisory enable is required
bool supervisory_sanity_fault      # True if a fault occurred in SUPERVISORY_CTRL message by customer ECU.
bool watchdog_sanity_fault         # True if a counter fault has occurred in COMPONENT_RPT of watchdog
bool watchdog_system_present_fault # True if WATCHDOG system present signal in the component report received from the
                                   # watchdog component is FALSE or any other system present signal is TRUE
