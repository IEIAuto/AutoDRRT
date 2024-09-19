# PACMod system watchdog report echoing
# states and faults of all present systems
# and monitors other faults.

std_msgs/Header header

bool global_enabled_flag                 # Echo of enabled bit in GLOBAL_RPT. True if enabled.
bool global_override_active              # Echo of override_active bit in GLOBAL_RPT. True if overridden.
bool global_command_timeout_error        # True if timeout occurred on command when enabled.
bool global_pacmod_subsystem_timeout     # Echo of subsystem_can_timeout bit in GLOBAL_RPT. True if timeout occurred.
bool global_vehicle_can_timeout          # True if a timeout has occurred on a vehicle CAN interface
bool global_pacmod_system_fault_active   # True if a fault is active on any system
bool global_config_fault_active          # True if a config fault is active on any pacmod system
bool global_timeout                      # True if timeout occurred on GLOBAL_RPT

bool accel_enabled                       # Echo of enabled bit in ACCEL_RPT. True if enabled.
bool accel_override_active               # Echo of override_active bit in ACCEL_RPT. True if overridden.
bool accel_command_output_fault          # Echo of command_output_fault bit in ACCEL_RPT. True if fault.
bool accel_input_output_fault            # Echo of input_output_fault bit in ACCEL_RPT. True if fault.
bool accel_output_reported_fault         # Echo of output_reported_fault bit in ACCEL_RPT. True if fault.
bool accel_pacmod_fault                  # Echo of pacmod_fault bit in ACCEL_RPT. True if fault.
bool accel_vehicle_fault                 # Echo of vehicle_fault bit in ACCEL_RPT. True if fault.
bool accel_timeout                       # True if timeout occurred on ACCEL_RPT

bool brake_enabled                       # Echo of enabled bit in BRAKE_RPT. True if enabled.
bool brake_override_active               # Echo of override_active bit in BRAKE_RPT. True if overridden.
bool brake_command_output_fault          # Echo of command_output_fault bit in BRAKE_RPT. True if fault.
bool brake_input_output_fault            # Echo of input_output_fault bit in BRAKE_RPT. True if fault.
bool brake_output_reported_fault         # Echo of output_reported_fault bit in BRAKE_RPT. True if fault.
bool brake_pacmod_fault                  # Echo of pacmod_fault bit in BRAKE_RPT. True if fault.
bool brake_vehicle_fault                 # Echo of vehicle_fault bit in BRAKE_RPT. True if fault.
bool brake_timeout                       # True if timeout occurred on BRAKE_RPT

bool shift_enabled                       # Echo of enabled bit in SHIFT_RPT. True if enabled.
bool shift_override_active               # Echo of override_active bit in SHIFT_RPT. True if overridden.
bool shift_command_output_fault          # Echo of command_output_fault bit in SHIFT_RPT. True if fault.
bool shift_input_output_fault            # Echo of input_output_fault bit in SHIFT_RPT. True if fault.
bool shift_output_reported_fault         # Echo of output_reported_fault bit in SHIFT_RPT. True if fault.
bool shift_pacmod_fault                  # Echo of pacmod_fault bit in SHIFT_RPT. True if fault.
bool shift_vehicle_fault                 # Echo of vehicle_fault bit in SHIFT_RPT. True if fault.
bool shift_timeout                       # True if timeout occurred on SHIFT_RPT

bool steer_enabled                       # Echo of enabled bit in STEER_RPT. True if enabled.
bool steer_override_active               # Echo of override_active bit in STEER_RPT. True if overridden.
bool steer_command_output_fault          # Echo of command_output_fault bit in STEER_RPT. True if fault.
bool steer_input_output_fault            # Echo of input_output_fault bit in STEER_RPT. True if fault.
bool steer_output_reported_fault         # Echo of output_reported_fault bit in STEER_RPT. True if fault.
bool steer_pacmod_fault                  # Echo of pacmod_fault bit in STEER_RPT. True if fault.
bool steer_vehicle_fault                 # Echo of vehicle_fault bit in STEER_RPT. True if fault.
bool steer_timeout                       # True if timeout occurred on STEER_RPT

bool mod1_config_fault                   # Echo of config_fault bit in COMPONENT_RPT of PACMod1. True if fault.
bool mod1_can_timeout                    # Echo of can_timeout bit in COMPONENT_RPT of PACMod1. True if fault.
bool mod1_counter_fault                  # True if counter fault has occurred on COMPONENT_RPT of PACMod1

bool mod2_config_fault                   # Echo of config_fault bit in COMPONENT_RPT of PACMod2. True if fault.
bool mod2_can_timeout                    # Echo of can_timeout bit in COMPONENT_RPT of PACMod2. True if fault.
bool mod2_counter_fault                  # True if counter fault has occurred on component report of PACMod2

bool mod3_config_fault                   # Echo of config_fault bit in COMPONENT_RPT of PACMod3. True if fault.
bool mod3_can_timeout                    # Echo of can_timeout bit in COMPONENT_RPT of PACMod3. True if fault.
bool mod3_counter_fault                  # True if counter fault has occurred on component report of PACMod3

bool mini1_rpt_timeout                   # True if timeout occurred on component report of PACMini1
bool mini1_config_fault                  # Echo of config_fault bit in COMPONENT_RPT of PACMini1. True if fault.
bool mini1_can_timeout                   # Echo of can_timeout bit in COMPONENT_RPT of PACMini1. True if fault.
bool mini1_counter_fault                 # True if counter fault has occurred on component report of PACMini1

bool mini2_rpt_timeout                   # True if timeout occurred on component report of PACMini2
bool mini2_config_fault                  # Echo of config_fault bit in COMPONENT_RPT of PACMini2. True if fault.
bool mini2_can_timeout                   # Echo of can_timeout bit in COMPONENT_RPT of PACMini2. True if fault.
bool mini2_counter_fault                 # True if counter fault has occurred on component report of PACMini2

bool mini3_rpt_timeout                   # True if timeout occurred on component report of PACMini3
bool mini3_config_fault                  # Echo of config_fault bit in COMPONENT_RPT of PACMini3. True if fault.
bool mini3_can_timeout                   # Echo of can_timeout bit in COMPONENT_RPT of PACMini3. True if fault.
bool mini3_counter_fault                 # True if counter fault has occurred on component report of PACMini3

bool mod_system_present_fault            # True if mismatch has occurred between platform specs and systems on PACMod
bool mini_system_present_fault           # True if mismatch has occurred between platform specs and systems on PACMini

bool global_internal_power_supply_fault  # True if any component report has an internal_supply_voltage_fault true.
