# Critical Stop Report from Safety Function System

std_msgs/Header header

bool automaual_opctrl_fault          # True if fault is active
bool remote_stop_fault               # True if fault is active
bool safety_brake_opctrl_off         # True if fault is active
bool safety_brake_cmd_timeout        # True if fault is active
bool safety_func_cmd_timeout         # True if fault is active
bool safety_func_critical_stop_1_cmd # True if fault is active
bool safety_func_critical_stop_2_cmd # True if fault is active
bool safety_func_none_cmd            # True if fault is active
bool pacmod_system_timeout           # True if fault is active
bool pacmod_system_fault             # True if fault is active
bool pacmod_system_not_active        # True if fault is active
bool vehicle_report_timeout          # True if fault is active
bool vehicle_report_fault            # True if fault is active
bool low_engine_rpm                  # True if fault is active
bool pri_safety_brake_signal_1_fault # True if fault is active
bool pri_safety_brake_signal_2_fault # True if fault is active
bool sec_safety_brake_signal_1_fault # True if fault is active
bool sec_safety_brake_signal_2_fault # True if fault is active
bool primary_processor_fault         # True if fault is active
bool secondary_processor_fault       # True if fault is active
bool remote_stop_cmd                 # True if fault is active
bool pri_safety_brake_pressure_fault # True if fault is active
