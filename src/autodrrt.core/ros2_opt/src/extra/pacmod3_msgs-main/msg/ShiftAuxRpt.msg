# Auxiliary feedback data related to shifting system
std_msgs/Header header

bool between_gears                 # True if gear is in "between gears" state
bool stay_in_neutral_mode          # True if ON
bool brake_interlock_active        # True if interlock is active
bool speed_interlock_active        # True if interlock is active
bool write_to_config               # True if CONFIG.TXT is being updated
bool between_gears_avail           # True if this information is available on the platform.
                                   # Does not change during operation.
bool stay_in_neutral_mode_avail    # True if this information is available on the platform.
                                   # Does not change during operation.
bool brake_interlock_active_avail  # True if this information is available on the platform.
                                   # Does not change during operation.
bool speed_interlock_active_avail  # True if this information is available on the platform.
                                   # Does not change during operation.
bool write_to_config_is_valid      # True if write_to_config information is valid
bool gear_number_avail             # True if this information is available on the platform.
                                   # Does not change during operation.
int8 gear_number                   # Current gear number as reported in the range [-10, 18]
