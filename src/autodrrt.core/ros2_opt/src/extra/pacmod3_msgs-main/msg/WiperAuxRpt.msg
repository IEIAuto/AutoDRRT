# Auxiliary wiper information as reported by vehicle

std_msgs/Header header

bool front_wiping               # True if wiping
bool front_spraying             # True if spraying
bool rear_wiping                # True if wiping
bool rear_spraying              # True if spraying
bool spray_near_empty           # True if near empty
bool spray_empty                # True if empty
bool front_wiping_avail         # True if this information is available on the platform.
                                # Does not change during operation.
bool front_spraying_avail       # True if this information is available on the platform.
                                # Does not change during operation.
bool rear_wiping_avail          # True if this information is available on the platform.
                                # Does not change during operation.
bool rear_spraying_avail        # True if this information is available on the platform.
                                # Does not change during operation.
bool spray_near_empty_avail     # True if this information is available on the platform.
                                # Does not change during operation.
bool spray_empty_avail          # True if this information is available on the platform.
                                # Does not change during operation.
