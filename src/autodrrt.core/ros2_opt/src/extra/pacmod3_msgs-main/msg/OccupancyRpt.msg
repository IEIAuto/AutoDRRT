# Seat and seatbelt status as reported by vehicle.

std_msgs/Header header

bool driver_seat_occupied                # True if occupied
bool passenger_seat_occupied             # True if occupied
bool rear_seat_occupied                  # True if occupied
bool driver_seatbelt_buckled             # True if buckled
bool passenger_seatbelt_buckled          # True if buckled
bool driver_rear_seatbelt_buckled        # True if buckled
bool pass_rear_seatbelt_buckled          # True if buckled
bool center_rear_seatbelt_buckled        # True if buckled
bool driver_seat_occupied_avail          # True if this information is available on the platform.
                                         # Does not change during operation.
bool passenger_seat_occupied_avail       # True if this information is available on the platform.
                                         # Does not change during operation.
bool rear_seat_occupied_avail            # True if this information is available on the platform.
                                         # Does not change during operation.
bool driver_seatbelt_buckled_avail       # True if this information is available on the platform.
                                         # Does not change during operation.
bool passenger_seatbelt_buckled_avail    # True if this information is available on the platform.
                                         # Does not change during operation.
bool driver_rear_seatbelt_buckled_avail  # True if this information is available on the platform.
                                         # Does not change during operation.
bool pass_rear_seatbelt_buckled_avail    # True if this information is available on the platform.
                                         # Does not change during operation.
bool center_rear_seatbelt_buckled_avail  # True if this information is available on the platform.
                                         # Does not change during operation.
