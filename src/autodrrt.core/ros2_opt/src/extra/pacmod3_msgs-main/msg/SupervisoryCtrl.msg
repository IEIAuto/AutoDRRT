# User command for platforms supporting Supervisory Enablement Rules

std_msgs/Header header

bool enable          # Set true to allow systems to enable
uint8 counter        # Message transmission counter which increases by 1 with every
                     # subsequent transmission up to and including 0xF. After this value
                     # is reached, the pattern restarts from 0.
uint8 complement     # Complement of counter. For example, if counter is 0x01, then the
                     # complement is 0x10.
