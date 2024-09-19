# This message relates only to FMCW radar.  
# All variables below are relative to the radar's frame of reference.
# This message is not meant to be used alone but as part of a stamped or array message.

# Object classifications (Additional vendor-specific classifications are permitted starting from 32000 eg. Car)

uint16 detection_id                       # Index of each radar detection point
geometry_msgs/Point position              # x, y, z position of each radar detection point

float32 azimuth							  # azimuth angle of each radar detection point in Degree
float32 rangerate						  # relative velocity of the radar detected target w.r.t radial direction
float32 amplitude						  # amplitude of the reflected signal of the radar detected target(rcs)