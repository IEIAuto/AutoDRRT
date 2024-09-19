# Projector type
# Also refer to https://github.com/autowarefoundation/autoware.universe/tree/main/map/map_projection_loader/README.md
string LOCAL = "local"
string LOCAL_CARTESIAN_UTM = "LocalCartesianUTM"
string MGRS = "MGRS"
string TRANSVERSE_MERCATOR = "TransverseMercator"
string projector_type

# Vertical datum
# Also refer to https://github.com/autowarefoundation/autoware.universe/tree/main/map/map_projection_loader/README.md
string WGS84 = "WGS84"
string EGM2008 = "EGM2008"
string vertical_datum

# Used for MGRS map
string mgrs_grid

# Used for some map projection types
# altitude may not be in ellipsoid height
geographic_msgs/GeoPoint map_origin
