# AutoDRRT tools Usage Instructions

A variety of development tools have been integrated to improve the efficiency of autonomous driving development. The basic functions of each tool are introduced below.
> The working directory is `src/autodrrt.tools`, different folders correspond to different tools.

### annotation
-  auto_annotation
Automatically annotate camera data based on LiDAR inference results.

-  manaul_annotation
Based on lidar data and camera data, the data is manually annotated.


### calibration
-  camera_calibration
Provide camera internal calibration function.

-  direct_visual_lidar_calibration
Provide joint calibration of camera and lidar external parameters.
-  multi_lidar_calibration
Provides joint calibration function for multiple LiDARs.


### map_format_conversion
Provides map format conversion function, converting opendrive format to lanelet2.

### time_analysis
Provides end-to-end latency measurement for autodrrt pipelines.\

### simulation
Provide a user guide for the simulation emulator