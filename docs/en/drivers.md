# Sensor Drivers Usage Instructions

Supports different sensor drivers, including camera, lidar, radar, IMU, serial port, CAN.
> The working directory is `src/autodrrt.driver`, different folders correspond to different sensor drivers

### Camera
Supports two cameras: 
-  sensing:(v4l2_camera) output format YUV. [product details](https://www.sensing-world.com/productinfo/2049260.html?fromMid=635#recommendFromPid=0)
-  leopard:(isaac_camera) output format RAW. [product details](https://leopardimaging.com/product/automotive-cameras/cameras-by-interface/maxim-gmsl-2-cameras/li-ar0233-gmsl2/li-ar0233-gmsl2-030h-2/).

### Lidar
Supports robosense.

### IMU
Supports two imu: cgi430 and cgi610.

### CAN
Supports socket can.

### radar
Supports ARS408.

### serial
Serial to ros2 interface.

### Nodev2x
Support v2x vehicle-road collaboration plugin.
