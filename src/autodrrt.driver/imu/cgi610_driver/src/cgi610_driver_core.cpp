#include "cgi610_driver/cgi610_driver_core.hpp"

CGI610_Analyze::CGI610_Analyze(const std::string & node_name, const rclcpp::NodeOptions & node_options)
: rclcpp::Node(node_name, node_options)
{
    SerialPortInit();
    pub_imu_ = create_publisher<sensor_msgs::msg::Imu>("imu_data", 1);
    pub_navsatfix_ = create_publisher<sensor_msgs::msg::NavSatFix>("fix", 1);

    Monitor_thread_ = std::thread(&CGI610_Analyze::monitor_func, this);
    Monitor_thread_.detach(); 

    SerialPortAnalyze();
}

CGI610_Analyze::~CGI610_Analyze(){}

void CGI610_Analyze::SerialPortInit()
{
    port_name = declare_parameter("port_name", "/dev/ttyUSB0");
    baudrate_ = declare_parameter("baudrate", 230400);
    output_hz_ = declare_parameter("output_hz", 100);

    try
    {
        ser.setPort(port_name); //设置串口属性，并打开串口
        serial::Timeout to = serial::Timeout::simpleTimeout(500);//超时定义。单位：ms
        ser.setTimeout(to);
        ser.open();
    }
    catch(serial::IOException& e)
    {
        while(!ser.isOpen()){
            RCLCPP_ERROR_STREAM(this->get_logger(), "Unable to open port: "<<ser.getPort()<<" please check serial port, meanwhile check the permission of port and run common sudo chmod 777 port_name and try again!");
            usleep(1000000);
            try
            {
                ser.open(); 
                break;
            }
            catch(serial::IOException& e)
            {

            }
            if(!rclcpp::ok()){
                break;
            }       
        }
    }
    if(ser.isOpen()){
        ser.setBaudrate(baudrate_);
        RCLCPP_INFO(get_logger(), ">> the Serial Baudrate is set to: %d", baudrate_);
        RCLCPP_INFO(this->get_logger(), "Serial Port initialized...");
    }else{
        RCLCPP_INFO(this->get_logger(),"the port open failed!");
        rclcpp::shutdown();
    }
}

void CGI610_Analyze::SerialPortAnalyze()
{
    rclcpp::Rate loop_rate(output_hz_);    //设置循环频率为100HZ
    // ser.flushInput();                      //在开始正式接收数据前先清除串口的接收缓冲区

    while(rclcpp::ok())
    {
        const auto exe_start_time = std::chrono::system_clock::now();
        if(ser.isOpen())
        {
            if(ser.available())
            {
                int ascii_num = ser.available();
                read = ser.read(ascii_num);
                RCLCPP_DEBUG(this->get_logger(), "read %i new characters from serial port, adding to %i characters of old input.", (int)read.size(), (int)str.size());
                str.append(read);
                if(str.size()>340)
                {
                    unsigned int loc1 = str.find(foundCHC,0);
                    if (loc1 > 340)
                    {
                        RCLCPP_WARN(this->get_logger(),"read GPCHC failed!!!!!");
                        str.erase();
                        continue;
                    }
                    unsigned int loc3 = str.find(foundCHC,loc1+1);
                    if(loc3 > 340)
                    {
                        str.erase();
                        continue;
                    }
                    double gpsweek,gpstime,yaw,pitch,roll,gyro_x,gyro_y,gyro_z,acc_x,acc_y,acc_z,latitude,longitude,altitude, ve, vn, vu, velocity;
                    int nsv1, nsv2, status, age, space;
                    int i = 0;
                    std::string gpchc(str, loc1, loc3-loc1);
                    char *serial_Ctype;
                    int len = gpchc.length(); //返回字符串长度
                    serial_Ctype = (char *)malloc((len+1)*sizeof(char)); //malloc() 分配长度为len+1所占的字节数
                    gpchc.copy(serial_Ctype, len, 0);
                    serial_Ctype[len] = '\0';
                    char *ptr = strtok(serial_Ctype, ","); //strtok() 以逗号为标志，分解字符串为一组字符串，第一个参数为要分解的字符串，第二个参数为分隔符

                    while(ptr !=NULL && i < Space +1)
                    {
                        i++;
                        switch(i)
                        {
                            case GPSWeek:
                                 gpsweek = atoi(ptr);
                                 break;

                            case GPSTime:
                                 gpstime = atof(ptr);
                                 break;
                            
                            case Heading:
                                 yaw = atof(ptr);
                                 break;

                            case Pitch:
                                 pitch = atof(ptr);
                                 break;

                            case Roll:
                                 roll = atof(ptr);
                                 break;

                            case Gyro_x:
                                 gyro_x = atof(ptr);
                                 break;

                            case Gyro_y:
                                 gyro_y = atof(ptr);
                                 break;

                            case Gyro_z:
                                 gyro_z = atof(ptr);
                                 break;

                            case Acc_x:
                                 acc_x = atof(ptr);
                                 break;

                            case Acc_y:
                                 acc_y = atof(ptr);
                                 break;

                            case Acc_z:
                                 acc_z = atof(ptr);
                                 break;

                            case Latitude:
                                 latitude = atof(ptr);
                                 break;

                            case Longitude:
                                 longitude = atof(ptr);
                                 break;

                            case Altitude:
                                 altitude = atof(ptr);
                                 break;

                            case VE:
                                 ve = atof(ptr);
                                 break;

                            case VN:
                                 vn = atof(ptr);
                                 break;

                            case VU:
                                 vu = atof(ptr);
                                 break;

                            case Velocity:
                                 velocity = atof(ptr);
                                 break;

                            case NSV1:
                                 nsv1 = atof(ptr);
                                 break;

                            case NSV2:
                                 nsv2 = atof(ptr);
                                 break;

                            case Status:
                                 status = atoi(ptr);
                                 break;
                            case Age:
                                 age = atoi(ptr);
                                 break;
                            case Space:
                                 space = atof(ptr);
                                 break;
                        }
                        ptr = strtok(NULL, ",");
                    }

                    imu_msg.header.stamp = this->now();
                    imu_msg.header.frame_id = "imu_link";
                    imu_msg.linear_acceleration.x = acc_x*9.8;
                    imu_msg.linear_acceleration.y = acc_y*9.8;
                    imu_msg.linear_acceleration.z = acc_z*9.8;
                    imu_msg.angular_velocity.x = (gyro_x/180.0)*M_PI;
                    imu_msg.angular_velocity.y = (gyro_y/180.0)*M_PI;
                    imu_msg.angular_velocity.z = (gyro_z/180.0)*M_PI;
                    pub_imu_->publish(imu_msg);

                    navsatfix_msg.header.stamp = this->now();
                    navsatfix_msg.header.frame_id = "base_lnik";
                    navsatfix_msg.latitude = latitude;
                    navsatfix_msg.longitude = longitude;
                    navsatfix_msg.altitude = altitude;

                    str.erase(0, loc3);
                }
            }
            else
            {
                RCLCPP_WARN(this->get_logger(),"The serial port data is no available!");
            }

        }
        else
        {
            try
            {
                ser.open();
            }
            catch(serial::IOException& e)
            {
                RCLCPP_ERROR_STREAM(this->get_logger()," the serial port"<<ser.getPort()<<" Disconnected, please check serial interface!");
                rclcpp::shutdown();
            }
        }
        loop_rate.sleep();
        const auto exe_end_time = std::chrono::system_clock::now();   
        const double exe_time = std::chrono::duration_cast<std::chrono::microseconds>(exe_end_time - exe_start_time).count() / 1000.0;
    }
}

void CGI610_Analyze::monitor_func()
{

}