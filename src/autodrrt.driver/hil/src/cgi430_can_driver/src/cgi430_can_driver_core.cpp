#include "cgi430_can_driver/cgi430_can_driver_core.hpp"


Cgi430Can::Cgi430Can(const std::string & node_name, const rclcpp::NodeOptions & node_options)
: rclcpp::Node(node_name, node_options),
pub_imu_rate_(declare_parameter("pub_imu_rate", 10.0)),
imu_topic_name_(declare_parameter("imu_topic_name", "imu")),
pub_imu_dt_(declare_parameter("pub_imu_dt", 0.001)),
gnss_topic_name_(declare_parameter("gnss_topic_name", "gnss")),
gnss_pose_topic_name_(declare_parameter("gnss_pose_topic_name", "gnsss/pose")),
pub_gnss_dt_(declare_parameter("pub_gnss_dt", 0.001)),
local_cartesian_(0.0, 0.0, 0.0)
{
    RCLCPP_INFO(this->get_logger(), "[%s] the param IMU Topic Name value is... %s", "cgi430_can_driver_node", imu_topic_name_.c_str());
    RCLCPP_INFO(this->get_logger(), "[%s] the param pub_imu_dt_ value is... %lf", "cgi430_can_driver_node", pub_imu_dt_);

    RCLCPP_INFO(this->get_logger(), "[%s] the param GNSS Topic Name value is... %s", "cgi430_can_driver_node", gnss_topic_name_.c_str());
    RCLCPP_INFO(this->get_logger(), "[%s] the param pub_gnss_dt_ value is... %lf", "cgi430_can_driver_node", pub_gnss_dt_);

    checkCanCardStatus();

    imu_data = std::make_shared<ImuData>();

    /*Publisher*/
    //To AutoWare
    Imu_pub_ = create_publisher<sensor_msgs::msg::Imu>(imu_topic_name_, rclcpp::QoS{1});
    Gnss_pub_ = create_publisher<sensor_msgs::msg::NavSatFix>(gnss_topic_name_, rclcpp::QoS{1});
    Gnss_pose_pub_ = create_publisher<geometry_msgs::msg::PoseStamped>(gnss_pose_topic_name_, rclcpp::QoS{1});

    handleCanFunc();
    const auto duration_imu = rclcpp::Duration::from_seconds(pub_imu_dt_);
    pub_imu_timer_ = rclcpp::create_timer(this, get_clock(), duration_imu, std::bind(&Cgi430Can::pubImuOnTimer, this));

    const auto duration_gnss = rclcpp::Duration::from_seconds(pub_gnss_dt_);
    pub_gnss_timer_ = rclcpp::create_timer(this, get_clock(), duration_gnss, std::bind(&Cgi430Can::pubGnssOnTimer, this));
}

Cgi430Can::~Cgi430Can(){}

void Cgi430Can::pubImuOnTimer()
{
    std::lock_guard<std::mutex> lock(imu_data_mutex_);
    if (!imu_data) {
        RCLCPP_INFO_THROTTLE(this->get_logger(), *get_clock(), std::chrono::milliseconds(1000).count(),
                           "imu_data = %f", imu_data);
        return;
    }
    sensor_msgs::msg::Imu imu_data_msg;
    // std_msgs::msg::Header header;
    // header.frame_id = "imu_link";
    // header.stamp = get_clock()->now();
    imu_data_msg.header.frame_id = "tamagawa/imu_link";
    imu_data_msg.header.stamp = get_clock()->now();
    imu_data_msg.angular_velocity.x = (imu_data->gyro_x /180.0)*M_PI;
    imu_data_msg.angular_velocity.y = (imu_data->gyro_y /180.0)*M_PI;
    imu_data_msg.angular_velocity.z = (imu_data->gyro_z /180.0)*M_PI;
    imu_data_msg.linear_acceleration.x = imu_data->accel_x*9.8;
    imu_data_msg.linear_acceleration.y = imu_data->accel_y*9.8;
    imu_data_msg.linear_acceleration.z = imu_data->accel_z*9.8;
    Imu_pub_-> publish(imu_data_msg);
}

void Cgi430Can::pubGnssOnTimer()
{
    
    std::lock_guard<std::mutex> lock(gnss_data_mutex_);
    sensor_msgs::msg::NavSatFix navsat_msg;
    geometry_msgs::msg::PoseStamped pose_stamped;

    navsat_msg.header.frame_id = "gnss_link";
    navsat_msg.header.stamp = get_clock()->now();
    navsat_msg.latitude = latest_latitude_;
    navsat_msg.longitude = latest_longitude_;
    navsat_msg.altitude = latest_altitude_;
    Gnss_pub_->publish(navsat_msg);

    local_cartesian_.Forward(latest_latitude_, latest_longitude_, latest_altitude_, x_, y_, z_);
    latest_roll_ = latest_roll_ * M_PI/180.0;
    latest_pitch_ = latest_pitch_ * M_PI/180.0;
    latest_heading_ = latest_heading_ * M_PI/180.0;
    Eigen::Quaterniond q_;
    Eigen::Vector3d eulerAngles(latest_heading_, latest_pitch_, latest_roll_);
    q_ = Eigen::AngleAxisd(eulerAngles[0], Eigen::Vector3d::UnitZ()) *
         Eigen::AngleAxisd(eulerAngles[1], Eigen::Vector3d::UnitY()) *
         Eigen::AngleAxisd(eulerAngles[2], Eigen::Vector3d::UnitX());
    
    pose_stamped.header.stamp = get_clock()->now();
    pose_stamped.pose.position.x = x_;
    pose_stamped.pose.position.y = y_;
    pose_stamped.pose.position.z = z_;
    pose_stamped.pose.orientation.x = q_.x();
    pose_stamped.pose.orientation.y = q_.y();
    pose_stamped.pose.orientation.z = q_.z();
    pose_stamped.pose.orientation.w = q_.w();

    Gnss_pose_pub_->publish(pose_stamped);
}

void Cgi430Can::cgi430_rec_func()
{
    while(rclcpp::ok())
    {
        const auto exe_start_time = std::chrono::system_clock::now();
        receive_length_ = VCI_Receive(VCI_USBCAN2, 1, 0, rec, len_, 0);
        // std::cout<<"***************receive_length_: "<<receive_length_<<std::endl;

        if(receive_length_ > 0)
        {
            for(int i=0; i<receive_length_; i++)
            {
                if(rec[i].ID == 0x321)
                {
                    unsigned int signed_rawx = ((rec[i].Data[0] & 0x80) >> 7);
                    unsigned int signed_rawy = ((rec[i].Data[2] & 0x08) >> 3);
                    unsigned int signed_rawz = ((rec[i].Data[5] & 0x80) >> 7);
                    int flag_1 = 0xFFFFF;

                    imu_data_mutex_.lock();
                    if(signed_rawx == 0)
                    {
                        imu_data->gyro_x = (((rec[i].Data[0] & 0xff) << 12) | ((rec[i].Data[1] & 0xff) << 4) | ((rec[i].Data[2] & 0xf0) >> 4))*0.01;
                    }else{
                        imu_data->gyro_x = -(((((rec[i].Data[0] & 0xff) << 12) | ((rec[i].Data[1] & 0xff) << 4) | ((rec[i].Data[2] & 0xf0) >> 4))- 0x01)^flag_1)*0.01;
                    }
                    if(signed_rawy == 0)
                    {
                        imu_data->gyro_y = (((rec[i].Data[2] & 0x0f) << 16) | ((rec[i].Data[3] & 0xff) << 8) | (rec[i].Data[4] & 0xff))*0.01;
                        // std::cout<<"angrateraw_y: "<<std::setprecision(15)<<(angrateraw_y /180.0)*M_PI<<std::endl;
                    }else{
                        imu_data->gyro_y = -(((((rec[i].Data[2] & 0x0f) << 16) | ((rec[i].Data[3] & 0xff) << 8) | (rec[i].Data[4] & 0xff))- 0x01)^flag_1)*0.01;
                        // std::cout<<"angrateraw_y: "<<std::setprecision(15)<<(angrateraw_y /180.0)*M_PI<<std::endl;
                    }
                    if(signed_rawz ==0)
                    {
                        imu_data->gyro_z = (((rec[i].Data[5] & 0xff) << 12) | ((rec[i].Data[6] & 0xff) << 4) | ((rec[i].Data[7] & 0xf0) >> 4))*0.01;
                    }else{
                        imu_data->gyro_z = -(((((rec[i].Data[5] & 0xff) << 12) | ((rec[i].Data[6] & 0xff) << 4) | ((rec[i].Data[4] & 0xf0) >> 4))- 0x01)^flag_1)*0.01;
                    }
                    std::cout<<"gyro_x: "<<imu_data->gyro_x<<" gyro_y: "<<imu_data->gyro_y<<" gyro_z: "<< imu_data->gyro_z<<std::endl;
                    imu_data_mutex_.unlock();
                    
                }

                if(rec[i].ID == 0x322)
                {
                    unsigned int signed_x = ((rec[i].Data[0] & 0x80) >> 7);
                    unsigned int signed_y = ((rec[i].Data[2] & 0x08) >> 3);
                    unsigned int signed_z = ((rec[i].Data[5] & 0x80) >> 7);
                    int flag_2 = 0xFFFFF;
                    imu_data_mutex_.lock();
                    if(signed_x == 0)
                    {
                        imu_data->accel_x = (((rec[i].Data[0] & 0xff) << 12) | ((rec[i].Data[1] & 0xff) << 4) | ((rec[i].Data[2] & 0xf0) >> 4))*0.0001;
                    }else{
                        imu_data->accel_x = -((((((rec[i].Data[0] & 0xff) << 12) | ((rec[i].Data[1] & 0xff) << 4) | ((rec[i].Data[2] & 0xf0) >> 4))- 0x01)^flag_2)*0.0001);
                    }
                    if(signed_y == 0)
                    {
                        imu_data->accel_y = (((rec[i].Data[2] & 0x0f) << 16) | ((rec[i].Data[3] & 0xff) << 8) | (rec[i].Data[4] & 0xff))*0.0001;
                    }else{
                        imu_data->accel_y = -(((((rec[i].Data[2] & 0x0f) << 16) | ((rec[i].Data[3] & 0xff) << 8) | (rec[i].Data[4] & 0xff))- 0x01)^flag_2)*0.0001;
                    }
                    if(signed_z == 0)
                    {
                        imu_data->accel_z = (((rec[i].Data[5] & 0xff) << 12) | ((rec[i].Data[6] & 0xff) << 4) | ((rec[i].Data[7] & 0xf0) >> 4))*0.0001;
                    }else{
                        imu_data->accel_z = -(((((rec[i].Data[5] & 0xff) << 12) | ((rec[i].Data[6] & 0xff) << 4) | ((rec[i].Data[4] & 0xf0) >> 4))- 0x01)^flag_2)*0.0001;
                    }
                    std::cout<<"accel_x: "<<imu_data->accel_x<<" accel_y: "<<imu_data->accel_y<<" accel_z: "<< imu_data->accel_z<<std::endl;
                    imu_data_mutex_.unlock();
                }

                if(rec[i].ID == 0x32E)
                {
                    // double latitude_1 = parseLatitude(rec[i].Data);
                    // std::lock_guard<std::mutex> lock(gnss_data_mutex_);
                    double latitude = decodeLatitudeOrLongitude(rec[i].Data, 0, 64, 1e-08, 0.0);
                    std::cout<<" latitude: "<<latitude<<std::endl;
                    gnss_data_mutex_.lock();
                    latest_latitude_ = latitude;
                    
                    gnss_data_mutex_.unlock();
                }
                if(rec[i].ID == 0x32D)
                {
                    double longitude = parseLatitude(rec[i].Data);
                    gnss_data_mutex_.lock();
                    latest_longitude_ = longitude;
                    std::cout<<"longitude: "<<longitude<<std::endl;
                    gnss_data_mutex_.unlock();
                }
                if(rec[i].ID == 0x325)
                {
                    // double altitude = parseAltitude(rec[i].Data);
                    double altitude = decodeLatitudeOrLongitude(rec[i].Data, 0, 32, 0.001, 0.0);
                    gnss_data_mutex_.lock();
                    latest_altitude_ = altitude;
                    std::cout<<"altitude: "<<altitude<<std::endl;
                    gnss_data_mutex_.unlock();
                }
                if(rec[i].ID ==0x32A)
                {
                    // double heading = decodeLatitudeOrLongitude(rec[i].Data, 0, 16, 0.01, 0.0);
                    // double heading = parseHeading(rec[i].Data);
                    double heading = (((rec[i].Data[0] & 0xff) << 8) | (rec[i].Data[0] & 0xff))*0.01;
                    double pitch = decodeLatitudeOrLongitude(rec[i].Data, 16, 16, 0.01, 0.0);
                    double roll = decodeLatitudeOrLongitude(rec[i].Data, 32, 16, 0.01, 0.0);
                    std::cout<<"heading: "<<heading<<std::endl;
                    std::cout<<"pitch: "<<pitch<<std::endl;
                    std::cout<<"roll: "<<roll<<std::endl;
                    gnss_data_mutex_.lock();
                    latest_heading_ = heading;
                    latest_pitch_ = pitch;
                    latest_roll_ = roll;
                    gnss_data_mutex_.unlock();

                }
            }

        }
        else if(receive_length_ == -1){
            VCI_CloseDevice(VCI_USBCAN2,1);
            RCLCPP_ERROR(this->get_logger(),"The CAN card disconnect,please check it!");
        }else{
            // RCLCPP_INFO(this->get_logger(), "this in can_rec_thread<<<");
            if(!(rclcpp::ok())){
            break;
            }  
        }
        // const auto exe_end_time = std::chrono::system_clock::now();
        // const double exe_time =std::chrono::duration_cast<std::chrono::microseconds>(exe_end_time - exe_start_time).count() / 1000.0;
        // RCLCPP_INFO_THROTTLE(get_logger(), *get_clock(), std::chrono::milliseconds(1000).count(),"exe_time = %f", exe_time);
        // RCLCPP_INFO(this->get_logger(), "exe_time: '%f'", exe_time);
        // usleep(10000);
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
}


void Cgi430Can::checkCanCardStatus()
{
    RCLCPP_INFO(this->get_logger(), ">> >> >> CAN card detection start << << <<");
    int num = VCI_FindUsbDevice2(pInfo);
    RCLCPP_INFO(this->get_logger(), "USBCAN DEVICE NUM: %d PCS", num);

    // for (int i = 0; i < num; i++)
    // {
    //     RCLCPP_INFO(this->get_logger(), "Device: %d", i);
    //     RCLCPP_INFO(this->get_logger(), "Serial_Num: %s", pInfo[i].str_Serial_Num);
    //     RCLCPP_INFO(this->get_logger(), "hw_Type: %s", pInfo[i].str_hw_Type);
    //     RCLCPP_INFO(this->get_logger(), "Firmware Version: V%x.%x%x",
    //                 (pInfo[i].fw_Version & 0xF00) >> 8,
    //                 (pInfo[i].fw_Version & 0xF0) >> 4,
    //                  pInfo[i].fw_Version & 0xF);
    // }

    if (VCI_OpenDevice(VCI_USBCAN2, 1, 0) == 1){
        RCLCPP_INFO(this->get_logger(), "open device1 successfully!");
    }else{
        RCLCPP_ERROR(this->get_logger(), "open device1 failed!");
        exit(1);
    }

    // VCI_BOARD_INFO pInfo;
    // if (VCI_ReadBoardInfo(VCI_USBCAN2, 0, &pInfo) == 1){
    //     RCLCPP_INFO(this->get_logger(), "Get VCI_ReadBoardInfo success!");
    //     RCLCPP_INFO(this->get_logger(), "Serial_Num: %s", pInfo.str_Serial_Num);
    //     RCLCPP_INFO(this->get_logger(), "hw_Type: %s", pInfo.str_hw_Type);
    //     RCLCPP_INFO(this->get_logger(), "Firmware Version: V%x.%x%x",
    //                 (pInfo.fw_Version & 0xF00) >> 8,
    //                 (pInfo.fw_Version & 0xF0) >> 4,
    //                 pInfo.fw_Version & 0xF);
    // }else{
    //     RCLCPP_ERROR(this->get_logger(), "Get VCI_ReadBoardInfo error!");
    //     exit(1);
    // }

    VCI_INIT_CONFIG can_param_config;
    can_param_config.AccCode = 0;
    can_param_config.AccMask = 0xFFFFFFFF; 
    can_param_config.Filter = 1;
    can_param_config.Timing0 = 0X00;
    can_param_config.Timing1 = 0X1C;
    can_param_config.Mode = 0;

    if (VCI_InitCAN(VCI_USBCAN2, 1, 0, &can_param_config) != 1)
    {
        RCLCPP_ERROR(this->get_logger(), "Init CAN2 failed");
        VCI_CloseDevice(VCI_USBCAN2, 1);
    }else{
        RCLCPP_INFO(this->get_logger(), "Init CAN2 successfully! ");
    }

    if (VCI_StartCAN(VCI_USBCAN2, 1, 0) != 1)
    {
        RCLCPP_ERROR(this->get_logger(), "Start CAN2 failed");
        VCI_CloseDevice(VCI_USBCAN2, 1);
    }else{
        RCLCPP_INFO(this->get_logger(), "Start CAN2 successfully! ");
    }

    RCLCPP_INFO(this->get_logger(), ">> >> >>  card detection end! << << <<");
}

void Cgi430Can::handleCanFunc()
{
    cgi430_rec_thread_ = std::thread(&Cgi430Can::cgi430_rec_func, this);
    usleep(100000);
    cgi430_rec_thread_.detach();
}

double Cgi430Can::parseLatitude(const uint8_t* can_data)
{
    unsigned int signed_x = ((can_data[0] & 0x80) >> 7);
    // std::cout<<"signed_x: "<<signed_x<<std::endl;
    int64_t raw_value = 0;
    int64_t flag = 0xFFFFFFFFFFFFFFFF;

    for(int i = 0; i< 8; ++i)
    {
        raw_value |= (static_cast<int64_t>(can_data[7-i]) << (i * 8));
    }

    if(signed_x == 0){
        raw_value = raw_value;
    }else{
        raw_value = -(raw_value -0x01)^flag;
    }

    double factor = 1e-8;
    double latitude = raw_value * factor;

    return latitude;
}

double Cgi430Can::parseHeading(const uint8_t* can_data)
{
    int16_t raw_value = 0;

    raw_value = ((can_data[0] & 0xff) << 8) | (can_data[1] & 0xff);
    double factor = 0.01;
    double altitude = raw_value * factor;

    return altitude;
}

double Cgi430Can::parseAltitude(const uint8_t* can_data)
{
    int32_t raw_value = 0;
    raw_value |= (static_cast<int32_t>(can_data[0]) << 24);
    raw_value |= (static_cast<int32_t>(can_data[1]) << 16);
    raw_value |= (static_cast<int32_t>(can_data[2]) << 8);
    raw_value |= (static_cast<int32_t>(can_data[3]));
    double factor = 0.001;
    double altitude = raw_value * factor;

    return altitude;
}

int64_t Cgi430Can::extractSignedSignal(const uint8_t* data, int startBit, int length) {
    uint64_t rawValue = 0;

    // Extract bits based on Motorola (Big Endian) format
    for (int i = 0; i < length; ++i) {
        int byteIndex = (startBit + i) / 8;
        int bitIndex = 7 - ((startBit + i) % 8);
        rawValue = (rawValue << 1) | ((data[byteIndex] >> bitIndex) & 0x1);
    }

    // Convert to signed integer if needed
    if (rawValue & (1ULL << (length - 1))) { // Check the sign bit
        rawValue -= (1ULL << length); // Convert to negative number
    }

    return static_cast<int64_t>(rawValue);
}

double Cgi430Can::decodeLatitudeOrLongitude(const uint8_t* data, int startBit, int length, double scale, double offset) {
    int64_t rawValue = extractSignedSignal(data, startBit, length);
    return rawValue * scale + offset;
}