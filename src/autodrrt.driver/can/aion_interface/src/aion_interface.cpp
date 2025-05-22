#include "aion_interface/aion_interface.hpp"

AionInterface::AionInterface(const std::string & node_name, const rclcpp::NodeOptions & node_options)
: rclcpp::Node(node_name, node_options),
  pub_rate_(declare_parameter("pub_rate", 10.0)), 
  report_dt_(declare_parameter("report_dt", 1.0)),
  command_dt_(declare_parameter("command_dt", 1.0)),
  base_frame_id_(declare_parameter("base_frame_id", "base_link"))
{
    RCLCPP_INFO(this->get_logger(), "[%s] the param pub_rate_ value is... %lf", "aion_interface", pub_rate_);
    RCLCPP_INFO(this->get_logger(), "[%s] the param report_dt_ value is... %lf", "aion_interface", report_dt_);
    RCLCPP_INFO(this->get_logger(), "[%s] the param command_dt_ value is... %lf", "aion_interface", command_dt_);
    checkCanCardStatus();

    /* subscribers */
    using std::placeholders::_1;
    //From Autoware
    control_cmd_sub_ = create_subscription<autoware_auto_control_msgs::msg::AckermannControlCommand>(
                       "/control/command/control_cmd", 1, std::bind(&AionInterface::callbackControlCmd, this, _1));
    gear_cmd_sub_ = create_subscription<autoware_auto_vehicle_msgs::msg::GearCommand>(
                    "/control/command/gear_cmd", 1, std::bind(&AionInterface::callbackGearCmd, this, _1));

    /* publisher */
    //To Autoware
    vehicle_twist_pub_ = create_publisher<autoware_auto_vehicle_msgs::msg::VelocityReport>("/vehicle/status/velocity_status", rclcpp::QoS{1});
    steering_status_pub_ = create_publisher<autoware_auto_vehicle_msgs::msg::SteeringReport>("/vehicle/status/steering_status", rclcpp::QoS{1});        
    gear_status_pub_ = create_publisher<autoware_auto_vehicle_msgs::msg::GearReport>("/vehicle/status/gear_status", rclcpp::QoS{1});

    handleCanFunc();

    const auto duration_s = rclcpp::Duration::from_seconds(report_dt_);
    pub_report_timer_ = rclcpp::create_timer(this, get_clock(), duration_s, std::bind(&AionInterface::pubReportOnTimer, this));  

    const auto duration_ = rclcpp::Duration::from_seconds(command_dt_);
    pub_command_timer_ = rclcpp::create_timer(this, get_clock(), duration_, std::bind(&AionInterface::pubCommandOnTimer, this));             
    
}
AionInterface::~AionInterface(){}


void AionInterface::pubReportOnTimer()
{
    std::lock_guard<std::mutex> lock(mux_);

    // if (!velocity_report_ptr_ ) {
    //     RCLCPP_INFO_THROTTLE(this->get_logger(), *get_clock(), std::chrono::milliseconds(1000).count(),
    //                        "vehicle_report = %d, velocity_report_ptr_ = %f", 
    //                        velocity_report_ptr_ != nullptr, velocity_report_ptr_);
    //     return;
    // }
    // RCLCPP_INFO(this->get_logger(), "vehicle_report =%d, velocity_report_ptr_ = %f", velocity_report_ptr_ != nullptr, *velocity_report_ptr_);

    // if (!gear_report_ptr_ ) {
    //     RCLCPP_INFO_THROTTLE(this->get_logger(), *get_clock(), std::chrono::milliseconds(1000).count(),
    //                        "gear_report = %d, gear_report_ptr_ = %d", 
    //                        gear_report_ptr_ != nullptr, *gear_report_ptr_);
    //     return;
    // }
    // RCLCPP_INFO(this->get_logger(), "gear_report =%d, gear_report_ptr_ = %d", gear_report_ptr_ != nullptr, *gear_report_ptr_);


    if (!steer_report_ptr_) {
        RCLCPP_INFO_THROTTLE(this->get_logger(), *get_clock(), std::chrono::milliseconds(1000).count(),
                           "steer_report = %d, steer_report_ptr_ = %f", 
                           steer_report_ptr_ != nullptr, *steer_report_ptr_);
        return;
    }
     RCLCPP_INFO(this->get_logger(), "steer_report =%d, steer_report_ptr_ = %f", steer_report_ptr_ != nullptr, *steer_report_ptr_);


    std_msgs::msg::Header header;
    header.frame_id = base_frame_id_;
    header.stamp = get_clock()->now();

    /* publish  aion vehicle status twist */
    // {
    //   autoware_auto_vehicle_msgs::msg::VelocityReport twist;
    //   twist.header = header;
    //   twist.longitudinal_velocity = *velocity_report_ptr_;
    //   vehicle_twist_pub_->publish(twist);
    // }

    /* publish aion steering wheel status */
    {
        autoware_auto_vehicle_msgs::msg::SteeringReport steer_msg;
        steer_msg.stamp = header.stamp;
        steer_msg.steering_tire_angle = SteerToWheelAngle(*steer_report_ptr_);
        steering_status_pub_->publish(steer_msg);
    }

    /*publish aion gear status*/
    // {
    //     autoware_auto_vehicle_msgs::msg::GearReport gear_report_msg;
    //     gear_report_msg.stamp = header.stamp;
    //     const auto opt_gear_report = toAutowareShiftReport(*gear_report_ptr_);
    //     if(opt_gear_report){
    //         gear_report_msg.report = *opt_gear_report;
    //         gear_status_pub_->publish(gear_report_msg);
    //     }
    // }
    // RCLCPP_INFO(this->get_logger(), *get_clock(), std::chrono::milliseconds(1000).count(), "pubReportOnTimer mode Normal operation");
} 


void AionInterface::can_rec_func()
{
    while(rclcpp::ok())
    {
        // RCLCPP_INFO(this->get_logger(), "this in can_rec_thread<<<");
        const auto exe_start_time = std::chrono::system_clock::now();
        receive_length_ = VCI_Receive(VCI_USBCAN2, 0, 0, rec, len_, 0);
        
        if(receive_length_>0)
        {
            for(int i=0; i<receive_length_; i++)
            {
                if(rec[i].ID == 0x180)
                {
                    float speed_ = (((rec[i].Data[4]&0x1f)<<8) | (rec[i].Data[5]&0xff))*0.05625;
                    // RCLCPP_INFO(this->get_logger(), "speed_ = '%f'", speed_);
                    mux_.lock();
                    velocity_report_ptr_ = std::make_shared<float>(speed_);
                    mux_.unlock();
                    // autoware_auto_vehicle_msgs::msg::VelocityReport twist;
                    // twist.header.frame_id = base_frame_id_;
                    // twist.header.stamp = this->now();
                    // twist.longitudinal_velocity = speed_; 
                    // vehicle_twist_pub_->publish(twist);

                }
                if(rec[i].ID ==0x22B)
                {
                    uint8_t gear_status_ = ((rec[i].Data[4]&0x70)>>4);
                    // RCLCPP_INFO(this->get_logger(), "gear_status_ = '%d'", gear_status_);
                    mux_.lock();
                    gear_report_ptr_ = std::make_shared<uint8_t>(gear_status_);
                    mux_.unlock();
                    // autoware_auto_vehicle_msgs::msg::GearReport gear_report_msg;
                    // gear_report_msg.stamp = this->now();
                    // const auto opt_gear_report = toAutowareShiftReport(gear_status_);
                    // if(opt_gear_report){
                    //     gear_report_msg.report = *opt_gear_report;
                    //     gear_status_pub_->publish(gear_report_msg);
                    // }

                }
                if(rec[i].ID == 0x229)
                {
                    float steering_ = ((((rec[i].Data[1]&0xff)<<8) | (rec[i].Data[2]&0xff))*0.1)-780;
                    RCLCPP_INFO(this->get_logger(), "steering_ = '%f'", steering_);
                    mux_.lock();
                    steer_report_ptr_ = std::make_shared<float>(steering_);
                    mux_.unlock();
                    // autoware_auto_vehicle_msgs::msg::SteeringReport steer_msg;
                    // steer_msg.stamp = this->now();
                    // steer_msg.steering_tire_angle = SteerToWheelAngle(steering_);
                    // steering_status_pub_->publish(steer_msg);

                }

                if(rec[i].ID == 0x22A)
                {
                    uint8_t scu_latctrmode_ = ((rec[i].Data[0]& 0x0c)>>2);
                    uint8_t scu_lngctrmode_ = (rec[i].Data[0]& 0x03);
                    // RCLCPP_INFO(this->get_logger(), "scu_latctrmode_: '%d', scu_lngctrmode_: '%d'", scu_latctrmode_, scu_lngctrmode_);
                    mux_scu_.lock();
                    scu_latctrmode_ptr_ = std::make_shared<uint8_t>(scu_latctrmode_);
                    scu_lngctrmode_ptr_ = std::make_shared<uint8_t>(scu_lngctrmode_);
                    mux_scu_.unlock();
                    // scu_mode.Scu_LatctrMode = scu_latctrmode_;
                    // scu_mode.Scu_LngctrMode = scu_lngctrmode_;
                    // scu_mode_v.push_back(scu_mode);
                    // if(scu_mode_v.size()>2){
                    //     scu_mode_v.pop_front();
                    // }
                }

                if(rec[i].ID == 0X22D)
                {

                }
                

            }

        }
        else if(receive_length_ == -1){
            VCI_CloseDevice(VCI_USBCAN2,0);
            RCLCPP_ERROR(this->get_logger(),"The CAN card disconnect,please check it!");
        }else{
            // RCLCPP_INFO(this->get_logger(), "this in can_rec_thread<<<");
            if(!(rclcpp::ok())){
            break;
            }  
        }
        const auto exe_end_time = std::chrono::system_clock::now();
        const double exe_time =std::chrono::duration_cast<std::chrono::microseconds>(exe_end_time - exe_start_time).count() / 1000.0;
        RCLCPP_INFO_THROTTLE(get_logger(), *get_clock(), std::chrono::milliseconds(1000).count(),"exe_time = %f", exe_time);
        RCLCPP_INFO(this->get_logger(), "exe_time: '%f'", exe_time);
        usleep(30000);
    }
}

void AionInterface::pubCommandOnTimer()
{
    // const auto exe_start_time = std::chrono::system_clock::now();
    std::lock_guard<std::mutex> lock(mux_scu_);
    // RCLCPP_INFO(this->get_logger(), "control_cmd_ptr_lat= %f, control_cmd_ptr_lng= %f",control_cmd_ptr_->lateral.steering_tire_angle, control_cmd_ptr_->longitudinal.acceleration);
    // RCLCPP_INFO(this->get_logger(), "gear_cmd_ptr_=%d", gear_cmd_ptr_-> command);
    // RCLCPP_INFO(this->get_logger(), "scu_latctrmode = %d, scu_lngctrmode = %d", *scu_latctrmode_ptr_, *scu_lngctrmode_ptr_);

    if(!control_cmd_ptr_ || !gear_cmd_ptr_ || !scu_latctrmode_ptr_ || !scu_lngctrmode_ptr_)
    {
        RCLCPP_INFO_THROTTLE(this->get_logger(),*get_clock(), std::chrono::milliseconds(1000).count(), 
                "control_cmd_ptr_= %d, gear_cmd_ptr_=%d, scu_latctrmode_ptr_=%d, scu_lngctrmode_ptr_=%d",
                control_cmd_ptr_ != nullptr, gear_cmd_ptr_!=nullptr, scu_latctrmode_ptr_!=nullptr, scu_lngctrmode_ptr_!=nullptr);
        return;
    }
    // RCLCPP_INFO(this->get_logger(), "control_cmd_ptr_= %d, gear_cmd_ptr_=%d,control_cmd_ptr_lat= %f, control_cmd_ptr_lng= %f, *gear_cmd_ptr_=%d",
    //             control_cmd_ptr_ != nullptr,gear_cmd_ptr_!=nullptr,control_cmd_ptr_->lateral.steering_tire_angle, control_cmd_ptr_->longitudinal.acceleration, gear_cmd_ptr_-> command);
    // RCLCPP_INFO(this->get_logger(), "scu_latctrmode = %d, scu_lngctrmode = %d, *scu_latctrmode_ptr_=%d, *scu_lngctrmode_ptr_=%d", 
    //             scu_latctrmode_ptr_!=nullptr, scu_lngctrmode_ptr_!=nullptr, *scu_latctrmode_ptr_, *scu_lngctrmode_ptr_);

    uint8_t latctrlreq_, lngctrlreq_;
    float trq_,brake_;
    float acceleration_ = control_cmd_ptr_ -> longitudinal.acceleration;
    float steerangreq_ = (((control_cmd_ptr_ -> lateral.steering_tire_angle)/M_PI)*180.0)*15;
    uint8_t gear_cmd_ = gear_cmd_ptr_ -> command;

    if(*scu_latctrmode_ptr_ == 0 || *scu_lngctrmode_ptr_ == 0)
    {
        latctrlreq_ = 1;
        lngctrlreq_ = 1;
    }else if(*scu_latctrmode_ptr_ == 2 || *scu_lngctrmode_ptr_ == 2){
        latctrlreq_ = 1;
        lngctrlreq_ = 1;
    }else{
        latctrlreq_ = 0;
        lngctrlreq_ = 0;
    }

    if(acceleration_>=0){
        trq_ = CalCtrlByacc(acceleration_);
        brake_ = 0;
    }else{
        brake_ = acceleration_;
        trq_ =0;
    }
    const auto gearlvlreq_opt = GearAutowareToAion(gear_cmd_);
    if(gearlvlreq_opt.has_value()){
        uint8_t gearlvlreq_ = gearlvlreq_opt.value(); 
        send_adcu(steerangreq_, trq_, brake_, gearlvlreq_, latctrlreq_, lngctrlreq_);
    }else {
        RCLCPP_INFO(this->get_logger(), "Invalid gear command");
    }

    // const auto exe_end_time = std::chrono::system_clock::now();
    // const double exe_time =std::chrono::duration_cast<std::chrono::microseconds>(exe_end_time - exe_start_time).count() / 1000.0;
    // RCLCPP_INFO_THROTTLE(get_logger(), *get_clock(), std::chrono::milliseconds(1000).count(),"exe_time = %f", exe_time);
    
}

void AionInterface::callbackControlCmd(const autoware_auto_control_msgs::msg::AckermannControlCommand::ConstSharedPtr msg)
{
    control_command_received_time_ = this->now();
    control_cmd_ptr_ = msg;
    // RCLCPP_INFO(this->get_logger(), "control_cmd_ptr_lat= %f, control_cmd_ptr_lng= %f",control_cmd_ptr_->lateral.steering_tire_angle, control_cmd_ptr_->longitudinal.acceleration);
}

void AionInterface::callbackGearCmd(const autoware_auto_vehicle_msgs::msg::GearCommand::ConstSharedPtr msg)
{
    gear_cmd_ptr_ = msg;
    // RCLCPP_INFO(this->get_logger(), "gear_cmd_ptr_=%d", gear_cmd_ptr_-> command);
}

// viod AionInterface::can_send_func(){}

void AionInterface::handleCanFunc()
{
    can_rec_thread_ = std::thread(&AionInterface::can_rec_func, this);
    // can_send_thread_ = std::thread(&AionInterface::can_send_func, this);
    usleep(100000);
    can_rec_thread_.detach();
    // can_send_thread_.detach();
}
//Function 

float AionInterface::CalCtrlByacc(float acc_)
{
    float ctrl = 0;
    const float acc_torque[] = {100,200,300,400,500,600,700,800,900,1000,1098,1196,1294,1392,
                                  1490,1570,1648,1726,1804,1882,1960,2166,2166,2500,2562,2625,2687,
                                  2750,2812,2875,2937,3000};
    if(acc_ <0){ acc_ = 0.0;}

    float prop = (fabs(acc_)/0.1);
    int index = fabs(acc_)/0.1;
    index = index > 30? 30 : index;
    ctrl = acc_torque[index] + (acc_torque[index+1] - acc_torque[index]) * (prop -index);
    return ctrl;
}

float AionInterface::SteerToWheelAngle(float steering)
{
    float wheel = ((steering/15.0)/180.0)*M_PI;
    return wheel;
}

void AionInterface::checkCanCardStatus()
{
    RCLCPP_INFO(this->get_logger(), ">> >> >> CAN card detection start << << <<");
    int num=VCI_FindUsbDevice2(pInfo);
    RCLCPP_INFO(this->get_logger(), "USBCAN DEVICE NUM: %d PCS", num);

    for (int i = 0; i < num; i++)
    {
        RCLCPP_INFO(this->get_logger(), "Device: %d", i);
        RCLCPP_INFO(this->get_logger(), "Serial_Num: %s", pInfo[i].str_Serial_Num);
        RCLCPP_INFO(this->get_logger(), "hw_Type: %s", pInfo[i].str_hw_Type);
        RCLCPP_INFO(this->get_logger(), "Firmware Version: V%x.%x%x",
                    (pInfo[i].fw_Version & 0xF00) >> 8,
                    (pInfo[i].fw_Version & 0xF0) >> 4,
                     pInfo[i].fw_Version & 0xF);
    }

    if (VCI_OpenDevice(VCI_USBCAN2, 0, 0) == 1){
        RCLCPP_INFO(this->get_logger(), "open device success!");
    }else{
        RCLCPP_ERROR(this->get_logger(), "open device error!");
        exit(1);
    }

    VCI_BOARD_INFO pInfo;
    if (VCI_ReadBoardInfo(VCI_USBCAN2, 0, &pInfo) == 1){
        RCLCPP_INFO(this->get_logger(), "Get VCI_ReadBoardInfo success!");
        RCLCPP_INFO(this->get_logger(), "Serial_Num: %s", pInfo.str_Serial_Num);
        RCLCPP_INFO(this->get_logger(), "hw_Type: %s", pInfo.str_hw_Type);
        RCLCPP_INFO(this->get_logger(), "Firmware Version: V%x.%x%x",
                    (pInfo.fw_Version & 0xF00) >> 8,
                    (pInfo.fw_Version & 0xF0) >> 4,
                    pInfo.fw_Version & 0xF);
    }else{
        RCLCPP_ERROR(this->get_logger(), "Get VCI_ReadBoardInfo error!");
        exit(1);
    }

    VCI_INIT_CONFIG can_param_config;
    can_param_config.AccCode = 0;
    can_param_config.AccMask = 0xFFFFFFFF; 
    can_param_config.Filter = 1;
    can_param_config.Timing0 = 0X00;
    can_param_config.Timing1 = 0X1C;
    can_param_config.Mode = 0;
    if(VCI_InitCAN(VCI_USBCAN2, 0, 0, &can_param_config)!=1)
    {
        RCLCPP_ERROR(this->get_logger(), "Init CAN1 error");
        VCI_CloseDevice(VCI_USBCAN2, 0);
    }else{
        RCLCPP_INFO(this->get_logger(), "Init CAN1 success! ");
    }

    if (VCI_StartCAN(VCI_USBCAN2, 0, 0) != 1)
    {
        RCLCPP_ERROR(this->get_logger(), "Start CAN1 error");
        VCI_CloseDevice(VCI_USBCAN2, 0);
    }else{
        RCLCPP_INFO(this->get_logger(), "Start CAN1 success! ");
    }

    if (VCI_InitCAN(VCI_USBCAN2, 0, 1, &can_param_config) != 1)
    {
        RCLCPP_ERROR(this->get_logger(), "Init CAN2 error");
        VCI_CloseDevice(VCI_USBCAN2, 0);
    }else{
        RCLCPP_INFO(this->get_logger(), "Init CAN2 success! ");
    }

    if (VCI_StartCAN(VCI_USBCAN2, 0, 1) != 1)
    {
        RCLCPP_ERROR(this->get_logger(), "Start CAN2 error");
        VCI_CloseDevice(VCI_USBCAN2, 0);
    }else{
        RCLCPP_INFO(this->get_logger(), "Start CAN1 success! ");
    }

    RCLCPP_INFO(this->get_logger(), ">> >> >>  card detection end! << << <<");
}

std::optional<int32_t> AionInterface::toAutowareShiftReport(const uint8_t gear_status)
{
    using autoware_auto_vehicle_msgs::msg::GearReport;
    
    if(gear_status == 1){
        return GearReport::DRIVE;
    }
    if(gear_status == 2){
        return GearReport::NEUTRAL;
    }
    if(gear_status == 3){
        return GearReport::REVERSE;
    }
    if(gear_status == 4){
        return GearReport::PARK;
    }
    return {};
}

std::optional<uint8_t> AionInterface::GearAutowareToAion(const uint8_t gear_cmd)
{
    uint8_t gearlvlreq;
    if(gear_cmd == 1){
        gearlvlreq = 0x3; 
        return gearlvlreq;
    }else if(gear_cmd == 2 || gear_cmd == 3 || gear_cmd == 4 || gear_cmd == 5 || gear_cmd == 6 || gear_cmd == 7 || gear_cmd == 8 || gear_cmd == 9 || gear_cmd == 10 || gear_cmd == 11
             || gear_cmd == 12 || gear_cmd == 13 || gear_cmd == 14 || gear_cmd == 15 || gear_cmd == 16 || gear_cmd == 17 || gear_cmd == 18 || gear_cmd == 19){
        gearlvlreq = 0x4;
        return gearlvlreq;
    }else if(gear_cmd == 20 || gear_cmd == 21){
        gearlvlreq = 0x2;
        return gearlvlreq;
    }else if(gear_cmd == 22){
        gearlvlreq = 0x1;
        return gearlvlreq;
    }
    return {};
}

void AionInterface::send_adcu(float steerangreq, float autotrqwhlreq, float brakereq, uint8_t gearlvlreq, uint8_t latctrlreq, uint8_t lngctrlreq)
{
    adcu_1.DATA01.bit.BrakeReq = ((brakereq+16)/0.0004882);
    adcu_1.DATA23.bit.LngCtrlReq = lngctrlreq;
    adcu_1.DATA23.bit.AutoTrqWhlReq = ((autotrqwhlreq+5000)/1);
    adcu_1.DATA4.bit.reserved = 0;
    // adcu_1.DATA4.bit.ParkingReqEPBVD = 0;
    // adcu_1.DATA4.bit.ParkingReqToEPB = 0;
    adcu_1.DATA4.bit.GearLvlReqVD = 1;
    adcu_1.DATA4.bit.GearLvlReq = gearlvlreq;
    adcu_1.DATA5.bit.reserved = 0;
    // adcu_1.DATA6.bit.MsgCounter = 0;
    // adcu_1.DATA6.bit.reserved = 0;
    // adcu_1.DATA7.bit.Checksum = 0;

    vco[0].ID = 0X220;
    vco[0].RemoteFlag = 0;
    vco[0].ExternFlag = 0;
    vco[0].DataLen = 8;
    vco[0].Data[0] = adcu_1.DATA01.D>>8;
    vco[0].Data[1] = adcu_1.DATA01.D;
    vco[0].Data[2] = adcu_1.DATA23.D>>8;
    vco[0].Data[3] = adcu_1.DATA23.D;
    vco[0].Data[4] = adcu_1.DATA4.D;
    vco[0].Data[5] = adcu_1.DATA5.D;
    vco[0].Data[6] = adcu_1.DATA6.D;
    vco[0].Data[7] = adcu_1.DATA7.D;

    adcu_2.DATA01.bit.SteerAngReq = ((steerangreq+780)/0.1);
    // adcu_2.DATA23.bit.SteerAngSpdLimt = 0;
    adcu_2.DATA45.bit.reserved = 0;
    adcu_2.DATA45.bit.LatCtrReq = latctrlreq/1;
    // adcu_2.DATA45.bit.SteerWhlTorqReq = ((3+10.24)/0.01);
    // adcu_2.DATA6.bit.MsgCounter = 0;
    // adcu_2.DATA6.bit.reserved = 0;
    // adcu_2.DATA7.bit.CheckSum = 0;

    vco[1].ID = 0x221;
    vco[1].RemoteFlag = 0;
    vco[1].ExternFlag = 0;
    vco[1].DataLen = 8;
    vco[1].Data[0] = adcu_2.DATA01.D>>8;
    vco[1].Data[1] = adcu_2.DATA01.D;
    vco[1].Data[2] = adcu_2.DATA23.D>>8;
    vco[1].Data[3] = adcu_2.DATA23.D;
    vco[1].Data[4] = adcu_2.DATA45.D>>8;
    vco[1].Data[5] = adcu_2.DATA45.D;
    vco[1].Data[6] = adcu_2.DATA6.D;
    vco[1].Data[7] = adcu_2.DATA7.D;

    VCI_Transmit(VCI_USBCAN2, 0, 0, vco, 2);
}