#include "cgi430_can_driver/cgi430_can_driver_core.hpp"
#include <iomanip>

CGI430_CAN::CGI430_CAN(const std::string & node_name, const rclcpp::NodeOptions & node_options)
: rclcpp::Node(node_name, node_options)
{
    init_param();
    pub_imu_ = create_publisher<sensor_msgs::msg::Imu>(_out_imu_topic, 10);

    CanPortInit();
    ReceiveDate();
}

void CGI430_CAN::init_param()
{
    _out_imu_topic = declare_parameter("out_imu_topic", "/dev/ttyUSB0");
    RCLCPP_INFO(this->get_logger(),"[%s] the param out_object_topic value is... %s", "cgi430_can_driver_node", _out_imu_topic.c_str());

    out_hz_ = declare_parameter("out_hz", 1000);
    RCLCPP_INFO(this->get_logger(),"[%s] the param out_hz value is... %lf", "cgi430_can_driver_node", out_hz_);

}

void CGI430_CAN::CanPortInit()
{
    std::cout<<"*****************"<<std::endl;
    /* 查找并返回设备数量*/
    int num=VCI_FindUsbDevice2(pInfo1);
	printf(">> >>USBCAN DEVICE NUM: %d\n", num);
	printf("\n");
    

    /*打开设备，判断是否打开成功*/
    if(VCI_OpenDevice(VCI_USBCAN2,0,0)==1){
        printf(">> >>open deivce success!\n");
    }else{
        printf(">>open deivce error!\n");
		exit(1);
    }

    /*VCI_InitCAN()　初始化指定ＣＡＮ通道，并检查是否初始化成功*/
    VCI_INIT_CONFIG cgi430_config;
    cgi430_config.AccCode = 0;          //
    cgi430_config.AccMask = 0xFFFFFFFF; 
    cgi430_config.Filter = 1;
    cgi430_config.Timing0 = 0x00;
    cgi430_config.Timing1 = 0x1c;
    cgi430_config.Mode = 0;
    if(VCI_InitCAN(VCI_USBCAN2, 0, 0, &cgi430_config)!=1){
		printf(">>Init CAN1 error\n");
		VCI_CloseDevice(VCI_USBCAN2,0);
	}else{
        printf(">>Init CAN1 success!\n");
    }

    /*启动CAN卡的某一个CAN通道。有多个CAN通道时，需要多次调用。*/
    if(VCI_StartCAN(VCI_USBCAN2, 0, 0)!=1){
		printf(">>Start CAN1 error\n");
		VCI_CloseDevice(VCI_USBCAN2,0);
	}else{
        printf(">>Start CAN1 success!\n");
    }
    
}

void CGI430_CAN::ReceiveDate()
{
    std::cout<<">> >> Start receiveDate ..."<<std::endl;
    cgi430_rec_thread_ = std::thread(&CGI430_CAN::CGI430_Analyze_func, this);
    usleep(100000);
    cgi430_rec_thread_.join();  //线程阻塞，让调用线程等待子线程执行完毕，然后再往下执行

    usleep(100000);
    VCI_ResetCAN(VCI_USBCAN2, 0, 0);//复位CAN1通道
    usleep(100000);
	VCI_CloseDevice(VCI_USBCAN2,0);//关闭设备
}

void CGI430_CAN::CGI430_Analyze_func()
{
    while(1)
    {
        receive_length_ = VCI_Receive(VCI_USBCAN2, 0, 0, rec, len_, 0);  //接收函数，返回值为实际读取的帧数
        if(receive_length_ > 0)
        {   //std::cout<<"receive_length: "<<receive_length_<<std::endl;
            for(int i=0; i<receive_length_; i++)
            {   //printf("%#x\n", rec[i].ID, "\n");
                if(rec[i].ID == 0x321 || rec[i].ID == 0x322) 
                {   double angrateraw_x, angrateraw_y, angrateraw_z;
                    double accelraw_x,accelraw_y,accelraw_z;

                    // printf("%#x in 0x801 and 0x802 ", rec[i].ID, "\n");
                    if(rec[i].ID == 0x321)
                    {   //printf("%#x in \n", rec[i].ID, "\n");
                        // double angrateraw_x, angrateraw_y, angrateraw_z;
                        unsigned int signed_rawx = ((rec[i].Data[0] & 0x80) >> 7);
                        unsigned int signed_rawy = ((rec[i].Data[2] & 0x08) >> 3);
                        unsigned int signed_rawz = ((rec[i].Data[5] & 0x80) >> 7);
                        int flag_1 = 0xFFFFF;


                        if(signed_rawx == 0)
                        {
                            angrateraw_x = (((rec[i].Data[0] & 0xff) << 12) | ((rec[i].Data[1] & 0xff) << 4) | ((rec[i].Data[2] & 0xf0) >> 4))*0.01;
                        }else{
                            angrateraw_x = -(((((rec[i].Data[0] & 0xff) << 12) | ((rec[i].Data[1] & 0xff) << 4) | ((rec[i].Data[2] & 0xf0) >> 4))- 0x01)^flag_1)*0.01;
                        }
                        if(signed_rawy == 0)
                        {
                            angrateraw_y = (((rec[i].Data[2] & 0x0f) << 16) | ((rec[i].Data[3] & 0xff) << 8) | (rec[i].Data[4] & 0xff))*0.01;
                            // std::cout<<"angrateraw_y: "<<std::setprecision(15)<<(angrateraw_y /180.0)*M_PI<<std::endl;
                        }else{
                            angrateraw_y = -(((((rec[i].Data[2] & 0x0f) << 16) | ((rec[i].Data[3] & 0xff) << 8) | (rec[i].Data[4] & 0xff))- 0x01)^flag_1)*0.01;
                            // std::cout<<"angrateraw_y: "<<std::setprecision(15)<<(angrateraw_y /180.0)*M_PI<<std::endl;
                        }
                        if(signed_rawz ==0)
                        {
                            angrateraw_z = (((rec[i].Data[5] & 0xff) << 12) | ((rec[i].Data[6] & 0xff) << 4) | ((rec[i].Data[7] & 0xf0) >> 4))*0.01;
                        }else{
                            angrateraw_z = -(((((rec[i].Data[5] & 0xff) << 12) | ((rec[i].Data[6] & 0xff) << 4) | ((rec[i].Data[4] & 0xf0) >> 4))- 0x01)^flag_1)*0.01;
                        }

                        // imu_data.angular_velocity.x = (angrateraw_x /180.0)*M_PI;
                        // imu_data.angular_velocity.y = (angrateraw_y /180.0)*M_PI;
                        // imu_data.angular_velocity.z = (angrateraw_z /180.0)*M_PI;
                    }else{
    
                        // double accelraw_x,accelraw_y,accelraw_z;
                        unsigned int signed_x = ((rec[i].Data[0] & 0x80) >> 7);
                        unsigned int signed_y = ((rec[i].Data[2] & 0x08) >> 3);
                        unsigned int signed_z = ((rec[i].Data[5] & 0x80) >> 7);
                        int flag_2 = 0xFFFFF;

                        if(signed_x == 0)
                        {
                            accelraw_x = (((rec[i].Data[0] & 0xff) << 12) | ((rec[i].Data[1] & 0xff) << 4) | ((rec[i].Data[2] & 0xf0) >> 4))*0.0001;
                        }else{
                            accelraw_x = -((((((rec[i].Data[0] & 0xff) << 12) | ((rec[i].Data[1] & 0xff) << 4) | ((rec[i].Data[2] & 0xf0) >> 4))- 0x01)^flag_2)*0.0001);
                        }

                        if(signed_y == 0)
                        {
                            accelraw_y = (((rec[i].Data[2] & 0x0f) << 16) | ((rec[i].Data[3] & 0xff) << 8) | (rec[i].Data[4] & 0xff))*0.0001;
                        }else{
                            accelraw_y = -(((((rec[i].Data[2] & 0x0f) << 16) | ((rec[i].Data[3] & 0xff) << 8) | (rec[i].Data[4] & 0xff))- 0x01)^flag_2)*0.0001;
                        }

                        if(signed_z == 0)
                        {
                            accelraw_z = (((rec[i].Data[5] & 0xff) << 12) | ((rec[i].Data[6] & 0xff) << 4) | ((rec[i].Data[7] & 0xf0) >> 4))*0.0001;
                        }else{
                            accelraw_z = -(((((rec[i].Data[5] & 0xff) << 12) | ((rec[i].Data[6] & 0xff) << 4) | ((rec[i].Data[4] & 0xf0) >> 4))- 0x01)^flag_2)*0.0001;
                        }

                        // imu_data.linear_acceleration.x = accelraw_x*9.8;
                        // imu_data.linear_acceleration.y = accelraw_y*9.8;
                        // imu_data.linear_acceleration.z = accelraw_z*9.8;
                    }
                    imu_data.angular_velocity.x = (angrateraw_x /180.0)*M_PI;
                    imu_data.angular_velocity.y = (angrateraw_y /180.0)*M_PI;
                    imu_data.angular_velocity.z = (angrateraw_z /180.0)*M_PI;
                    imu_data.linear_acceleration.x = accelraw_x*9.8;
                    imu_data.linear_acceleration.y = accelraw_y*9.8;
                    imu_data.linear_acceleration.z = accelraw_z*9.8;
                    
                    imu_data.header.stamp = this->now();
                    imu_data.header.frame_id = "imu";
                    pub_imu_->publish(imu_data);

                }
                // std::cout<<"waht "<<std::endl;
            }    
        }else if(receive_length_ == -1){
            VCI_CloseDevice(VCI_USBCAN2,0);
            std::cout<<">> The CAN card disconnect,please check it!"<<std::endl;
            break;
        }else{
            // std::cout<<"receive_length: "<<receive_length_<<std::endl;
            if(!(rclcpp::ok())){
            break;
            }   
        }
        usleep(out_hz_);

        if(!(rclcpp::ok())){
            break;
        }
    }
}