#include "ARS408_driver/ARS408_CAN_core.hpp"

Radar_408_60b radar_60b = {0,0,0,0,0,0,0};//dbc结构体
Radar_408_200 radar_200 = {0,0,0,0,0,0,0};//dbc结构体
Radar_408_202 radar_202 = {0,0,0,0,0,0};//dbc结构体
Radar_408_60d radar_60d = {0,0,0,0,0,0};

ARS408_CAN::ARS408_CAN(const std::string & node_name, const rclcpp::NodeOptions & node_options)
: rclcpp::Node(node_name, node_options)
{
    pub_object_ = create_publisher<radar_msgs::msg::RadarTrack>("~/input/radar_object", 10);
    pub_objects_ = create_publisher<radar_msgs::msg::RadarTracks>("~/input/radar_objects", 10);
    // markerArrayPub_ = create_publisher<visualization_msgs::msg::MarkerArray>("visualization_marker_radar", 10);

    init_param();
    CanPortInit();
    // Radar_cfg();
    ReceiveDate();
}

void ARS408_CAN::init_param()
{
    _out_object_topic = declare_parameter("out_object_topic", "/dev/ttyUSB0");
    RCLCPP_INFO(this->get_logger(),"[%s] the param out_object_topic value is... %s", "ARS408_CAN_node", _out_object_topic.c_str());

    out_hz_ = declare_parameter("out_hz", 1000);
    RCLCPP_INFO(this->get_logger(),"[%s] the param out_hz value is... %lf", "ARS408_CAN_node", out_hz_);

    Threshold_x_ = declare_parameter("Threshold_x", 150);
    RCLCPP_INFO(this->get_logger(),"[%s] the param Threshold_x_ value is... %d", "ARS408_CAN_node", Threshold_x_);

    Threshold_y_ = declare_parameter("Threshold_y", 4);
    RCLCPP_INFO(this->get_logger(),"[%s] the param Threshold_y_ value is... %d", "ARS408_CAN_node", Threshold_y_);
}

void ARS408_CAN::CanPortInit()
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
		exit(1);}

    /*VCI_InitCAN()　初始化指定ＣＡＮ通道，并检查是否初始化成功*/
    VCI_INIT_CONFIG ARS408_config;
    ARS408_config.AccCode = 0;          //
    ARS408_config.AccMask = 0xFFFFFFFF; 
    ARS408_config.Filter = 1;
    ARS408_config.Timing0 = 0x00;
    ARS408_config.Timing1 = 0x1c;
    ARS408_config.Mode = 0;
    if(VCI_InitCAN(VCI_USBCAN2, 0, 0, &ARS408_config)!=1){
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

void ARS408_CAN::Radar_cfg()
{
    while(Radar_cfg_can_send()==false){
	               if (Radar_cfg_can_send()==true){
	                        break;
	               }
	               printf("Radar继续配置.\n");
	               sleep(1);
	         }
	         
	         printf("Radar配置完成.\n");
	         return;
}

bool ARS408_CAN::Radar_cfg_can_send()
{
    radar_200.DATA0.bit.RadarCfg_MaxDistance_Valid = 1;//允许配置最大距离
    radar_200.DATA0.bit.RadarCfg_SensorID_Valid    = 1;//允许配置radarID
    radar_200.DATA0.bit.RadarCfg_RadarPower_Valid  = 1;//允许配置radar功率
    radar_200.DATA0.bit.RadarCfg_OutputType_Valid  = 1;//允许配置radar输出模式
    radar_200.DATA0.bit.RadarCfg_SendQuality_Valid = 1;//允许配置radar输出cluster和object的质量信息
    radar_200.DATA0.bit.RadarCfg_SendExtInfo_Valid = 1;//允许配置radar输出object的扩展信息
    radar_200.DATA0.bit.RadarCfg_SortIndex_Valid   = 1;//Object目标列表的当前排序索引值配置
    radar_200.DATA0.bit.RadarCfg_StoreInNVM_Valid  = 1;//使能；+++++++++++++++++++++++++++++++++
    //radar相关参数配置
    //最大距离设置
    radar_200.DATA12.bit.RadarCfg_MaxDistance = ((150 - 0)/2);//150为真实物理值：150m
    radar_200.DATA12.bit.reserved             =0;
    //保留位
    radar_200.DATA3.bit.reserved              =0;
    //设置radarID，输出类型，radar功率
    radar_200.DATA4.bit.RadarCfg_sensorID   = (0-0)/1;//radarID：为0
    radar_200.DATA4.bit.RadarCfg_OutputType =(1-0)/1;//object模式
    radar_200.DATA4.bit.RadarCfg_RadarPower =(1-0)/1;//标准发射功率
    //
    radar_200.DATA5.bit.RadarCfg_CtrlRelay_Valid = 0;
    radar_200.DATA5.bit.RadarCfg_CtrlRelay       = 0;
    radar_200.DATA5.bit.RadarCfg_SendQuality     = 1;
    radar_200.DATA5.bit.RadarCfg_SendExtInfo     = 1;
    radar_200.DATA5.bit.RadarCfg_SortIndex       = 1;//按距离排序输出
    radar_200.DATA5.bit.RadarCfg_StoreInNVM      = 1;//使能；+++++++++++++++++++++++++++++++++
    //
    radar_200.DATA6.bit.RadarCfg_RCS_threshold_Valid   = 1;
    radar_200.DATA6.bit.RadarCfg_RCS_threshold         = 0;//标准灵敏度
    radar_200.DATA6.bit.RadarCfg_InvalidClusters_Valid = 0;
    radar_200.DATA6.bit.reserved                       = 0;
    //
    radar_200.DATA7.bit.RadarCfg_InvalidClusters       = 0;

    vco[0].ID = 0x200;
    vco[0].RemoteFlag = 0;
    vco[0].ExternFlag = 0;
    vco[0].DataLen = 8;
    vco[0].Data[0] = radar_200.DATA0.D;
    vco[0].Data[2] = radar_200.DATA12.D;
    vco[0].Data[1] = radar_200.DATA12.D >> 8;
    vco[0].Data[3] = radar_200.DATA3.D;
    vco[0].Data[4] = radar_200.DATA4.D;
    vco[0].Data[5] = radar_200.DATA5.D;
    vco[0].Data[6] = radar_200.DATA6.D;
    vco[0].Data[7] = radar_200.DATA7.D;

    //radar过滤器配置----------
    vco[1].ID = 0x200;
    vco[1].RemoteFlag = 0;
    vco[1].ExternFlag = 0;
    vco[1].DataLen = 8;
    //过滤器使能
    radar_202.DATA0.bit.reserved         = 0;
    radar_202.DATA0.bit.FilterCfg_Valid  = 1;//使能过滤器  
    radar_202.DATA0.bit.FilterCfg_Active = 1;//激活过滤器
    radar_202.DATA0.bit.FilterCfg_Index  = 0x9;//筛选y方向的距离用过滤条件为:0x9(长度为12bit),0x5为
    radar_202.DATA0.bit.FilterCfg_Type   = 1;//object过滤器
    //最小距离设置
    radar_202.DATA12.bit.FilterCfg_Min_X =(-2 + 409.5)/0.2;//车头右边1.8米以外的目标筛选掉
    radar_202.DATA12.bit.reserved        = 0;
    //最大距离设置   
    radar_202.DATA34.bit.FilterCfg_Max_X =(2 + 409.5)/0.2;//车头左边1.8米以外的目标筛选掉
    radar_202.DATA34.bit.reserved        = 0;
    //
    radar_202.DATA5.bit.reserved = 0;
    radar_202.DATA6.bit.reserved = 0;
    radar_202.DATA7.bit.reserved = 0;
    
    vco[1].Data[0] = radar_202.DATA0.D;
    vco[1].Data[2] = radar_202.DATA12.D;
    vco[1].Data[1] = radar_202.DATA12.D >> 8;
    vco[1].Data[4] = radar_202.DATA34.D;
    vco[1].Data[3] = radar_202.DATA34.D >> 8;
    vco[1].Data[5] = radar_202.DATA5.D;
    vco[1].Data[6] = radar_202.DATA6.D;
    vco[1].Data[7] = radar_202.DATA7.D;
       
    int send_code = VCI_Transmit(VCI_USBCAN2, 0, 0, vco, 1);
    if(send_code == -1){
        printf("Error\n!");
	    return false;
    }else{
        printf("send_tag:%d: \n", send_code);
        return true;
    }
}


void ARS408_CAN::ReceiveDate()
{
    radar_rec_thread_ = std::thread(&ARS408_CAN::ARS408_Analyze_func, this);
    usleep(100000);
    radar_rec_thread_.join();  //线程阻塞，让调用线程等待子线程执行完毕，然后再往下执行

    usleep(100000);
    VCI_ResetCAN(VCI_USBCAN2, 0, 0);//复位CAN1通道
    usleep(100000);
	VCI_CloseDevice(VCI_USBCAN2,0);//关闭设备

}

void ARS408_CAN::ARS408_Analyze_func()
{
    while(1)
    {
        receive_length_ = VCI_Receive(VCI_USBCAN2, 0, 0, rec, len_, 0);  //接收函数，返回值为实际读取的帧数
        if(receive_length_ > 0)
        {   //std::cout<<"receive_length: "<<receive_length_<<std::endl;
            for(int i=0; i<receive_length_; i++)
            {   //printf("%#x\n", rec[i].ID, "\n");
                if(rec[i].ID == 0x60b || rec[i].ID == 0x60d) 
                {   
                    // printf("%#x in 0x60b and 0x60d ", rec[i].ID, "\n");
                    if(rec[i].ID == 0x60b)
                    {   printf("%#x in \n", rec[i].ID, "\n");
                        radar_60b.DATA0.D  = rec[i].Data[0];
                        radar_60b.DATA12.D = (rec[i].Data[1] << 8) + rec[i].Data[2];
                        radar_60b.DATA23.D = (rec[i].Data[2] << 8) + rec[i].Data[3];
                        radar_60b.DATA45.D = (rec[i].Data[4] << 8) + rec[i].Data[5];
                        radar_60b.DATA56.D = (rec[i].Data[5] << 8) + rec[i].Data[6];

                        radar_datas.header.stamp = this->now();
                        radar_datas.header.frame_id = "radar_frame";
                        radar_data.uuid = radar_60b.DATA0.bit.obj_id;
                        radar_data.position.x = radar_60b.DATA12.bit.obj_long*0.2 - 500;
                        radar_data.position.y = radar_60b.DATA23.bit.obj_lat*0.2 - 204.6;
                        radar_data.velocity.x = radar_60b.DATA45.bit.obj_vlong*0.25 - 128;
                        radar_data.velocity.y = radar_60b.DATA56.bit.obj_vlat*0.25 - 64;

                        radar_datas.tracks.push_back(radar_data);

                    }else{
                        printf("%#x in \n", rec[i].ID, "\n");
                        radar_60d.DATA0.D = rec[i].Data[0];
                        radar_60d.DATA12.D = (rec[i].Data[1] << 8) + rec[i].Data[2];
                        radar_60d.DATA23.D = (rec[i].Data[2] << 8) + rec[i].Data[3];
                        radar_60d.DATA45.D = (rec[i].Data[4] << 8) + rec[i].Data[5];
                        radar_60d.DATA6.D = rec[i].Data[6];
                        radar_60d.DATA7.D = rec[i].Data[7];

                        radar_datas.header.stamp = this->now();
                        radar_datas.header.frame_id = "radar_frame";
                        radar_data.uuid = radar_60d.DATA0.bit.obj_id;
                        radar_data.size.x = radar_60d.DATA7.bit.obj_width*0.2;
                        radar_data.size.y = radar_60d.DATA6.bit.obj_length*0.2;
                        radar_datas.tracks.push_back(radar_data);
                    }
                    
                    pub_objects_->publish(radar_datas);

               

                        // int obj_ID = sensor_data.obj_id;
	                    // // visualization(sensor_data.x, sensor_data.y, sensor_data.vx, obj_id, markerArrayPub_);

                        // pub_object_->publish(sensor_data);
	                    // printf("Topic(sensorRawData:3)-----radar:目标ID%d---横向距离%f---纵向距离%f---横向速度%f---纵向速度%f\n", 
                        // sensor_data.obj_id, sensor_data.y, sensor_data.x, sensor_data.vy,sensor_data.vx); 
                        

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

// ARS408_CAN::~ARS408_CAN();