#ifndef ARS408_DRIVER__ARS408_CAN_CORE_HPP_
#define ARS408_DRIVER__ARS408_CAN_CORE_HPP_

#include <iostream>
#include <thread>
#include <string>
#include <ctime>
#include <unistd.h>

#include <rclcpp/rclcpp.hpp>
// #include "radar_msgs/msg/sensordata.hpp"
#include "radar_msgs/msg/radar_track.hpp"
#include "radar_msgs/msg/radar_tracks.hpp"


#include "ARS408_driver/controlcan.h"


class ARS408_CAN : public rclcpp::Node
{
public:
    ARS408_CAN(const std::string & node_name, const rclcpp::NodeOptions & node_options);
    ~ARS408_CAN() = default;

private:
    rclcpp::Publisher<radar_msgs::msg::RadarTracks>::SharedPtr pub_objects_;
    rclcpp::Publisher<radar_msgs::msg::RadarTrack>::SharedPtr pub_object_;
	radar_msgs::msg::RadarTrack  radar_data;
    radar_msgs::msg::RadarTracks  radar_datas;

    double out_hz_;
    int Threshold_x_, Threshold_y_;

    bool once1 = true;

private:
    std::string _out_object_topic;

    //CAN卡相关
    VCI_BOARD_INFO pInfo1[4];          //用来获取设备信息

    int receive_length_;
    VCI_CAN_OBJ rec[50];  //CAN帧结构体
	VCI_CAN_OBJ vco[2];
    int len_=200;           //设定用来接收的帧结构体数组长度

    std::thread radar_rec_thread_;

private:
    void init_param();
    void CanPortInit();
    void Radar_cfg();
	bool Radar_cfg_can_send();
    void ReceiveDate();
    void ARS408_Analyze_func();
    // void visualization(float & obj_x, float & obj_y, float & obj_Vx, int obj_ID, ros::Publisher & markerArrayPub_);

};


struct Radar_408_60b
{
	union
	{
		uint8_t D;
		struct
		{
			uint8_t Length : 4;
			uint8_t reserved : 4;
		} bit;
	} MsgLengh;

	union
	{
		uint32_t D;
		struct
		{
			uint32_t ID : 32;
		} bit;
	} MsgID;
	
	
	
	//--------用到的如下
	union
	{
		uint8_t D;
		struct
		{

			uint8_t obj_id : 8;

		} bit;
	} DATA0;

	union
	{
		uint16_t D;
		struct
		{
			uint16_t reserved : 3;
			uint16_t obj_long : 13;

		} bit;
	} DATA12;
	union
	{
		uint16_t D;
		struct
		{
			uint16_t obj_lat : 11;
			uint16_t reserved : 5;
		} bit;
	} DATA23;
	union
	{
		uint16_t D;
		struct
		{
			uint16_t reserved : 6;
			uint16_t obj_vlong : 10;

		} bit;
	} DATA45;

	union
	{
		uint16_t D;
		struct
		{
			uint16_t obj_dynprop : 3;//在C结构体中，如果一个位域A在另一个位域B之前定义，那么位域A将存储在比B小的位地址中
			uint16_t reserved1 : 2;
			uint16_t obj_vlat : 9;
			uint16_t reserved2 : 2;

		} bit;
	} DATA56;
	
   union
	{
		int16_t D;
		struct
		{
			int16_t rcs : 16;
		} bit;
	} DATA7;
};

struct Radar_408_60d
{
	union
	{
		uint8_t D;
		struct
		{
			uint8_t obj_id : 8;
		} bit;
	} DATA0;

	union
	{
		uint16_t D;
		struct
		{
			uint16_t reserved : 5;
			uint16_t obj_long : 11;
		} bit;
	} DATA12;

	union
	{
		uint16_t D;
		struct
		{
			uint16_t obj_class : 3;
			uint16_t reserved1 : 1;
			uint16_t obj_arelat : 9;
			uint16_t reserved2 : 3;
		} bit;
	} DATA23;

	union
	{
		uint16_t D;
		struct
		{
			uint16_t reserved : 6;
			uint16_t obj_orientationangle : 10;
		} bit;
	} DATA45;

	union
	{
		uint8_t D;
		struct
		{
			uint8_t obj_length : 8;
		} bit;
	} DATA6;
	
   union
	{
		uint8_t D;
		struct
		{
			int8_t obj_width : 8;
		} bit;
	} DATA7;
};

//配置报文-----------0x200
struct Radar_408_200
{  
      //如果要更改雷达的配置，对应的以下有效性要配置的为有效；
      union{
            uint8_t D;
   	      struct
		      {
			   uint8_t RadarCfg_MaxDistance_Valid:1;//0x0无效，0x1有效
			   uint8_t RadarCfg_SensorID_Valid:1;//0x0无效，0x1有效
			   uint8_t RadarCfg_RadarPower_Valid:1;//0x0无效，0x1有效
			   uint8_t RadarCfg_OutputType_Valid:1;//0x0无效，0x1有效
		   	uint8_t RadarCfg_SendQuality_Valid:1;//0x0无效，0x1有效
			   uint8_t RadarCfg_SendExtInfo_Valid:1;//0x0无效，0x1有效
			   uint8_t RadarCfg_SortIndex_Valid:1;
			   uint8_t RadarCfg_StoreInNVM_Valid:1;
		      } bit;
      }DATA0;

		union{
				uint16_t D;
		      struct
		      {
			   uint16_t reserved:6;
			   uint16_t RadarCfg_MaxDistance:10;			   
		      } bit;
		}DATA12;

		union{
				uint8_t D;
				struct
		      {
			   uint8_t reserved:8;			   
		      } bit;				
		}DATA3;
		
		union{
				uint8_t D;
		      struct
		      {
			   uint8_t RadarCfg_sensorID:3;
			   uint8_t RadarCfg_OutputType:2;
			   uint8_t RadarCfg_RadarPower:3;
		      } bit;
		}DATA4;
		
		union{
				uint8_t D;
		      struct
		      {
			   uint8_t RadarCfg_CtrlRelay_Valid:1;
			   uint8_t RadarCfg_CtrlRelay:1;
			   uint8_t RadarCfg_SendQuality:1;
			   uint8_t RadarCfg_SendExtInfo:1;
			   uint8_t RadarCfg_SortIndex:3;
			   uint8_t RadarCfg_StoreInNVM:1;
		      } bit;
		}DATA5;
		
		union{
				uint8_t D;
		      struct
		      {
			   uint8_t RadarCfg_RCS_threshold_Valid:1;
			   uint8_t RadarCfg_RCS_threshold:3;
			   uint8_t RadarCfg_InvalidClusters_Valid:1;
			   uint8_t reserved:3;
		      } bit;
		}DATA6;
		
		union{
				uint8_t D;
		      struct
		      {
			   uint8_t RadarCfg_InvalidClusters:8;
		      } bit;
		}DATA7;	
};

//radar过滤器配置
struct Radar_408_202
{
		union{
				uint8_t D;
		      struct
		      {
			   uint8_t reserved:1;
			   uint8_t FilterCfg_Valid:1;
			   uint8_t FilterCfg_Active:1;
			   uint8_t FilterCfg_Index:4;
			   uint8_t FilterCfg_Type:1;
		      } bit;
		}DATA0;	
		
		union{
				uint16_t D;
		      struct
		      {
			   uint16_t FilterCfg_Min_X:13;
			   uint16_t reserved:3;
		      } bit;
		}DATA12;	
		
		union{
				uint16_t D;
		      struct
		      {
			   uint16_t FilterCfg_Max_X:13;
			   uint16_t reserved:3;
		      } bit;
		}DATA34;			
		
		union{
				uint8_t D;
		      struct
		      {
			   uint8_t reserved:8;
		      } bit;
		}DATA5;			
		
		union{
				uint8_t D;
		      struct
		      {
			   uint8_t reserved:8;
		      } bit;
		}DATA6;			
		
		union{
				uint8_t D;
		      struct
		      {
			   uint8_t reserved:8;
		      } bit;
		}DATA7;			
		
};

#endif