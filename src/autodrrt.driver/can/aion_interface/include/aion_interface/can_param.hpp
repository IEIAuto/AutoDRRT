#ifndef AIONINTERFACE__CANPARAM_HPP_
#define AIONINTERFACE__CANPARAM_HPP_

#include <iostream>
#include <deque>

struct AION_ADCU_1
{
    union{
        uint16_t D;
        struct
        {
            uint16_t BrakeReq:16;
        }bit; 
    }DATA01;

    union{
        uint16_t D;
        struct
        {
            uint16_t LngCtrlReq:2;
            uint16_t AutoTrqWhlReq:14;
        }bit; 
    }DATA23;

    union{
        uint8_t D;
        struct
        {
            uint8_t reserved:2;
            uint8_t ParkingReqEPBVD:1;
            uint8_t ParkingReqToEPB:1;
            uint8_t GearLvlReqVD:1;
            uint8_t GearLvlReq:3;
        }bit;
    }DATA4;

    union{
        uint8_t D;
        struct
        {
            uint8_t reserved:8;
        }bit;
    }DATA5;

    union{
        uint8_t D;
        struct
        {
            uint8_t MsgCounter:4;
            uint8_t reserved:4;
        }bit;
    }DATA6;

    union{
        uint8_t D;
        struct
        {
            uint8_t Checksum:8;
        }bit;
    }DATA7;

};

struct AION_ADCU_2
{
    union{
        uint16_t D;
        struct
        {
            uint16_t SteerAngReq:16;
        }bit;

    }DATA01;

    union{
        uint16_t D;
        struct
        {
            uint16_t SteerAngSpdLimt:16;
        }bit;
        
    }DATA23;

    union{
        uint16_t D;
        struct 
        {
            uint16_t reserved:4;
            uint16_t LatCtrReq:1;   
            uint16_t SteerWhlTorqReq:11;
        }bit;
        
    }DATA45;
    
    union{
        uint8_t D;
        struct 
        {
            uint8_t MsgCounter:4;
            uint8_t reserved:4;   
        }bit;

    }DATA6;

    union{
        uint8_t D;
        struct 
        {
            uint8_t CheckSum:8;
        }bit;

    }DATA7;

};

struct SCU_Mode
{
    uint8_t Scu_LatctrMode;
    uint8_t Scu_LngctrMode;
}scu_mode;


#endif