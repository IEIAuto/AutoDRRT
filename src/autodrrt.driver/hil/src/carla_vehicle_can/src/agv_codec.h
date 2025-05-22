#pragma once

#include "agv.h"



int encode_throttle_command(uint8_t* buf, size_t bufsize, double value, int en_ctrl=1);

int encode_brake_command(uint8_t* buf, size_t bufsize, double value, int en_ctrl=1);

int encode_steer_command(uint8_t* buf, size_t bufsize, double value, int en_ctrl=1);

int encode_gear_command(uint8_t* buf, size_t bufsize, int value);



struct ecu_status {
    int ctrl_sts;
    double speed;
    double acc;
};

int decode_ecu_status_1(const uint8_t *buffer, size_t buffer_size, ecu_status &obj);

int decode_steer_status(const uint8_t *buffer, size_t buffer_size, double &value, int& en_sts);
int decode_throttle_status(const uint8_t *buffer, size_t buffer_size, double &value, int& en_sts);
int decode_brake_status(const uint8_t *buffer, size_t buffer_size, double &value, int& en_sts);
int decode_gear_status(const uint8_t *buffer, size_t buffer_size, int &value);