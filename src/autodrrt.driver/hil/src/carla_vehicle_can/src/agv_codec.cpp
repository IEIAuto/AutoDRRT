
#include "agv_codec.h"
int encode_throttle_command(uint8_t* buf, size_t bufsize, double value, int en_ctrl) {

    struct agv_throttle_command_t throttle_command;

    throttle_command.throttle_pedal_en_ctrl = en_ctrl;
    throttle_command.throttle_pedal_cmd = agv_throttle_command_throttle_pedal_cmd_encode(value);
    

    return agv_throttle_command_pack(buf, &throttle_command, bufsize);
}


int encode_brake_command(uint8_t* buf, size_t bufsize, double value, int en_ctrl) {

    struct agv_brake_command_t brake_command;

    brake_command.brake_pedal_en_ctrl = en_ctrl;
    brake_command.brake_pedal_cmd = agv_brake_command_brake_pedal_cmd_encode(value);
    

    return agv_brake_command_pack(buf, &brake_command, bufsize);
}


int encode_steer_command(uint8_t* buf, size_t bufsize, double value, int en_ctrl) {

    struct agv_steer_command_t steer_command;

    steer_command.steer_angle_en_ctrl = en_ctrl;
    steer_command.steer_angle_cmd = agv_steer_command_steer_angle_cmd_encode(value);

    return agv_steer_command_pack(buf, &steer_command, bufsize);
}


int encode_gear_command(uint8_t* buf, size_t bufsize, int value) {

    struct agv_gear_command_t gear_command;

    gear_command.gear_cmd = agv_gear_command_gear_cmd_encode(value);

    return agv_gear_command_pack(buf, &gear_command, bufsize);
}


int decode_ecu_status_1(const uint8_t *buffer, size_t buffer_size, ecu_status &obj) {
    
    agv_ecu_status_1_t s;
    int ret = agv_ecu_status_1_unpack(&s, buffer, buffer_size);

    obj.speed = agv_ecu_status_1_speed_decode(s.speed);
    obj.acc = agv_ecu_status_1_acc_speed_decode(s.acc_speed);
    obj.ctrl_sts = agv_ecu_status_1_ctrl_sts_decode(s.ctrl_sts);
    return ret;
}


int decode_steer_status(const uint8_t *buffer, size_t buffer_size, double &value, int &en_sts) {
    
    agv_steer_status__t s;
    int ret = agv_steer_status__unpack(&s, buffer, buffer_size);

    value = agv_steer_status__steer_angle_sts_decode(s.steer_angle_sts);
    en_sts = agv_steer_status__steer_angle_en_sts_decode(s.steer_angle_en_sts);

    return ret;
}
int decode_throttle_status(const uint8_t *buffer, size_t buffer_size, double &value, int &en_sts) {
    
    agv_throttle_status__t s;
    int ret = agv_throttle_status__unpack(&s, buffer, buffer_size);

    value = agv_throttle_status__throttle_pedal_sts_decode(s.throttle_pedal_sts);
    en_sts = agv_throttle_status__throttle_pedal_en_sts_decode(s.throttle_pedal_en_sts);

    return ret;
}

int decode_brake_status(const uint8_t *buffer, size_t buffer_size, double &value, int &en_sts) {
    
    agv_brake_status__t s;
    int ret = agv_brake_status__unpack(&s, buffer, buffer_size);

    value = agv_brake_status__brake_pedal_sts_decode(s.brake_pedal_sts);
    en_sts = agv_brake_status__brake_pedal_en_sts_decode(s.brake_pedal_en_sts);

    return ret;
}
int decode_gear_status(const uint8_t *buffer, size_t buffer_size, int &value) {
    
    agv_gear_status_t s;
    int ret = agv_gear_status_unpack(&s, buffer, buffer_size);

    value = agv_gear_status_gear_sts_decode(s.gear_sts);

    return ret;
}