#!/bin/bash

source ./utils.sh

check_i2c_device() {
    local bus=$1
    local addr=$2
    i2cdetect -y $bus | grep -q "${addr:2}"
    if [ $? -eq 0 ]; then
        log_info "I2C 设备 $addr 在位"
        return 0
    else
        log_error "I2C 设备 $addr 不在位"
        return 1
    fi
}

send_i2c_command() {
    local cmd=$1
    eval $cmd
    if [ $? -eq 0 ]; then
        log_info "$cmd 执行成功"
        return 0
    else
        log_error "$cmd 执行失败"
        return 1
    fi
}