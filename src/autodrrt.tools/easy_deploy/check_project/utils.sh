#!/bin/bash

log_info()  { echo "[INFO] $1"; }
log_warn()  { echo "[WARN] $1"; }
log_error() { echo "[ERROR] $1"; }

check_device_exists() {
    local device=$1
    if [ -e "$device" ]; then
        log_info "$device 存在"
        return 0
    else
        log_warn "$device 不存在"
        return 1
    fi
}