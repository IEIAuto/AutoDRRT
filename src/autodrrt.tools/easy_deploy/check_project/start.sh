#!/bin/bash

# 引入通用函数和模块
source ./utils.sh
source ./i2c_utils.sh
source ./dmesg_utils.sh
source ./video_utils.sh
source ./system_status.sh

# 默认执行所有功能
DO_ALL=true

# 解析命令行参数
while getopts "i:d:v:s:h" opt; do
    case "$opt" in
        i) DO_I2C=true ;;         # 检查 I2C
        d) DO_DMESG=true ;;       # 检查 dmesg
        v) DO_VIDEO=true ;;       # 检查 video 设备
        s) DO_SYSTEM=true ;;      # 检查系统状态
        c) DO_CAMERA=true ;;      # 更换相机驱动
        h)
            echo "Usage: $0 [-i] [-d] [-v] [-s]"
            echo "    -i: 检查 I2C 设备"
            echo "    -d: 检查 dmesg 日志"
            echo "    -v: 检查 video 设备"
            echo "    -s: 检查系统状态"
            echo "    -c: 更换相机驱动"
            exit 0 ;;
        *) echo "Invalid option"; exit 1 ;;
    esac
done

# 如果没有指定任何选项，默认执行所有
[ "$DO_I2C" != true ] && [ "$DO_DMESG" != true ] && [ "$DO_VIDEO" != true ] && [ "$DO_SYSTEM" != true ] && DO_ALL=true

# 开始执行模块
log_info "开始执行检查"

if [ "$DO_I2C" == true ] || [ "$DO_ALL" == true ]; then
    log_info "开始 I2C 设备检查"
    check_i2c_device 1 0x74 || exit 1
    check_i2c_device 1 0x70 || exit 1
    send_i2c_command "i2ctransfer -f -y 1 w2@0x74 0x81 0x07" || exit 1
    send_i2c_command "i2ctransfer -f -y 1 w1@0x70 0x01" || exit 1
fi

if [ "$DO_DMESG" == true ] || [ "$DO_ALL" == true ]; then
    log_info "检查 dmesg 中关键词"
    check_dmesg_keyword "max9295"
    check_dmesg_keyword "max9296"
fi

if [ "$DO_VIDEO" == true ] || [ "$DO_ALL" == true ]; then
    log_info "检查视频设备"
    check_video_devices
    test_video_streaming
fi

if [ "$DO_SYSTEM" == true ] || [ "$DO_ALL" == true ]; then
    log_info "检查系统状态"
    check_system_status
fi


if [ "$DO_CAMERA" == true ]; then
    select_and_apply_camera_driver
fi

log_info "检查完毕"
