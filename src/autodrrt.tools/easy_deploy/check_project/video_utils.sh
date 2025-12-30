#!/bin/bash

source ./utils.sh

FRAME_COUNT=10  # 可调帧数

check_video_devices() {
    for i in {0..7}; do
        check_device_exists "/dev/video$i"
    done
}

test_video_streaming() {
    for i in {0..7}; do
        local dev="/dev/video$i"
        if [ -e "$dev" ]; then
            log_info "测试视频设备 $dev 推流能力..."

            # 输出支持的格式（用于调试）
            v4l2-ctl --device="$dev" --list-formats-ext >> /tmp/stream_test.log

            # 尝试抓取帧
            v4l2-ctl --device="$dev" --stream-mmap --stream-count=$FRAME_COUNT --stream-to=/dev/null &>> /tmp/stream_test.log
            if [ $? -eq 0 ]; then
                log_info "$dev 推流成功"
            else
                log_error "$dev 推流失败，详情见 /tmp/stream_test.log"
            fi
        fi
    done
}

select_and_apply_camera_driver() {
    local driver_dir="./camera_drivers"
    local boot_dir="/boot"
    local choices=()
    local index=1

    if [ ! -d "$driver_dir" ]; then
        log_error "未找到 camera_drivers 目录: $driver_dir"
        return 1
    fi

    log_info "检测到以下可用相机驱动："
    for dir in "$driver_dir"/*/; do
        [ -d "$dir" ] || continue
        driver=$(basename "$dir")
        echo "  [$index] $driver"
        choices+=("$driver")
        ((index++))
    done

    if [ ${#choices[@]} -eq 0 ]; then
        log_error "未检测到任何可用相机驱动"
        return 1
    fi

    echo -n "请输入要使用的驱动编号 [1-${#choices[@]}]: "
    read -r selection

    if ! [[ "$selection" =~ ^[0-9]+$ ]] || [ "$selection" -lt 1 ] || [ "$selection" -gt "${#choices[@]}" ]; then
        log_error "无效输入"
        return 1
    fi

    local chosen="${choices[$((selection-1))]}"
    local chosen_path="$driver_dir/$chosen"

    log_info "你选择了相机驱动: $chosen"

    if [ ! -f "$chosen_path/Image" ] || [ ! -f "$chosen_path/"tegra234-*.dtb ]; then
        log_error "$chosen 驱动文件不完整（缺少 Image 或 dtb）"
        return 1
    fi

    log_info "正在替换系统驱动..."

    sudo cp "$chosen_path/Image" "$boot_dir/Image"
    sudo cp "$chosen_path/"tegra234-*.dtb "$boot_dir/"

    log_info "驱动替换完成。请重启系统以生效。"
    return 0
}
