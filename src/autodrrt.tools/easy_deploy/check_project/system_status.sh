#!/bin/bash

# 引入通用函数
source ./utils.sh

check_jetpack_version() {
    if [ -f /etc/nv_tegra_release ]; then
        # 读取第一行内容
        local line
        line=$(head -n 1 /etc/nv_tegra_release)

        # 提取第一行中的 major 部分，去掉前面的 "R"
        local major
        major=$(echo "$line" | awk '{print $2}' | sed 's/^R//')

        # 提取 REVISION 后面的数字部分，比如 "4.3"
        local revision
        revision=$(echo "$line" | sed -n 's/.*REVISION:[[:space:]]*\([0-9]\+\.[0-9]\+\).*/\1/p')

        if [[ -n "$major" && -n "$revision" ]]; then
            # 拼接成版本号，格式为 major.revision，即例如 "36.4.3"
            local version="${major}.${revision}"
            echo "L4T 版本: $version"
        else
            echo "无法解析 L4T 版本"
        fi
    else
        echo "未找到 /etc/nv_tegra_release 文件"
    fi
}

log_info "开始检测 Jetson 系统状态"

# ----------------------------
# CPU 使用率
log_info "CPU 使用率:"
top -bn1 | grep "Cpu(s)" | sed "s/.*, *\([0-9.]*\)%* id.*/\1/" | awk '{print "CPU 使用率：" 100 - $1 "%"}'

# ----------------------------
# 内存使用情况
log_info "内存使用情况:"
free -h | grep Mem | awk '{print "已用/总内存：" $3 "/" $2}'

# ----------------------------
# 磁盘空间
log_info "磁盘空间:"
df -h | grep -E "^/dev/mmcblk0p|^/dev/sda" | awk '{print $1, $3 "/" $2, $5}'

# ----------------------------
# 网络状态
log_info "网络状态:"
ip -br addr | grep -E 'eth0|wlan0' | awk '{print $1 ": " $3}'

# ----------------------------
# 系统温度
log_info "系统温度:"
cat /sys/class/thermal/thermal_zone0/temp | awk '{printf("系统温度：%.1f°C\n", $1/1000)}'

# ----------------------------
# GPU 使用率
log_info "GPU 使用率:"
tegrastats --interval 1000 --count 1 | grep "GR3D" | awk '{print "GPU 使用率：" $3}'

# ----------------------------
# JetPack 版本
log_info "JetPack 版本:"
check_jetpack_version

# ----------------------------
# CUDA 版本
log_info "CUDA 版本:"
if command -v nvcc &>/dev/null; then
    nvcc --version | grep "release" | sed 's/.*release //' | awk '{print "CUDA 版本：" $1}'
else
    log_warn "nvcc 命令未安装，CUDA 可能未安装或未正确配置"
fi

log_info "Jetson 系统状态检测完成"