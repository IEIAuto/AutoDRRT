# README

## 整体架构

```shell
orin_check_project/
├── start.sh        # 主脚本
├── utils.sh        # 通用日志/设备检测函数
├── i2c_utils.sh      # I2C 专用检测与控制
├── dmesg_utils.sh     # dmesg 相关工具函数
├── video_utils.sh     # /dev/video 检测函数
├── camera_trace.sh    # 启用内核跟踪并捕获摄像头相关的调试信息
├── set_max_clk.sh     # 锁定并设置一些关键时钟（VI、ISP、NVCSI）为最大频率
├── system_status.sh     # 获取系统信息
```

## 更换相机驱动示例用法

```bash
./start.sh -c
```

它将输出：

```
[INFO] 检测到以下可用相机驱动：
  [1] Leopard-AR0233
  [2] Sensing-AR0233
  [3] Leopard-ISX031
请输入要使用的驱动编号 [1-3]:
```