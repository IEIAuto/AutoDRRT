# Check Status Project

A comprehensive toolkit for monitoring, testing, and configuring Jetson devices.

## 📋 Overview

This project provides easy-to-deploy tools for:
- **Device Status Checking**: Monitor I2C devices, video devices, system logs, and system status
- **Camera Driver Management**: Switch between different camera drivers
- **Real-time Performance Testing**: Evaluate system real-time performance using cyclictest
- **System Configuration**: Set maximum clock frequencies and enable kernel tracing

## 🚀 Quick Start

### Installation

```bash
# Clone or download the project
cd easy_deploy

# Make scripts executable
chmod +x check_project/*.sh
chmod +x rt_test/*.sh
```

## 📦 Module Description

### 1. Check Project

A comprehensive device checking and configuration tool for Jetson platforms.

#### Features

- **I2C Device Checking**: Verify I2C device connectivity and send control commands
- **dmesg Log Analysis**: Check for specific keywords in kernel logs (e.g., max9295, max9296)
- **Video Device Detection**: Detect and test video devices (`/dev/video*`)
- **System Status Monitoring**: Gather system information
- **Camera Driver Management**: Switch between different camera drivers
- **Clock Frequency Configuration**: Set maximum clock frequencies for VI, ISP, and NVCSI
- **Camera Tracing**: Enable kernel tracing for camera debugging

#### Usage

```bash
cd orin_check_project

# Run all checks (default)
./start.sh

# Run specific checks
./start.sh -i          # Check I2C devices only
./start.sh -d          # Check dmesg logs only
./start.sh -v          # Check video devices only
./start.sh -s          # Check system status only
./start.sh -c          # Switch camera driver

# Combine multiple checks
./start.sh -i -v -s    # Check I2C, video devices, and system status
```

#### Camera Driver Switching

```bash
./start.sh -c
```

The script will display available camera drivers and prompt for selection:

```
[INFO] 检测到以下可用相机驱动：
  [1] Leopard-AR0233
  [2] Sensing-AR0233
  [3] Leopard-ISX031
请输入要使用的驱动编号 [1-3]:
```

#### Script Functions

| Script | Description |
|--------|-------------|
| `start.sh` | Main entry point, orchestrates all checks |
| `utils.sh` | Common logging and device detection functions |
| `i2c_utils.sh` | I2C-specific detection and control functions |
| `dmesg_utils.sh` | dmesg log analysis utilities |
| `video_utils.sh` | Video device detection and testing |
| `camera_trace.sh` | Enable kernel tracing for camera debugging |
| `set_max_clk.sh` | Set maximum clock frequencies for critical components |
| `system_status.sh` | System information gathering |

#### Configuration

Edit `start.sh` to customize I2C device addresses and commands:

```bash
# Example I2C checks
check_i2c_device 1 0x74
check_i2c_device 1 0x70
send_i2c_command "i2ctransfer -f -y 1 w2@0x74 0x81 0x07"
```

---

### 2. Jetson RT Test

Real-time performance testing tool using cyclictest for evaluating system latency.

#### Features

- **Cyclictest Integration**: Automated cyclictest execution
- **Stress Testing**: Concurrent stress load testing
- **Result Analysis**: Automatic latency analysis and evaluation
- **Visualization**: Generate plots from test results
- **Log Management**: Timestamped result logs

#### Prerequisites

```bash
# Install required packages
sudo apt update
sudo apt install rt-tests stress python3-matplotlib
```

#### Usage

```bash
cd jetson_rt_test

# Run real-time test (default 60 seconds)
./rt_test.sh

# The script will:
# 1. Start stress load (CPU, IO, memory)
# 2. Run cyclictest for specified duration
# 3. Save results to results/ directory
# 4. Analyze maximum latency
# 5. Provide real-time performance assessment
```

#### Test Parameters

Edit `rt_test.sh` to customize test duration:

```bash
DURATION=60  # Test duration in seconds
```

#### Result Analysis

Test results are saved with timestamps:
```
results/cyclictest_result_YYYYMMDD_HHMMSS.log
```

#### Visualization

Parse and visualize test results:

```bash
# Parse and plot results
python3 parse_and_plot.py results/cyclictest_result_YYYYMMDD_HHMMSS.log

# Plot will be saved to plot/ directory
```

#### Real-time Performance Standards

| Max Latency | Performance Level | Description |
|-------------|-------------------|-------------|
| < 100 μs | Excellent | Suitable for high-precision industrial control |
| < 1000 μs | Good | Meets most non-critical real-time requirements |
| 1-5 ms | Fair | Needs optimization, affected by interrupts/scheduler |
| > 5 ms | Poor | Not suitable for hard real-time control |

#### Jetson Optimization Recommendations

1. Enable PREEMPT-RT real-time kernel
2. Use `SCHED_FIFO` or `SCHED_RR` real-time scheduling policies
3. Disable CPU frequency scaling (cpu governor)
4. Configure `/dev/cpu_dma_latency` to avoid deep sleep
5. Use kernel boot parameters: `isolcpus`, `nohz_full`, `rcu_nocbs`
6. Pin interrupts and real-time threads to specific CPUs (`taskset`, `irqbalance`)
7. Disable unnecessary services or hardware (GPU, cameras, etc.)
8. Use `stress` tool for stress testing to evaluate system limits

For detailed cyclictest documentation, see `jetson_rt_test/readme.md`.

## 🔧 Advanced Usage

### Setting Maximum Clock Frequencies

```bash
cd orin_check_project
./set_max_clk.sh
```

This script locks and sets critical clocks (VI, ISP, NVCSI) to maximum frequency.

### Enabling Camera Tracing

```bash
cd orin_check_project
./camera_trace.sh
```

Enables kernel tracing and captures camera-related debug information.

### Custom I2C Commands

Edit `start.sh` to add custom I2C commands:

```bash
# Add custom I2C checks
check_i2c_device <bus> <address>
send_i2c_command "<i2c_command>"
```

## 📊 Output Examples

### I2C Check Output

```
[INFO] 开始 I2C 设备检查
[INFO] /dev/i2c-1 存在
[INFO] I2C device 0x74 on bus 1 is accessible
```

### Video Device Check Output

```
[INFO] 检查视频设备
[INFO] 检测到以下视频设备:
  /dev/video0
  /dev/video1
```

### Real-time Test Output

```
T: 0 ( 4850) P:99 I:1000 C:  59997 Min:      2 Act:    7 Avg:    6 Max:     191
T: 4 ( 4854) P:99 I:3000 C:  19298 Min:      2 Act:    4 Avg:  158 Max:    7151

Max Latency: 7151 μs
Performance Level: Fair
```
