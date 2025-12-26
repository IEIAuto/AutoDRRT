## comm_lib overview

This repository is a **ROS2 + CUDA + POSIX shared memory** sample project for high performance data transfer.
It demonstrates how to keep ROS2 nodes loosely coupled while using shared memory + CUDA mapped memory to transfer large data (e.g. GPU images) efficiently across processes.

- Transport: ROS2 only sends a small “ready” signal (`UInt64` timestamp), bulk data goes through POSIX shared memory

Main components:
- `producer`: generates image data on GPU, writes it into shared memory, and publishes a ROS2 “ready” signal
- `consumer`: receives the signal, reads data from shared memory, maps/copies it to GPU for processing and can save images to disk
- `cuda_ipc_api`: convenience API wrapping publish/subscribe (`CudaIpcPublisher` / `CudaIpcSubscriber`)

## Run producer / consumer demo

## Basic API usage idea

In your own ROS2 nodes you can use the `cuda_ipc_api` library to hide the low‑level shared‑memory/CUDA details.

- As a publisher:
  - create an `rclcpp::Node`
  - construct a `CudaIpcPublisher`
  - call `publish()` or `publish_direct()` with CPU/GPU buffers

- As a subscriber:
  - create an `rclcpp::Node`
  - construct a `CudaIpcSubscriber`
  - register CPU/GPU callbacks (`set_cpu_callback`, `set_gpu_callback`)
  - call `start()` and then `rclcpp::spin(node)`

For complete examples, see:
- `src/example_api_usage.cu`
- `src/publisher_node_example.cu`
- `src/subscriber_node_example.cu`


## CPU–CPU / CPU–GPU / GPU–GPU modes

The `CudaIpcPublisher` / `CudaIpcSubscriber` pair can be used in three typical data‑flow modes:

- **CPU → CPU**  
  - Publisher: data lives in host memory, call  
    `publish(cpu_ptr, size, DataLocation::CPU);`  
  - Subscriber: register a CPU callback with `set_cpu_callback(...)`, you receive a plain host pointer and can process on CPU or copy to GPU yourself.

- **CPU → GPU**  
  - Publisher: still publishes from host memory, same as above:  
    `publish(cpu_ptr, size, DataLocation::CPU);`  
  - Subscriber: register a GPU callback with `set_gpu_callback(...)`; the library maps/copies shared memory into a device buffer and passes a device pointer into your callback so you can launch CUDA kernels directly.

- **GPU → GPU**  
  - Publisher: data is already on GPU, call  
    `publish(gpu_ptr, size, DataLocation::GPU);`  
    or use the zero‑copy style:
    1. `void* gpu_buf = publisher.get_gpu_buffer(size);`
    2. launch your kernel to write into `gpu_buf`
    3. `publisher.publish_direct(size, ...)`  
  - Subscriber: same as CPU→GPU case, use `set_gpu_callback(...)` and consume the device pointer; no round‑trip through your own CPU code is required (the library handles any needed host/device mapping internally).


## ROS2 Launch Examples

After building the project, you can launch publisher and subscriber nodes in different modes using `ros2 run`. Make sure to source the workspace first:

```bash
source install/setup.bash
```

### CPU → CPU Mode

**Terminal 1 (Publisher):**
```bash
ros2 run comm_lib publisher_node_example --location CPU --size 8388608 --rate 10
```

**Terminal 2 (Subscriber):**
```bash
ros2 run comm_lib subscriber_node_example --mode CPU
```

The publisher sends data from CPU memory, and the subscriber receives it as a CPU pointer for host-side processing.

### CPU → GPU Mode

**Terminal 1 (Publisher):**
```bash
ros2 run comm_lib publisher_node_example --location CPU --size 8388608 --rate 10
```

**Terminal 2 (Subscriber):**
```bash
ros2 run comm_lib subscriber_node_example --mode GPU
```

The publisher sends data from CPU memory, and the subscriber receives it as a GPU device pointer (the library handles the shared memory → GPU mapping/copy internally).

### GPU → GPU Mode

**Terminal 1 (Publisher):**
```bash
ros2 run comm_lib publisher_node_example --location GPU --size 8388608 --rate 10
```

Or use zero-copy direct mode (recommended for best performance):
```bash
ros2 run comm_lib publisher_node_example --direct --size 8388608 --rate 10
```

**Terminal 2 (Subscriber):**
```bash
ros2 run comm_lib subscriber_node_example --mode GPU
```

Both publisher and subscriber work with GPU memory. In zero-copy mode (`--direct`), data is generated directly in mapped shared memory, avoiding GPU→GPU copies.

### Additional Options

**Publisher options:**
- `--size <bytes>`: data size in bytes (default: 8MB)
- `--location CPU|GPU`: data location (default: CPU)
- `--direct` or `-d`: enable zero-copy direct mode (forces GPU location)
- `--rate <hz>` or `--hz <hz>`: publish rate in Hz (default: 10.0)
- `--count <n>`: maximum number of messages to publish (0 = infinite)

**Subscriber options:**
- `--mode CPU|GPU|BOTH`: callback mode (default: CPU)
  - `CPU`: only register CPU callback
  - `GPU`: only register GPU callback  
  - `BOTH`: register both callbacks
- `--stats`: enable detailed timing statistics

