#pragma once
#include <cuda_runtime.h>
#include <mutex>

class GPUStreamManager
{
public:
  static GPUStreamManager & instance()
  {
    static GPUStreamManager s;
    return s;
  }

  cudaStream_t & high()
  {
    init();
    return high_stream_;
  }

  cudaStream_t & low()
  {
    init();
    return low_stream_;
  }

private:
  bool initialized_ = false;
  cudaStream_t high_stream_;
  cudaStream_t low_stream_;
  std::mutex mtx_;

  void init()
  {
    std::lock_guard<std::mutex> lock(mtx_);
    if (initialized_) return;

    int lowPrio, highPrio;
    cudaDeviceGetStreamPriorityRange(&lowPrio, &highPrio);

    cudaStreamCreateWithPriority(&high_stream_, cudaStreamDefault, highPrio);
    cudaStreamCreateWithPriority(&low_stream_, cudaStreamDefault, lowPrio);

    initialized_ = true;
  }
};