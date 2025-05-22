#ifndef SHARED_GPU_MEMORY_HPP
#define SHARED_GPU_MEMORY_HPP

#include <cuda_runtime.h>
#include <iostream>

class SharedGPUMemory {
public:
  static SharedGPUMemory & get_instance() {
    static SharedGPUMemory instance;
    return instance;
  }

  void allocate_memory(size_t size) {
    if (cudaMalloc(&gpu_memory_, size) != cudaSuccess) {
      std::cerr << "Failed to allocate GPU memory." << std::endl;
    }
  }

  void * get_memory() const {
    return gpu_memory_;
  }

  void free_memory() {
    if (gpu_memory_) {
      cudaFree(gpu_memory_);
    }
  }

private:
  SharedGPUMemory() : gpu_memory_(nullptr) {}
  ~SharedGPUMemory() {
    free_memory();
  }

  void *gpu_memory_;
};

#endif
