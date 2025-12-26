#pragma once

#include <string>

namespace gpu_utils
{

enum class ComputeTarget
{
  GPU,
  DLA
};

enum class GpuPriority
{
  HIGH,
  LOW
};

struct InferenceAssignment
{
  ComputeTarget target{ComputeTarget::GPU};

  // DLA
  int dla_core{-1};  // valid if target == DLA

  // GPU
  GpuPriority gpu_priority{GpuPriority::LOW};  // valid if target == GPU

  std::string debug_string() const
  {
    if (target == ComputeTarget::DLA) {
      return "DLA(core=" + std::to_string(dla_core) + ")";
    } else {
      return gpu_priority == GpuPriority::HIGH ? "GPU(HIGH)" : "GPU(LOW)";
    }
  }
};

}  // namespace gpu_utils   