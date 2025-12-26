#include "gpu_utils/inference_scheduler.hpp"

namespace gpu_utils
{

InferenceScheduler & InferenceScheduler::instance()
{
  static InferenceScheduler inst;
  return inst;
}

InferenceAssignment InferenceScheduler::assign(const ModelProfile & profile)
{
  std::lock_guard<std::mutex> lock(mtx_);

  InferenceAssignment result;

  // ---------- 优先尝试 DLA ----------
  if (profile.can_run_on_dla && !dla0_in_use_) {
    result.target = ComputeTarget::DLA;
    result.dla_core = 0;
    dla0_in_use_ = true;
    return result;
  }

  // ---------- 回退 GPU ----------
  result.target = ComputeTarget::GPU;

  if (profile.is_critical) {
    result.gpu_priority = GpuPriority::HIGH;
  } else {
    result.gpu_priority = GpuPriority::LOW;
  }

  return result;
}

}  // namespace gpu_utils