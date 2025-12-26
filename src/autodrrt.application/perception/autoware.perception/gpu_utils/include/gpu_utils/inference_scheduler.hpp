#pragma once

#include "gpu_utils/inference_assignment.hpp"
#include <mutex>
#include <string>

namespace gpu_utils
{

struct ModelProfile
{
  std::string name;
  bool can_run_on_dla{false};
  bool is_critical{false};  // e.g. CenterPoint = true
};

class InferenceScheduler
{
public:
  static InferenceScheduler & instance();

  InferenceAssignment assign(const ModelProfile & profile);

private:
  InferenceScheduler() = default;

  std::mutex mtx_;

  // 简化：只支持一个 DLA core
  bool dla0_in_use_{false};
};

}  // namespace gpu_utils