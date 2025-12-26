#pragma once
#include <unordered_set>
#include <sched.h>
#include <pthread.h>


void set_rt_properties(int prio, const std::unordered_set<size_t> & affinity)
{
  struct sched_param sched_param = { 0 };
  sched_param.sched_priority = prio;
  sched_setscheduler(0, SCHED_RR, &sched_param);

  cpu_set_t cpuset;
  CPU_ZERO(&cpuset);
  for (const auto cpu : affinity) {
    CPU_SET(cpu, &cpuset);
  }
  sched_setaffinity(0, sizeof(cpuset), &cpuset);
}