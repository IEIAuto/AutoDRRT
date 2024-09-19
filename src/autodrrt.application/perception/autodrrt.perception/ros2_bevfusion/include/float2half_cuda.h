#ifndef __FLOAT2HALFCUDA_H__
#define __FLOAT2HALFCUDA_H__

#include <cuda_fp16.h>
#include <common.h>

void float2half_cuda(const float* d_input, half* d_output, int size);

#endif

