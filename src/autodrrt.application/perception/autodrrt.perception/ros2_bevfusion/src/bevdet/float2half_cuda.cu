#include "float2half_cuda.h"
#include <iostream>

__global__ void float2half_cuda_kernel(const float* __restrict__ input, half* __restrict__ output, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) { 
        output[idx] = __float2half(input[idx]);
    }
}

void float2half_cuda(const float* input, half* output, int size) {
    int threadsPerBlock = 256;
    int blocksPerGrid = (size + threadsPerBlock - 1) / threadsPerBlock;
    float2half_cuda_kernel<<<blocksPerGrid, threadsPerBlock>>>(input, output, size);
}

