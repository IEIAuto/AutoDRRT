#ifndef __COMMON_H__
#define __COMMON_H__


#include <assert.h>
#include <stdlib.h>
#include <stdio.h>
#include <string>

#define NUM_THREADS 512 

#define DIVUP(m, n) (((m) + (n)-1) / (n))

#define CUDA_1D_KERNEL_LOOP(i, n) \
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < (n); i += blockDim.x * gridDim.x)

const int MAXTENSORDIMS = 6;
struct TensorDesc {
  int shape[MAXTENSORDIMS];
  int stride[MAXTENSORDIMS];
  int dim;
};

inline int GET_BLOCKS(const int N) {
  int optimal_block_num = DIVUP(N, NUM_THREADS);
  int max_block_num = 4096;
  return optimal_block_num < max_block_num ? optimal_block_num : max_block_num;
}



#endif