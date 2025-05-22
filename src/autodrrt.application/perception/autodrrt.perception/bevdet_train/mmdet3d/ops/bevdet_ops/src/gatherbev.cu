#include "common.h"

template<typename T>
__global__ void copy_feat_kernel(int nthreads, // b * (adj_num + 1) * map_size
                                int adj_num,
                                int channel,
                                int map_size,
                                const T* adj_feats,
                                const T* curr_feat,
                                const int* flag,
                                T* out_feats){
    CUDA_1D_KERNEL_LOOP(idx, nthreads){
        int b = idx / ((adj_num + 1) * map_size);
        int n = (idx / map_size) % (adj_num + 1);
        int m = idx % map_size;

        int start = b * (adj_num + 1) * channel * map_size + n * channel * map_size + m;
        int end = start + channel * map_size;
        for(int i = start, c = 0; i < end; i += map_size, c++){
            if(flag[b] == 0 || n == 0){
                out_feats[i] = curr_feat[b * channel * map_size + c * map_size + m];
            }
            else{
                out_feats[i] = adj_feats[i - channel * map_size];
            }
        }
    }
}


void gatherbev(const void* adj_feats,
               const void* curr_feats,
               const void* flags,
               void* output,
               int b,
               int adj_num,
               int map_size,
               int channel){

    int nthreads = b * (adj_num + 1) * map_size;

    dim3 grid(GET_BLOCKS(nthreads));
    dim3 block(NUM_THREADS);

    copy_feat_kernel<<<grid, block>>>(nthreads,
                                    adj_num,
                                    channel,
                                    map_size,
                                    reinterpret_cast<const float*>(adj_feats),
                                    reinterpret_cast<const float*>(curr_feats),
                                    reinterpret_cast<const int*>(flags),
                                    reinterpret_cast<float*>(output));
}