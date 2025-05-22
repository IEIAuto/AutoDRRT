#include "common.h"


template<typename T1, typename T2>
__global__ void bev_pool_v2_kernel(int channel, 
                                    int n_intervals,
                                    int map_size,
                                    int img_size,
                                    const T1 *__restrict__ depth,
                                    const T1 *__restrict__ feat,
                                    const int *__restrict__ ranks_depth,
                                    const int *__restrict__ ranks_feat,
                                    const int *__restrict__ ranks_bev,
                                    const int *__restrict__ interval_starts,
                                    const int *__restrict__ interval_lengths,
                                    T2 * __restrict__ out) {
    CUDA_1D_KERNEL_LOOP(idx, n_intervals * channel){
        int index = idx / channel;    // bev grid index
        int curr_c = idx % channel;    // channel index
        int interval_start = interval_starts[index];  
        int interval_length = interval_lengths[index];  

        int curr_step = curr_c * img_size;
        int chan_step = channel * img_size;

        T2 sum = 0;

        int feat_offset = 0;
        for(int i = 0; i < interval_length; i++){
            feat_offset = ranks_feat[interval_start + i] / img_size * chan_step + 
                          curr_step + ranks_feat[interval_start + i] % img_size;
  
            sum += static_cast<T2>(feat[feat_offset]) * static_cast<T2>(depth[ranks_depth[interval_start + i]]);
        }
        out[curr_c * map_size + ranks_bev[interval_start]] = sum;
    }
}


void bevpool(const void* depth,
             const void* feat,
             const void* ranks_depth,
             const void* ranks_feat,
             const void* ranks_bev,
             const void* interval_starts,
             const void* interval_lengths,
             void* output,
             int channel,
             int n_intervals,
             int map_size,
             int img_size){
    // input[0] == depth            b*n x d x h x w
    // input[1] == feat             b*n x c x h x w
    // input[2] == ranks_depth      m
    // input[3] == ranks_feat       m
    // input[4] == ranks_bev        m
    // input[5] == interval_starts  k
    // input[6] == interval_lengths k

    dim3 grid(GET_BLOCKS(n_intervals * channel));
    dim3 block(NUM_THREADS);

    bev_pool_v2_kernel<float, float><<<grid, block>>>(
                                                channel, 
                                                n_intervals,
                                                map_size,
                                                img_size,
                                                reinterpret_cast<const float *>(depth),
                                                reinterpret_cast<const float *>(feat),
                                                reinterpret_cast<const int *>(ranks_depth),
                                                reinterpret_cast<const int *>(ranks_feat),
                                                reinterpret_cast<const int *>(ranks_bev),
                                                reinterpret_cast<const int *>(interval_starts),
                                                reinterpret_cast<const int *>(interval_lengths),
                                                reinterpret_cast<float *>(output));
}