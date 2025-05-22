#include "common.h"


static inline __device__ bool within_bounds_2d(int h, int w, int H, int W){
    return h >= 0 && h < H && w >= 0 && w < W;
}

template <typename T1, typename T2>
__global__ void align_bev_kernel(const int nthreads, const T1 *input,
                                       const float *trans, T2 *output,
                                       TensorDesc output_desc){
    int C = output_desc.shape[1];         // 80
    int out_H = output_desc.shape[2];     // 128
    int out_W = output_desc.shape[3];     // 128
    int out_sN = output_desc.stride[0];   // 80 * 128 * 128
    int out_sC = output_desc.stride[1];   // 128 * 128
    int out_sH = output_desc.stride[2];   // 128
    int out_sW = output_desc.stride[3];   // 1

    CUDA_1D_KERNEL_LOOP(index, nthreads){
        const int w = index % out_W;               //  j
        const int h = (index / out_W) % out_H;     //  i
        const int n = index / (out_H * out_W);     // batch

        float ix = trans[n * 6 + 0 * 3 + 0] * w + 
                   trans[n * 6 + 0 * 3 + 1] * h + 
                   trans[n * 6 + 0 * 3 + 2];
        float iy = trans[n * 6 + 1 * 3 + 0] * w + 
                   trans[n * 6 + 1 * 3 + 1] * h + 
                   trans[n * 6 + 1 * 3 + 2];

        // NE, NW, SE, SW point
        int ix_nw = static_cast<int>(::floor(ix));
        int iy_nw = static_cast<int>(::floor(iy));
        int ix_ne = ix_nw + 1;
        int iy_ne = iy_nw;
        int ix_sw = ix_nw;
        int iy_sw = iy_nw + 1;
        int ix_se = ix_nw + 1;
        int iy_se = iy_nw + 1;

        T2 nw = (ix_se - ix) * (iy_se - iy);
        T2 ne = (ix - ix_sw) * (iy_sw - iy);
        T2 sw = (ix_ne - ix) * (iy - iy_ne);
        T2 se = (ix - ix_nw) * (iy - iy_nw);

        // bilinear
        auto inp_ptr_NC = input + n * out_sN;
        auto out_ptr_NCHW = output + n * out_sN + h * out_sH + w * out_sW;
        for (int c = 0; c < C; ++c, inp_ptr_NC += out_sC, out_ptr_NCHW += out_sC){
            *out_ptr_NCHW = static_cast<T2>(0);
            if (within_bounds_2d(iy_nw, ix_nw, out_H, out_W)){
                *out_ptr_NCHW += static_cast<T2>(inp_ptr_NC[iy_nw * out_sH + ix_nw * out_sW]) * nw;
            }
            if (within_bounds_2d(iy_ne, ix_ne, out_H, out_W)){
                *out_ptr_NCHW += static_cast<T2>(inp_ptr_NC[iy_ne * out_sH + ix_ne * out_sW]) * ne;
            }
            if (within_bounds_2d(iy_sw, ix_sw, out_H, out_W)){
                *out_ptr_NCHW += static_cast<T2>(inp_ptr_NC[iy_sw * out_sH + ix_sw * out_sW]) * sw;
            }
            if (within_bounds_2d(iy_se, ix_se, out_H, out_W)){
                *out_ptr_NCHW += static_cast<T2>(inp_ptr_NC[iy_se * out_sH + ix_se * out_sW]) * se;
            }
        }
    }
}

void create_desc(const int *dims, int nb_dims, TensorDesc &desc){
    memcpy(&desc.shape[0], dims, sizeof(int) * nb_dims);
    desc.stride[nb_dims - 1] = 1;
    for (int i = nb_dims - 2; i >= 0; --i){
        desc.stride[i] = desc.stride[i + 1] * desc.shape[i + 1];
    }
}


void alignbev(const void* adj_feat,
              const void* transform,
              void* output,
              int bev_channel,
              int bev_h,
              int bev_w,
              int adj_num
              ){
    // inputs[0] == adj_feat  b x 8 x 80 x 128 x 128
    // inputs[1] == transform b x 8 x 6
    int output_dim[4] = {adj_num, bev_channel, bev_h, bev_w};
    TensorDesc output_desc;
    create_desc(output_dim, 4, output_desc);

    int count = 1;
    for (int i = 0; i < 4; ++i){
        if (i == 1){
            continue;
        }
        count *= output_desc.shape[i];
    }
    align_bev_kernel<float, float><<<GET_BLOCKS(count), NUM_THREADS>>>(
                                            count, 
                                            reinterpret_cast<const float *>(adj_feat), 
                                            reinterpret_cast<const float*>(transform), 
                                            reinterpret_cast<float *>(output), 
                                            output_desc);
}