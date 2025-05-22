#include <torch/torch.h>
#include <c10/cuda/CUDAGuard.h>

void preprocess(const void* src_imgs,
                void* dst_imgs,
                const void* mean,
                const void* std,
                int n_img,
                int src_img_h,
                int src_img_w,
                int dst_img_h,
                int dst_img_w,
                int crop_h,
                int crop_w,
                float resize_radio);

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
             int img_size);

void alignbev(const void* adj_feat,
              const void* transform,
              void* output,
              int bev_channel,
              int bev_h,
              int bev_w,
              int adj_num
              );

void gatherbev(const void* adj_feats,
               const void* curr_feats,
               const void* flags,
               void* output,
               int b,
               int adj_num,
               int map_size,
               int channel);

void preprocess_forward(const at::Tensor src_imgs,  // 6 x 3 x 900 x 1600
                        const at::Tensor mean,      
                        const at::Tensor _std,
                        at::Tensor dst_imgs,        // 6 x 3 x 256 * 704
                        float resize_radio,
                        int crop_h,
                        int crop_w){
    int n_img = src_imgs.size(0);
    int src_img_h = src_imgs.size(2);
    int src_img_w = src_imgs.size(3) * 4;

    int dst_img_h = dst_imgs.size(2);
    int dst_img_w = dst_imgs.size(3);

    const int* src_dev = src_imgs.data_ptr<int>();
    const float* mean_dev = mean.data_ptr<float>();
    const float* std_dev = _std.data_ptr<float>();
    float* out_dev = dst_imgs.data_ptr<float>();

    preprocess((const void*)src_dev,
               (void*)out_dev,
               (const void*)mean_dev,
               (const void*)std_dev,
               n_img,
               src_img_h,
               src_img_w,
               dst_img_h,
               dst_img_w,
               crop_h,
               crop_w,
               resize_radio
               );
}

void bevpool_forward(const at::Tensor _depth,         // b*n x d x h x w
                     const at::Tensor _feat,          // b*n x c x h x w 
                     const at::Tensor _ranks_depth,
                     const at::Tensor _ranks_feat,
                     const at::Tensor _ranks_bev,
                     const at::Tensor _interval_starts,
                     const at::Tensor _interval_lengths,
                     at::Tensor _out,
                     int bev_h,
                     int bev_w,
                     int n
                     ){
    int channel = _feat.size(1);
    int n_intervals = _interval_starts.size(0);
    int map_size = bev_h * bev_w;
    int img_size = _feat.size(2) * _feat.size(3);
    
    const float* depth = _depth.data_ptr<float>();
    const float* feat = _feat.data_ptr<float>();
    const int* ranks_depth = _ranks_depth.data_ptr<int>();
    const int* ranks_feat = _ranks_feat.data_ptr<int>();
    const int* ranks_bev = _ranks_bev.data_ptr<int>();
    const int* interval_starts = _interval_starts.data_ptr<int>();
    const int* interval_lengths = _interval_lengths.data_ptr<int>();

    float* out = _out.data_ptr<float>();

    bevpool((const void*)depth,
            (const void*)feat,
            (const void*)ranks_depth,
            (const void*)ranks_feat,
            (const void*)ranks_bev,
            (const void*)interval_starts,
            (const void*)interval_lengths,
            (void*)out,
            channel,
            n_intervals,
            map_size,
            img_size);
}

// adj_feat  b x 8 x 80 x 128 x 128
// transform b x 8 x 6
// out       b x 8 x 80 x 128 x 128
void alignbev_forward(const at::Tensor _adj_feat,
                      const at::Tensor _trans,
                      at::Tensor _out
                      ){
    int bev_channel = _adj_feat.size(2);
    int bev_h = _adj_feat.size(3);
    int bev_w = _adj_feat.size(4);
    int adj_num = _adj_feat.size(1);

    const float* adj_feat = _adj_feat.data_ptr<float>();
    const float* trans = _trans.data_ptr<float>();
    float* out = _out.data_ptr<float>();

    alignbev((const void*)adj_feat, 
             (const void*)trans,
             (void*)out,
             bev_channel,
             bev_h,
             bev_w,
             adj_num
             );
}


// adj_feat  : b x 8 x 80 x 128 x 128
// curr_feat : b x 80 x 128 x 128
// flag      : b x 1
void gatherbev_forward(const at::Tensor _adj_feat,
                       const at::Tensor _curr_feat,
                       const at::Tensor _flag,
                       at::Tensor _out){
    int b = _adj_feat.size(0);
    int adj_num = _adj_feat.size(1);
    int map_size = _adj_feat.size(3) * _adj_feat.size(4);
    int channel = _adj_feat.size(2);

    const float* adj_feat = _adj_feat.data_ptr<float>();
    const float* curr_feat = _curr_feat.data_ptr<float>();
    const int* flag = _flag.data_ptr<int>();

    float* out = _out.data_ptr<float>();

    gatherbev((const void*)adj_feat,
              (const void*)curr_feat,
              (const void*)flag,
              (void*)out,
              b,
              adj_num,
              map_size,
              channel);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("preprocess_forward", &preprocess_forward, "preprocess_forward");
    m.def("bevpool_forward", &bevpool_forward, "bevpool_forward");
    m.def("alignbev_forward", &alignbev_forward, "alignbev_forward");
    m.def("gatherbev_forward", &gatherbev_forward, "gatherbev_forward");
}

