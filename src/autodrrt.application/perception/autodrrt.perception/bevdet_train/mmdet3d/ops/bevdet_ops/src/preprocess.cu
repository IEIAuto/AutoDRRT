#include "common.h"

template<typename T>
__global__ void preprocess_kernel(const __uint8_t * src_dev, 
                                T* dst_dev, 
                                int src_row_step, 
                                int dst_row_step, 
                                int src_img_step, 
                                int dst_img_step,
                                int src_h, 
                                int src_w, 
                                float radio_h, 
                                float radio_w, 
                                float offset_h, 
                                float offset_w, 
                                const float * mean, 
                                const float * std,
                                int dst_h,
                                int dst_w,
                                int n){

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= dst_h * dst_w * n) return;
    
    int i = (idx / n) / dst_w;
    int j = (idx / n) % dst_w;
    int k = idx % n;

	int pX = (int) roundf((i / radio_h) + offset_h);
	int pY = (int) roundf((j / radio_w) + offset_w);
 
	if (pX < src_h && pX >= 0 && pY < src_w && pY >= 0){
        int s1 = k * src_img_step + 0 * src_img_step / 3 + pX * src_row_step + pY;
        int s2 = k * src_img_step + 1 * src_img_step / 3 + pX * src_row_step + pY;
        int s3 = k * src_img_step + 2 * src_img_step / 3 + pX * src_row_step + pY;

        int d1 = k * dst_img_step + 0 * dst_img_step / 3 + i * dst_row_step + j;
        int d2 = k * dst_img_step + 1 * dst_img_step / 3 + i * dst_row_step + j;
        int d3 = k * dst_img_step + 2 * dst_img_step / 3 + i * dst_row_step + j;

		dst_dev[d1] = (static_cast<T>(src_dev[s1]) - static_cast<T>(mean[0])) / static_cast<T>(std[0]);
		dst_dev[d2] = (static_cast<T>(src_dev[s2]) - static_cast<T>(mean[1])) / static_cast<T>(std[1]);
		dst_dev[d3] = (static_cast<T>(src_dev[s3]) - static_cast<T>(mean[2])) / static_cast<T>(std[2]);
	}
}


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
                float resize_radio
                 ){
    int src_row_step = src_img_w;
    int dst_row_step = dst_img_w;

    int src_img_step = src_img_w * src_img_h * 3;
    int dst_img_step = dst_img_w * dst_img_h * 3;

    float offset_h = 1.f * crop_h / resize_radio;
    float offset_w = 1.f * crop_w / resize_radio;

    dim3 grid(DIVUP(dst_img_h * dst_img_w * n_img,  NUM_THREADS));
    dim3 block(NUM_THREADS);

    preprocess_kernel<<<grid, block>>>(reinterpret_cast<const __uint8_t *>(src_imgs),
                                        reinterpret_cast<float *>(dst_imgs),
                                        src_row_step, 
                                        dst_row_step, 
                                        src_img_step,
                                        dst_img_step, 
                                        src_img_h, 
                                        src_img_w, 
                                        resize_radio,
                                        resize_radio, 
                                        offset_h, 
                                        offset_w, 
                                        reinterpret_cast<const float *>(mean), 
                                        reinterpret_cast<const float *>(std),
                                        dst_img_h, 
                                        dst_img_w,
                                        n_img);
}