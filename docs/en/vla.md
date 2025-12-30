// ...existing code...
# VLA Pipeline Usage Instructions

The following usage example describes VLA model training, model conversion, and inference optimization. Refer to this process to perform high-performance training and inference for the VLA model.

The VLA model is trained and deployed based on [OmniDrive](https://github.com/NVlabs/OmniDrive). For the required Docker images and environment, contact the project maintainers.

## Model Training

The original model has been modified, so re-fine-tuning is required. If you only want to test inference performance, skip to the Model Inference section. Key modifications include parallel decoding and data preprocessing optimizations.

```bash
# Set environment
export CUDA_VISIBLE_DEVICES=0
export CUDA_ARCH=80
export TRT_HOME=./TensorRT-10.4.0.26
export LD_LIBRARY_PATH=${TRT_HOME}/lib/:$LD_LIBRARY_PATH

# Distributed training
./tools/dist_train.sh ./vla/projects/configs/OmniDrive/eva_base_tinyllama.py 1

# Distributed testing (format-only)
./tools/dist_test.sh ./projects/configs/OmniDrive/eva_base_tinyllama.py \
  ./vla/work_dirs/eva_base_tinyllama/iter_8790.pth 1 --format-only

# Evaluation
cd ./vla/evaluation
python3.8 eval_planning.py
```

> Note: To reduce inference time, parallel decoding and preprocessing optimizations are enabled during training.

## Model Conversion

The model was optimized by folding constant expressions to eliminate redundant computations. Subsequently, the vision component was quantized to INT8 using ONNX, while the LLM component utilized W4A16 quantization. The detailed workflow is as follows:

Since the pth file uses a model trained using the perf training framework, it needs to be modified. Execute the following model conversion command.
```
python3.8 ./deploy/merge_lora.py --input_pth ./vla/work_dirs/eva_base_tinyllama/iter_8790.pth --output_pth ./vla/work_dirs/eva_base_tinyllama/iter_8790.pth

PYTHONPATH="./":$PYTHONPATH python3.8 ./deploy/save_llm_checkpoint.py --config ./projects/configs/OmniDrive/eva_base_tinyllama.py --checkpoint ./vla/work_dirs/eva_base_tinyllama/iter_8790.pth --llm_checkpoint ./vla/ckpts/finetune-1.1b-llava-tiny-llama-eva640/ --save_checkpoint_pth ./vla/llm_ckpts/

LD_LIBRARY_PATH=${TRT_HOME}/lib/:$LD_LIBRARY_PATH python3.8 ./deploy/convert_llm_checkpoint.py --model_dir ./llm_ckpts --output_dir ./llm_safetensor/x86_1gpu_fp16/ --dtype float16 --use_weight_only --weight_only_precision int4
```

For visual ONNX files, the usage process is as follows:
```
PYTHONPATH="./":$PYTHONPATH python3.8 ./deploy/export_vision.py ./projects/configs/OmniDrive/eva_base_tinyllama.py ./vla/work_dirs/eva_base_tinyllama/iter_8790.pth

python -m modelopt.onnx.quantization \
    --onnx_path=eva_base_tinyllama.onnx \
    --quantize_mode=int8 \
    --calibration_data=calib.npy \
    --calibration_method=max \
    --output_path=eva_base_tinyllama_delete_img2lidar_quant_max.onnx  
```


> Note: The model conversion process includes modifications to the model and model quantization.:
> - Model quantization
> - ONNX optimization


## Model Inference

Model inference is deployed on the Jetson platform for trajectory planning and latency benchmarking. This requires specific model weights, datasets and a pre-configured Docker environment — contact the project maintainers to obtain these assets.

Required items:
- Exported model weights (visual + LLM)
- Inference dataset and config
- Pre-built Docker image  with TensorRT and TensorRT-LLM





Run (example):
```bash

/usr/src/tensorrt/targets/aarch64-linux-gnu/bin/trtexec   --onnx=eva_base_tinyllama_delete_img2lidar_quant_max.onnx   --saveEngine=eva_base_tinyllama_delete_img2lidar_quant_max.engine --verbose --best --useCudaGraph --directIO --avgTiming=10 --iterations=100 --useSpinWait --profilingVerbosity=detailed --dumpLayerInfo --dumpProfile

LD_LIBRARY_PATH=${TRT_HOME}/lib/:$LD_LIBRARY_PATH trtllm-build --checkpoint_dir ./x86_1gpu_w4a16/ --output_dir ./llm_engine/x86_1gpu_w4a16/ --max_prompt_embedding_table_size 1024 --max_batch_size 1 --max_multimodal_len 2048 --gemm_plugin float16 --remove_input_padding enable --context_fmha enable --paged_kv_cache enable --enable_debug_output

# from inside the inference environment / container
python3 trtllm_test.py
```

If there are two Jetson platforms, asynchronous pipelined inference can be used for testing, with communication based on ZMQ.

> Note: Edge inference was optimized for lower latency using:
> - dual SoCs for asynchronous pipeline parallelism
> - visual token pruning
> - CUDA Graph
> - Inference framework optimization
> - Compiler option optimization