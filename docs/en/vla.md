# VLA Pipeline Usage Instructions

The following usage example introduces the process of VLA model training, model conversion, and model inference optimization for the VLA model. You can refer to this process to complete high-performance training and inference optimization of the VLA model.

The VLA model is primarily trained and deployed based on [OmniDrive](https://github.com/NVlabs/OmniDrive). For information regarding the model training and inference environment for this project, please contact us via email to obtain the relevant Docker images.


## Model Training
1. Change to the working directory `src/autodrrt.application/perception/autodrrt.perception/vla/`. Prepare your own custom data.
2. Modify the configuration file and perform model training
   ```./tools/dist_train.sh ./projets/config/OmniDrive/eva_base_tinyllama.py```
> Note: In order to reduce the inference time, parallel decoding is used during training.

## Model Conversion
1. Change to the working directory `src/autodrrt.computing/realtime/pth_qdq`. Perform model ptq quantification.
   ```python pth2qdq.py configs_path pth_path --eval map```
2. Convert pth file to onnx model.
   ```python export_onnx_quntization.py configs_path pth_path --work_dir=./ --postfix='_lt_d'```
3. Generate the final onnx file and cache data.
   ```python qdq_translator.py --input_onnx_models=img_stage_lt_d.onnx --output_dir=./ --infer_concat_scales --infer_mul_scales```
> Note: In order to reduce the inference time, model quantization is used during model conversion.

## Inference Optimization
1. Change to the working directory `src/autodrrt.application/perception/autodrrt.perception/ros2_bevdet`. Generate engine model and perform inference.
   ```trtexec --onnx=img_stage_lt_d.onnx --saveEngine=img_stage_lt_d.engine --int8 --fp16 --calib=img_stage_lt_d_precision_config_calib.cache --sparsity```
   ```trtexec --onnx=bev_stage_lt_d.onnx --saveEngine=bev_stage_lt_d.engine --int8 --fp16 --calib=bev_stage_lt_d_precision_config_calib.cache --sparsity```

2. Model Inference. 

  ```python3 trtllm_test.py```
> Note: When performing inference on the edge, we mainly optimized the TensorRT-LLM inference framework, used dual SOCs for asynchronous pipeline parallelism, implemented visual token pruning, and incorporated CudaGraph and operator fusion to optimize model inference. 