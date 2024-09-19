# BEV Pipeline Usage Instructions

The following usage example introduces the process of model training, model conversion, and model inference optimization for the Bevdet model. You can refer to this process to complete high-performance training and inference optimization of the BEV model.


## Model Training
1. Change to the working directory `src/autodrrt.application/perception/autodrrt.perception/bevdet_train`. Prepare your own custom data.
   ```python3 tools/create_data_custom.py```
2. Modify the configuration file and perform model training
   ```./tools/dist_train.sh ./configs/bevdet/bevdet-r50-custom-data.py gpu_num ```
> Note: In order to reduce the inference time, model sparsification is used during training.

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
  ```ros2 launch ros2_bevdet bevdet.launch.xml```
> Note: In order to reduce the inference time, Cuda kernel optimization is used during model inference.