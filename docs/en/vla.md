# VLA Pipeline Usage Instructions

The following usage example introduces the process of VLA model training, model conversion, and model inference optimization for the VLA model. You can refer to this process to complete high-performance training and inference optimization of the VLA model.

The VLA model is primarily trained and deployed based on [OmniDrive](https://github.com/NVlabs/OmniDrive). For information regarding the model training and inference environment for this project, please contact us via email to obtain the relevant Docker images.


## Model Training
1. Change to the working directory `src/autodrrt.application/perception/autodrrt.perception/vla/`. Prepare your own custom data.
2. Modify the configuration file and perform model training
   ```./tools/dist_train.sh ./projets/config/OmniDrive/eva_base_tinyllama.py```
> Note: In order to reduce the inference time, parallel decoding is used during training.

## Model Conversion
> Note: After generating the visual model and LLM model files, int8 quantization was performed on the transformer visual model, and compilation options were optimized for the LLM.

## Model Inference

  ```python3 trtllm_test.py```
> Note: When performing inference on the edge, we mainly optimized the TensorRT-LLM inference framework, used dual SOCs for asynchronous pipeline parallelism, implemented visual token pruning, and incorporated CudaGraph and operator fusion to optimize model inference. 