# A fork of BEVDet

<div align="center">

English | [简体中文](README_zh-CN.md)

</div>

This repository provides the script for exporting onnx required by [bevdet-tensorrt-cpp](https://github.com/LCH1238/bevdet-tensorrt-cpp/tree/master).

## Environment
Please refer to [BEVDet](https://github.com/HuangJunJie2017/BEVDet/tree/dev2.1) and [mmdetection3d](https://github.com/open-mmlab/mmdetection3d).
## Running
export onnx
```shell
python tools/export/export_onnx.py configs/bevdet/bevdet-r50-4dlongterm-depth-cbgs.py <checkpoint> --postfix='_lt_d' 
```
This repository provides [checkpoint](https://drive.google.com/drive/folders/1jSGT0PhKOmW3fibp6fvlJ7EY6mIBVv6i?usp=drive_link).

export yaml required by [bevdet-tensorrt-cpp](https://github.com/LCH1238/bevdet-tensorrt-cpp/tree/master) 
```shell
python tools/export/export_yaml.py configs/bevdet/bevdet-r50-4dlongterm-depth-cbgs.py --prefix='bevdet_lt_d'
```

test
```shell
python tools/export/test.py configs/bevdet/bevdet-r50-4dlongterm-depth-cbgs.py path_to_results
```

python3 tools/test_kunyi.py /home/liry/swpld/BEVDet-export/configs/bevdet/bevdet-export-official.py /home/liry/swpld/BEVDet-export/work_dirs/bevdet-export-official/epoch_20_ema.pth --format-only --eval-options jsonfile_prefix=./


python3 tools/test_kunyi.py /home/liry/swpld/BEVDet-export/configs/bevdet/kunyi_six_camera.py /home/liry/swpld/BEVDet-export/work_dirs/kunyi_six_camera/epoch_30_ema.pth --format-only --eval-options jsonfile_prefix=./




# qunt

python tools/test_qdq.py /home/liry/swpld/BEVDet-export/configs/bevdet/kunyi_six_camera.py ./work_dirs/kunyi_six_camera/epoch_20_ema.pth --eval map
python tools/export/export_onnx_qunt.py configs/bevdet/kunyi_six_camera.py ./qdq_model/model_ptq.pth --work_dir=./qdq_model/ --postfix='_lt_d'  | tee export_onnx.log
python tools/qdq_translator.py --input_onnx_models=./qdq_model/img_stage_lt_d.onnx --output_dir=./qdq_model/ --infer_concat_scales --infer_mul_scales | tee a.log
python tools/qdq_translator.py --input_onnx_models=./qdq_model/bev_stage_lt_d.onnx --output_dir=./qdq_model/ --infer_concat_scales --infer_mul_scales | tee a.log