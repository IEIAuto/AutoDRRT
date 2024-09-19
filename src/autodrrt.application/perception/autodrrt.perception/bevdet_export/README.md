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