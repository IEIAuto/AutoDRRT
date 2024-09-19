# A fork of BEVDet

<div align="center">

[English](README.md) | 简体中文

</div>


本仓库提供了导出 [bevdet-tensorrt-cpp](https://github.com/LCH1238/bevdet-tensorrt-cpp/tree/master) 所需要的onnx模型的脚本。

## 环境
请参考 [BEVDet](https://github.com/HuangJunJie2017/BEVDet/tree/dev2.1) 与 [mmdetection3d](https://github.com/open-mmlab/mmdetection3d)。
## 运行
导出onnx模型
```shell
python tools/export/export_onnx.py configs/bevdet/bevdet-r50-4dlongterm-depth-cbgs.py <checkpoint> --postfix='_lt_d' 
```
提供了去除BEV encoder部分的preprocess模块，并采用nearest采样预处理微调过的 [checkpoint](https://drive.google.com/drive/folders/1jSGT0PhKOmW3fibp6fvlJ7EY6mIBVv6i?usp=drive_link)。

导出 [bevdet-tensorrt-cpp](https://github.com/LCH1238/bevdet-tensorrt-cpp/tree/master) 所需要的yaml文件
```shell
python tools/export/export_yaml.py configs/bevdet/bevdet-r50-4dlongterm-depth-cbgs.py --prefix='bevdet_lt_d'
```

根据生成的boxes来评测
```shell
python tools/export/test.py configs/bevdet/bevdet-r50-4dlongterm-depth-cbgs.py path_to_results
```