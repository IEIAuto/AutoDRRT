## ros2_bevdet
1. onnx to engine 
```shell
python tools/export_engine.py cfgs/bevdet_lt_depth.yaml model/img_stage_lt_d.onnx model/bev_stage_lt_d.onnx --postfix="_lt_d_fp16" --fp16=True
```
2. edit launch and config 
  edit launch file bevdet.launch.xml and bevdet.param.yaml

3, launch 
  ros2 launch ros2_bevdet bevdet.launch.xml


## ros2_bevformer 
1. launch 
  ros2 launch ros2_bevformer bevformer.launch.xml