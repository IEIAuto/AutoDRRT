import os
import argparse
import numpy as np
import torch
from tools.misc.fuse_conv_bn import fuse_module 

from mmcv import Config
from mmcv.runner import load_checkpoint
import mmcv
try:
    # If mmdet version > 2.23.0, compat_cfg would be imported and
    # used from mmdet instead of mmdet3d.
    from mmdet.utils import compat_cfg
except ImportError:
    from mmdet3d.utils import compat_cfg

from mmdet3d.models import build_model




def parse_args():
    parser = argparse.ArgumentParser(description='Export Onnx Model')
    parser.add_argument('config', help='deploy config file path')
    parser.add_argument('--work_dir', default='deploy_out/', help='work dir to save file')
    parser.add_argument(
        '--prefix', default='bevdet', help='prefix of the save file name')
    args = parser.parse_args()
    return args

def fixup(x):
    if type(x) == tuple:
        x = list(x)
    if type(x) in [list, dict, mmcv.utils.config.ConfigDict] and len(x) > 0:
        if type(x) == list:
            for i, data in enumerate(x):
                x[i] = fixup(data)
            return x
        if type(x) == mmcv.utils.config.ConfigDict:
            x = dict(x)
        for k in x.keys():
            x[k] = fixup(x[k])
    elif type(x) == np.ndarray:
        x = x.tolist()
    return x


if __name__ == '__main__':
    args = parse_args()
    if not os.path.exists(args.work_dir):
        os.makedirs(args.work_dir)
    
    cfg = Config.fromfile(args.config)
    cfg.model.pretrained = None
    cfg = compat_cfg(cfg)

   
    img_H, img_W = cfg.data_config['input_size']
    downsample_factor = cfg.model.img_view_transformer.downsample
    feat_w, feat_h = img_H // downsample_factor, img_W // downsample_factor
    D = len(torch.arange(*cfg.grid_config['depth']))
    
    bev_h = len(torch.arange(*cfg.grid_config['x']))
    bev_w = len(torch.arange(*cfg.grid_config['y']))
    bev_inchannels = cfg.model.img_view_transformer.out_channels
    
    
    tasks = cfg.model.pts_bbox_head.tasks
    common_heads = cfg.model.pts_bbox_head.common_heads
    class_num_pre_task = [len(t['class_names']) for t in tasks]
    channel_num_pre_head = [v[0] for _, v in common_heads.items()]
    names_pre_head = [k for k, _ in common_heads.items()]
    names_pre_head.append('heatmap')
    channel_num_all_heads = [channel_num_pre_head + [c] for c in class_num_pre_task]
    output_dim = [[1, c, bev_h, bev_w] for t in channel_num_all_heads for c in t]
    
    bev_output_names = [f'{name}_{i}' for i in range(len(tasks)) for name in names_pre_head]
    
    
    yaml_dict = {}
    yaml_dict['bev_range'] = cfg.point_cloud_range
    yaml_dict['data_config'] = dict(cfg.data_config)
    yaml_dict['data_config'].pop('crop_h')
    yaml_dict['data_config']['crop'] = [140, 0]
    
    yaml_dict['grid_config'] = dict(cfg.grid_config)
    yaml_dict['test_cfg'] = dict(cfg.model.test_cfg)

    model = {}
    model['tasks'] = cfg.model.pts_bbox_head.tasks
    model['bevpool_channels'] = cfg.model.img_view_transformer.out_channels
    model['down_sample'] = downsample_factor
    model['coder'] = dict(cfg.model.pts_bbox_head.bbox_coder)
    # model['output'] = dict(map(tuple, zip(bev_output_names, output_dim)))
    model['common_head'] =  {'names' : [k for k in common_heads.keys()], 'channels': [v[0] for v in common_heads.values()]}
   
    
    yaml_dict['model'] = model

    fixup(yaml_dict)

    yaml_dict['model']['coder'].pop('type')
    yaml_dict['model']['coder'].pop('voxel_size')
    yaml_dict['model']['coder'].pop('out_size_factor')
    yaml_dict['model']['coder'].pop('pc_range')
    
    yaml_dict['test_cfg'] = yaml_dict['test_cfg']['pts']
    yaml_dict['test_cfg'].pop('voxel_size')
    yaml_dict['test_cfg'].pop('out_size_factor')
    yaml_dict['test_cfg'].pop('pc_range')
    # yaml_dict['test_cfg']['nms_rescale_factor'] = 
    nms_rescale_factor = yaml_dict['test_cfg']['nms_rescale_factor']
    nms_rescale_factor = [[factor] if type(factor) == float else factor 
                                            for factor in nms_rescale_factor]
    yaml_dict['test_cfg']['nms_rescale_factor'] = nms_rescale_factor
    
    yaml_dict['mean'] = [123.675, 116.28, 103.53]
    yaml_dict['std'] = [58.395, 57.12, 57.375]
    yaml_dict['use_depth'] = 'Depth' in cfg.model.img_view_transformer.type
    yaml_dict['use_adj'] = cfg.multi_adj_frame_id_cfg[1] is not 1
    yaml_dict['adj_num'] = cfg.multi_adj_frame_id_cfg[1] - 1
    yaml_dict['sampling'] = 'nearest'
   
    from ruamel import yaml
    
    
    f = open(f'{args.work_dir}/{args.prefix}.yaml', 'w', encoding='utf-8')
    yaml.dump(yaml_dict, f, Dumper=yaml.Dumper, indent=4)
    print(yaml_dict)
    
