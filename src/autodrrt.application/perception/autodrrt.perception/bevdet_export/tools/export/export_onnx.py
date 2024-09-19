import os
import argparse

import torch
from tools.misc.fuse_conv_bn import fuse_module 

from mmcv import Config
from mmcv.runner import load_checkpoint

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
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument('--work_dir', default='deploy_out/', help='work dir to save file')
    parser.add_argument(
        '--postfix', default='', help='prefix of the save file name')
    parser.add_argument(
        '--fuse-conv-bn',
        type=bool,
        default=True,
        help='Whether to fuse conv and bn, this will slightly increase the inference speed'
    )
    parser.add_argument(
        '--gpu-id',
        type=int,
        default=0,
        help='Which gpu to be used'
    )
    args = parser.parse_args()
    return args

def add_prefix(file):
    x = torch.load(file)
    state = x['state_dict']

    flag = False
    state_keys = state.keys()
    for k in list(state_keys):
        prefix = k.split('.')[0]
        if prefix in ['img_backbone', 'img_neck', 'img_view_transformer']:
            state[f'img_encoder.{k}'] = state[k]
            state.pop(k)
            flag = True
        elif prefix in ['img_bev_encoder_backbone', 'img_bev_encoder_neck', 'pts_bbox_head', 'pre_process_net']:
            state[f'bev_encoder.{k}'] = state[k]
            state.pop(k)
            flag = True
    if flag == False:
        return file
    
    path = file.split('/')
    new_file = ''
    for i in range(len(path) - 1):
        new_file += f'{path[i]}/'
    new_file += f'new-{path[-1]}' 
    
    torch.save(x, new_file)
    return new_file


if __name__ == '__main__':
    args = parse_args()
    if not os.path.exists(args.work_dir):
        os.makedirs(args.work_dir)
    
    cfg = Config.fromfile(args.config)
    cfg.model.pretrained = None
    cfg.model.type = 'BEVModel'
    cfg = compat_cfg(cfg)
    cfg.gpu_ids = [args.gpu_id]

    
    use_depth = 'Depth' in cfg.model.img_view_transformer.type
    cfg.model.use_depth = use_depth
    
    cfg.model.train_cfg = None
    model = build_model(cfg.model, test_cfg=cfg.get('test_cfg'))
    
    checkpoint_file = add_prefix(args.checkpoint)
    load_checkpoint(model, checkpoint_file, map_location='cpu')
    
    if args.fuse_conv_bn:
        model = fuse_module(model)
        
    model.cuda()
    model.eval()
    
    image_stage_model = model.img_encoder
    bev_stage_model = model.bev_encoder
    
    img_H, img_W = cfg.data_config['input_size']
    downsample_factor = cfg.model.img_view_transformer.downsample
    feat_w, feat_h = img_H // downsample_factor, img_W // downsample_factor
    D = len(torch.arange(*cfg.grid_config['depth']))  
    
    bev_h = len(torch.arange(*cfg.grid_config['x']))
    bev_w = len(torch.arange(*cfg.grid_config['y']))
    bev_inchannels = cfg.model.img_bev_encoder_backbone.numC_input
    
    img_input = torch.rand([6, 3, img_H, img_W], dtype=torch.float, device=f'cuda:{args.gpu_id}')
    bev_input = torch.rand([1, bev_inchannels, bev_h, bev_w], dtype=torch.float, device=f'cuda:{args.gpu_id}')
    
    if use_depth:
        rot = torch.rand([1, 6, 3, 3], dtype=torch.float, device=f'cuda:{args.gpu_id}')
        tran = torch.rand([1, 6, 3], dtype=torch.float, device=f'cuda:{args.gpu_id}')
        intrin = torch.rand([1, 6, 3, 3], dtype=torch.float, device=f'cuda:{args.gpu_id}')
        post_rot = torch.rand([1, 6, 3, 3], dtype=torch.float, device=f'cuda:{args.gpu_id}')
        post_tran = torch.rand([1, 6, 3], dtype=torch.float, device=f'cuda:{args.gpu_id}')
        bda = torch.rand([1, 3, 3], dtype=torch.float, device=f'cuda:{args.gpu_id}')
        
    
    tasks = cfg.model.pts_bbox_head.tasks
    common_heads = cfg.model.pts_bbox_head.common_heads
    class_num_pre_task = [len(t['class_names']) for t in tasks]
    channel_num_pre_head = [v[0] for _, v in common_heads.items()]
    names_pre_head = [k for k, _ in common_heads.items()]
    names_pre_head.append('heatmap')
    channel_num_all_heads = [channel_num_pre_head + [c] for c in class_num_pre_task]
    output_dim = [[1, c, bev_h, bev_w] for t in channel_num_all_heads for c in t]
    
    bev_output_names = [f'{name}_{i}' for i in range(len(tasks)) for name in names_pre_head]
    
    input_names = ['images']
    if use_depth:
        input_names += ['rot', 'trans', 'intrin', 'post_rot', 'post_trans', 'bda']
        
        
    print(output_dim)
    print(bev_output_names)
    with torch.no_grad():
        torch.onnx.export(
            image_stage_model, 
            (img_input, rot, tran, intrin, post_rot, post_tran, bda) if use_depth else img_input,
            f'{args.work_dir}/img_stage{args.postfix}.onnx', 
            input_names=input_names, 
            output_names=['images_feat', 'depth'],
            opset_version=11,
        )
        
        torch.onnx.export(
            bev_stage_model,
            bev_input,
            f'{args.work_dir}/bev_stage{args.postfix}.onnx',
            input_names=['BEV_feat'],
            output_names=bev_output_names,
            opset_version=11,
        )