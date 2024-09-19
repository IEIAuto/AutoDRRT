import os
import mmcv 
import argparse
import numpy as np
from ruamel import yaml


def parse_args():
    parser = argparse.ArgumentParser(description='Export Onnx Model')
    parser.add_argument('--data-pkl',
                        default='data/nuscenes/bevdetv2-nuscenes_infos_val.pkl', 
                        help='deploy config file path')
    parser.add_argument('--work_dir',
                        default='deploy_out/data_infos/',
                        help='work dir to save file')
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
        
    data = mmcv.load(args.data_pkl, file_format='pkl')
    
    infos_path = os.path.join(args.work_dir, 'samples_info')
    if not os.path.exists(infos_path):
        os.makedirs(infos_path)
    
    samples_info = data['infos']
    prog_bar = mmcv.ProgressBar(len(samples_info))
    for i in range(len(samples_info)):
        sample = fixup(samples_info[i])
        sample_file = os.path.join(infos_path, 'sample{:04d}.yaml'.format(i))
        f = open(sample_file, 'w', encoding='utf-8')
        yaml.dump(sample, f, Dumper=yaml.Dumper, indent=4)
        prog_bar.update()

    print('')
    save_infos = []
    for i in range(len(samples_info)):
        info = samples_info[i]
        info.pop('ann_infos')
        info.pop('gt_boxes')
        info.pop('gt_names')
        info.pop('gt_velocity')
        info.pop('num_lidar_pts')
        info.pop('num_radar_pts')
        info.pop('valid_flag')
        
        save_infos.append(info)
    
    
    f = open(os.path.join(args.work_dir, 'samples_info.yaml'), 'w', encoding='utf-8')
    yaml.dump(fixup(save_infos), f, Dumper=yaml.Dumper, indent=4)
    
    
    data_infos = list(sorted(data['infos'], key=lambda e: e['timestamp']))
    timestamps = np.array([info['timestamp'] for info in data['infos']])
    
    time_sequence = {'time_sequence' : timestamps.argsort().tolist()}
    f = open(os.path.join(args.work_dir, 'time_sequence.yaml'), 'w', encoding='utf-8')
    yaml.dump(time_sequence, f, Dumper=yaml.Dumper, indent=4)
    
