import argparse
import os
import torch
import mmcv
from mmcv import Config
from mmdet.utils import compat_cfg
from mmdet3d.core.bbox import LiDARInstance3DBoxes

from mmdet3d.datasets import build_dataset



def parse_args():
    parser = argparse.ArgumentParser(description='TRTCPP test (and eval) a model')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('path',
                        default="./results" ,
                        help='path to results')

    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    print(args.config)
    cfg = Config.fromfile(args.config)
    cfg = compat_cfg(cfg)

    dataset = build_dataset(cfg.data.test)

    result_files = []
    if os.path.exists(args.path):
        result_files = [f'{args.path}/bevdet_egoboxes_{i}.txt' 
                                for i in range(len(os.listdir(args.path)))]
    else:
        assert 0
    
    
    results = []
    prog_bar = mmcv.ProgressBar(len(result_files))
    
    for file in result_files:
        boxes = []
        scores = []
        class_ids = []
        with open(file, 'r', encoding='utf-8') as f:
            data = f.readlines()
            for line in data:
                box = []
                line = line.strip().split(' ')
                for i in range(9):
                    box.append(eval(line[i]))
                scores.append(eval(line[9]))
                class_ids.append(eval(line[10]))
                boxes.append(box)
        boxes = LiDARInstance3DBoxes(boxes, 9)

        scores = torch.tensor(scores)
        class_ids = torch.tensor(class_ids, dtype=torch.int32)

        results.append(
            dict(pts_bbox=dict(boxes_3d=boxes, scores_3d=scores, labels_3d=class_ids))
        )
        prog_bar.update()
        
    
    eval_kwargs = cfg.get('evaluation', {}).copy()
    kwargs = {}
    # hard-code way to remove EvalHook args
    for key in [
            'interval', 'tmpdir', 'start', 'gpu_collect', 'save_best',
            'rule'
    ]:
        eval_kwargs.pop(key, None)
    print(eval_kwargs)
    eval_kwargs.update(dict(metric="mAP+NDS", **kwargs))
    print(dataset.evaluate(results, **eval_kwargs))

        
if __name__ == '__main__':
    main()
