# Copyright (c) Phigent Robotics. All rights reserved.
import torch
from mmdet3d.core import bbox3d2result

from mmdet.models import DETECTORS
from .. import builder
from .base import Base3DDetector
from mmdet3d.ops import BEVPool, AlignBEV, GatherBEV, Preprocess

import torch.nn.functional as F

class ImageStage(torch.nn.Module):
    def __init__(self, img_backbone=None, img_neck=None, img_view_transformer=None, **kwargs):
        super(ImageStage, self).__init__()
        assert img_backbone != None and img_neck != None and img_view_transformer != None
        
        self.img_backbone = builder.build_backbone(img_backbone)
        self.img_neck = builder.build_neck(img_neck)
        self.img_view_transformer = builder.build_neck(img_view_transformer)        
    
    def encode(self, imgs):
        B, N, C, imH, imW = imgs.shape
        imgs = imgs.view(B * N, C, imH, imW)
        x = self.img_backbone(imgs)
        x = self.img_neck(x)
        if type(x) in [list, tuple]:
            x = x[0]
        _, output_dim, ouput_H, output_W = x.shape
        x = x.view(B, N, output_dim, ouput_H, output_W)
        return x
    
    def forward(self, img): # for trt
        '''
        img : 6 x 3 x H x W
        '''
        x = self.img_backbone(img) 
        x = self.img_neck(x)  # 6 x 512 x h x w
        x = self.img_view_transformer.depth_net(x)

        D = self.img_view_transformer.D
        C = self.img_view_transformer.out_channels
        
        depth = x[:, : D].softmax(dim=1)
        
        x = x[:, D : D + C]        # 6 x C x h x w
        x = x.permute(0, 2, 3, 1)  # 6 x h x w x C

        return x.contiguous(), depth.contiguous()


class ImageStageDepth(ImageStage):
    def __init__(self, img_backbone=None, img_neck=None, img_view_transformer=None, **kwargs):
        super(ImageStageDepth, self).__init__(img_backbone, img_neck, img_view_transformer, **kwargs)
        
    def forward(self, img, rot, tran, intrin, post_rot, post_tran, bda): # for trt
        '''
        img       : 6 x 3 x H x W
        rot       : 1 x 6 x 3 x 3
        tran      : 1 x 6 x 3
        intrin    : 1 x 6 x 3 x 3
        post_rot  : 1 x 6 x 3 x 3
        post_tran : 1 x 6 x 3
        bda       : 1 x 3 x 3
        '''
        x = self.img_backbone(img) 
        r = self.img_neck(x)  # 6 x 512 x h x w
        
        mlp_input = self.img_view_transformer.get_mlp_input(rot, tran, intrin, post_rot, post_tran, bda)
        x, tran_feat = self.img_view_transformer.depth_net(r, mlp_input)
        #x, mlp_bn, con_se, depth_se = self.img_view_transformer.depth_net(r, mlp_input)

        D = self.img_view_transformer.D
        C = self.img_view_transformer.out_channels
        
        depth = x.softmax(dim=1)
        #depth = x[:, : D].softmax(dim=1)
        
        #x = x[:, D : D + C]        # 6 x C x h x w
        tran_feat = tran_feat.permute(0, 2, 3, 1)  # 6 x h x w x C
        #x = x.permute(0, 2, 3, 1)  # 6 x h x w x C

        return tran_feat.contiguous(), depth.contiguous()
        #return x.contiguous(), depth.contiguous(), mlp_input.contiguous(), r.contiguous(), mlp_bn.contiguous(), con_se.contiguous(), depth_se.contiguous()


class BEVStage(torch.nn.Module):
    def __init__(self, pre_process=None, img_bev_encoder_backbone=None, img_bev_encoder_neck=None, pts_bbox_head=None, **kwargs):
        super(BEVStage, self).__init__()
        assert pre_process != None and img_bev_encoder_backbone != None and img_bev_encoder_neck != None and pts_bbox_head != None
        
        self.pre_process_net = builder.build_backbone(pre_process)
        self.img_bev_encoder_backbone = builder.build_backbone(img_bev_encoder_backbone)
        self.img_bev_encoder_neck = builder.build_neck(img_bev_encoder_neck)
        self.pts_bbox_head = builder.build_head(pts_bbox_head)
        
    def encode(self, x):

        x = self.pre_process_net(x)[0]
        x = self.img_bev_encoder_backbone(x)
        x = self.img_bev_encoder_neck(x)
        if type(x) in [list, tuple]:
            x = x[0]
        x = self.pts_bbox_head([x])

        return x
    
    def forward(self, x): # for trt 
        '''
        x : 1 x 720 x 128 x 128 
        '''
        x = x.contiguous()

        x[:, 0:80, :, :] = self.pre_process_net(x[:, 0:80, :, :])[0]

        '''  swpld added '''
        if True:
            x = F.pad(x, [2, 2, 2, 2])
 
        x = self.img_bev_encoder_backbone(x)
        x = self.img_bev_encoder_neck(x)

        '''  swpld added '''
        if True:
            x = x[:, :, 1:181, 1:181]

        x = self.pts_bbox_head([x])
        return x


@DETECTORS.register_module()
class BEVModel(Base3DDetector):
    def __init__(self,                
                img_backbone=None,
                img_neck=None,
                img_view_transformer=None, 
                img_bev_encoder_backbone=None, 
                img_bev_encoder_neck=None, 
                pts_bbox_head=None,
                train_cfg=None,
                test_cfg=None,
                pretrained=None,
                use_depth=False,
                align_after_view_transfromation=True,
                num_adj=0,
                pre_process=None,
                **kwargs):
        super(BEVModel, self).__init__(**kwargs)
        
        assert img_backbone != None and img_neck != None and img_view_transformer != None and pre_process != None and \
               img_bev_encoder_backbone != None and img_bev_encoder_neck != None and pts_bbox_head != None
        self.use_depth = use_depth
        pts_train_cfg = train_cfg.pts if train_cfg else None
        pts_bbox_head.update(train_cfg=pts_train_cfg)
        pts_test_cfg = test_cfg.pts if test_cfg else None
        pts_bbox_head.update(test_cfg=pts_test_cfg)
        
        if self.use_depth:
            self.img_encoder = ImageStageDepth(img_backbone, img_neck, img_view_transformer)
        else:
            self.img_encoder = ImageStage(img_backbone, img_neck, img_view_transformer)
        self.bev_encoder = BEVStage(pre_process, img_bev_encoder_backbone, img_bev_encoder_neck, pts_bbox_head)
        
    
    def forward(self, return_loss=True, **kwargs):
        if return_loss:
            return self.forward_train(**kwargs)
        else:
            return self.forward_test(**kwargs)
    
    def forward_train(self, gt_bboxes_3d=None, gt_labels_3d=None, img_inputs=None, **kwargs):
        imgs = img_inputs[0]
        x = self.img_encoder.encode(imgs)
        x, depth = self.img_encoder.img_view_transformer([x] + img_inputs[1:7])
        outs = self.bev_encoder.encode(x)
        
        losses = dict()
        loss_inputs = [gt_bboxes_3d, gt_labels_3d, outs]
        losses.update(self.bev_encoder.pts_bbox_head.loss(*loss_inputs))
        return losses
    
    def forward_test(self, points=None, img_metas=None, img_inputs=None, gt_bboxes_3d=None, gt_labels_3d=None, **kwargs):

        assert img_inputs != None and img_metas != None 
        if gt_bboxes_3d == None and gt_labels_3d == None:
            return self.simple_test(img_inputs[0], img_metas[0], **kwargs)
        assert 0 # TODO
        
    def simple_test(self, img_input, img_meta, **kwargs):
        x = self.img_encoder.encode(img_input[0])

        print('1111111111111111111111111111')

        x, _ = self.img_encoder.img_view_transformer([x] + img_input[1:7])
        x = self.bev_encoder.encode(x)
        bbox_list = [dict() for _ in range(len(img_meta))]
        bbox_lists = self.bev_encoder.pts_bbox_head.get_bboxes(x, img_meta, rescale=False)
        bbox_results = [
            bbox3d2result(bboxes, scores, labels) for bboxes, scores, labels in bbox_lists
        ]
        for result_dict, pts_bbox in zip(bbox_list, bbox_results):
            result_dict['pts_bbox'] = pts_bbox
        # self.save_bboxes(bbox_lists[0])
        return bbox_list
    
    def aug_test(self, ):
        assert 0
    
    def extract_feat(self, ):
        assert 0 

    def save_bboxes(self, bboxes_list):
        boxes, scores, labels = bboxes_list
        bottom_center = boxes.bottom_center
        bottom_dims = boxes.dims
        bottom_rot = boxes.yaw
        files = 'pytorch_boxes.txt'
        with open(files, 'w', encoding='utf-8') as f:
            for i in range(len(scores)):
                out = "{:.5f} {:.5f} {:.5f} {:.5f} {:.5f} {:.5f} {:.5f} {:.5f} {}\n".format(
                    bottom_center[i, 0], bottom_center[i, 1], bottom_center[i, 2],
                    bottom_dims[i, 0], bottom_dims[i, 1], bottom_dims[i, 2], 
                    bottom_rot[i], scores[i], labels[i]
                )
                f.write(out)
                
        print("Done!")        


@DETECTORS.register_module()
class BEVONE(Base3DDetector):
    def __init__(self,                
                img_backbone=None,
                img_neck=None,
                img_view_transformer=None, 
                img_bev_encoder_backbone=None, 
                img_bev_encoder_neck=None, 
                pts_bbox_head=None,
                train_cfg=None,
                test_cfg=None,
                pretrained=None,
                use_depth=False,
                align_after_view_transfromation=True,
                num_adj=0,
                pre_process=None,
                **kwargs):
        super(BEVONE, self).__init__(**kwargs)
        
        assert img_backbone != None and img_neck != None and img_view_transformer != None and \
               img_bev_encoder_backbone != None and img_bev_encoder_neck != None and pts_bbox_head != None
        self.use_depth = use_depth
        pts_train_cfg = train_cfg.pts if train_cfg else None
        pts_bbox_head.update(train_cfg=pts_train_cfg)
        pts_test_cfg = test_cfg.pts if test_cfg else None
        pts_bbox_head.update(test_cfg=pts_test_cfg)
        
        assert img_backbone != None and img_neck != None and img_view_transformer != None
        self.img_backbone = builder.build_backbone(img_backbone)
        self.img_neck = builder.build_neck(img_neck)
        self.img_view_transformer = builder.build_neck(img_view_transformer)    
        
        assert img_bev_encoder_backbone != None and img_bev_encoder_neck != None and pts_bbox_head != None
        self.img_bev_encoder_backbone = builder.build_backbone(img_bev_encoder_backbone)
        self.img_bev_encoder_neck = builder.build_neck(img_bev_encoder_neck)
        self.pts_bbox_head = builder.build_head(pts_bbox_head)
        
        self.preprocess = Preprocess()
        self.bevpool = BEVPool()
        self.align = AlignBEV()
        
        self.gather_bev = GatherBEV()
    
    def forward(self, 
                img,
                mean,
                std,
                cam_params,
                ranks_depth,
                ranks_feat, 
                ranks_bev,
                interval_starts,
                interval_lengths,
                adj_bevfeats,
                adj_transforms,
                flag,
                ):
        '''
        img              : B*6 x 3 x 256 x 704
        mean             : 3
        std              : 3
        cam_params       : B x 6 x 27
        ranks_depth      : ~179535
        ranks_feat       : ~179535
        ranks_bev        : ~179535
        interval_starts  : ~11404
        interval_lengths : ~11404
        adj_bevfeats     : B x 8 x 80 x 128 x 128, ...
        adj_transforms   : B x 8 x 6
        flag             : B
        '''
        # image stage
        img = self.preprocess.apply(img, mean, std)
        x = self.img_backbone(img) 
        x = self.img_neck(x)  # B*6 x 512 x h x w
        
        depth, x = self.img_view_transformer.depth_net(x, cam_params)
        
        D = self.img_view_transformer.D
        C = self.img_view_transformer.out_channels
        
        depth = depth.softmax(dim=1).contiguous()  # b*6 x 118 x 16 x 44
        
        x = x.contiguous()                 # b*6 x C x h x w
        # x = x.permute(0, 2, 3, 1).contiguous()         # b*6 x 16 x 44 x 80
        
        # bevpool
        curr_bevfeat = self.bevpool.apply(depth, x, ranks_depth, ranks_feat, ranks_bev, interval_starts, interval_lengths)  # 80 x 128 x 128
        
        
        adj_bevfeats = self.align.apply(adj_bevfeats, adj_transforms)  # b x 8 x 80 x 128 * 128
        # adj_bevfeats = torch.cat(adj_bevs, dim=0).contiguous()
        
        x = self.gather_bev.apply(adj_bevfeats, curr_bevfeat, flag)
        # print(bevfeats.size())
        # # 1 x 720 x 128 x 128
        # bevfeats = torch.cat((curr_bevfeat, *adj_bevfeats), dim=0).contiguous()
        
        # x = bevfeats
        x = self.img_bev_encoder_backbone(x)
        x = self.img_bev_encoder_neck(x)
        x = self.pts_bbox_head([x])        
        x = x[0][0]
        return x['reg'], x['height'], x['dim'], x['rot'], x['vel'], x['heatmap'], curr_bevfeat
    
    def aug_test(self, ):
        assert 0
    
    def extract_feat(self, ):
        assert 0 

    def simple_test(self, ):
        assert 0

