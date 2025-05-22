import numpy as np
import torch

from . import bevdet_ops_ext

class BEVPool(torch.autograd.Function):
    @staticmethod
    def symbolic(g,
                 depth,
                 feat,
                 ranks_depth,
                 ranks_feat,
                 ranks_bev,
                 interval_starts,
                 interval_lengths,
                 bev_h=128,
                 bev_w=128,
                 n=6,
                 ):
        return g.op(
            'bevdet::BEVPool',
             depth,
             feat,
             ranks_depth,
             ranks_feat,
             ranks_bev,
             interval_starts,
             interval_lengths,
             bev_h_i=bev_h,   # dim_i
             bev_w_i=bev_w,   # dim_i
             n_i=n
             )
        
    @staticmethod
    def forward(g,
                depth,            # b*n x d x h x w
                feat,             # b*n x c x h x w    
                ranks_depth,
                ranks_feat,
                ranks_bev,
                interval_starts,
                interval_lengths,
                bev_h=128,
                bev_w=128,
                n=6
                ):
        # depth : [6, 118, 16, 44]
        # feat  : [6, 80, 16, 44]
        feat_channel = feat.size(1)
        bn = feat.size(0)
        out = feat.new_zeros((int(bn / n), feat_channel, bev_h, bev_w))
        bevdet_ops_ext.bevpool_forward(depth, 
                                       feat,
                                       ranks_depth, 
                                       ranks_feat, 
                                       ranks_bev,
                                       interval_starts,
                                       interval_lengths,
                                       out,
                                       bev_h,
                                       bev_w,
                                       n)
        
        return out
    

class Preprocess(torch.autograd.Function):
    @staticmethod
    def symbolic(g,
                 img,
                 mean,
                 std,
                 crop_h=140,
                 crop_w=0,
                 resize_radio=0.44,
                 ):
        return g.op(
            'bevdet::Preprocess',
             img,
             mean,
             std,
             crop_h_i=140,
             crop_w_i=0,
             resize_radio_f=0.44,
             
        )
    @staticmethod
    def forward(g, 
                img, 
                mean,
                std,
                resize_radio=0.44, 
                crop_h=140,
                crop_w=0):
        N, C, H, W = img.size()
        
        out = mean.new_zeros((N, C, int(H * resize_radio - crop_h), int(W * 4 * resize_radio - crop_w)))
        bevdet_ops_ext.preprocess_forward(img, mean, std, out, resize_radio, crop_h, crop_w)
        return out
        


class AlignBEV(torch.autograd.Function):
    @staticmethod
    def symbolic(g,
                 adj_feats,
                 transforms):
        return g.op(
            'bevdet::AlignBEV',
            adj_feats,
            transforms)
    @staticmethod
    def forward(g,
                adj_feats,
                transforms,
                ):
        out = adj_feats.new_zeros(adj_feats.shape)
        bevdet_ops_ext.alignbev_forward(adj_feats, transforms, out)
        return out
    
class GatherBEV(torch.autograd.Function):
    @staticmethod
    def symbolic(g,
                 adj_feats,
                 curr_feat,
                 flag):
        return g.op(
            'bevdet::GatherBEV',
            adj_feats,
            curr_feat,
            flag
        )
    @staticmethod
    def forward(g,
                adj_feat,   # b x 8 x 80 x 128 x 128
                curr_feat,  # b x 80 x 128 x 128
                flag 
                ):    
        b, n, c, h , w = adj_feat.size()
        out = adj_feat.new_zeros((b, (n + 1) * c, h, w))
        bevdet_ops_ext.gatherbev_forward(adj_feat, curr_feat, flag, out)
        return out
        # b * 720 * 128 * 128