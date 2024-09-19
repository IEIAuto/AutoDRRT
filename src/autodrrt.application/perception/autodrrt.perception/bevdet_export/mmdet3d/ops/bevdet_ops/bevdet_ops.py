import torch
import numpy as np

from . import bevdet_ops_ext


def test_preprocess():
    img = torch.ones((6, 3, 900, 400), dtype=torch.int32, device="cuda:0")
    mean = torch.rand((3), dtype=torch.float32, device="cuda:0")    
    std = torch.rand((3), dtype=torch.float32, device="cuda:0")  
    
    out = torch.zeros((6, 3, 256, 704), dtype=torch.float32, device="cuda:0")  
        
    bevdet_ops_ext.preprocess_forward(img,
                                      mean,
                                      std,
                                      out,
                                      0.44,
                                      140,
                                      0)
            

def test_alignbev():
    adj_feat = torch.rand(size=(1, 8, 80, 128, 128), dtype=torch.float32, device="cuda:0")
    trans = torch.rand(size=(1, 8, 6), dtype=torch.float32, device="cuda:0")
    out = torch.zeros(size=(1, 8, 80, 128, 128), dtype=torch.float32, device="cuda:0")

    bevdet_ops_ext.alignbev_forward(adj_feat, trans, out)
    

def test_gather_bev():
    adj_feat = torch.rand(size=(1, 8, 80, 128, 128), dtype=torch.float32, device="cuda:0")
    curr_feat = torch.rand(size=(1, 80, 128, 128), dtype=torch.float32, device="cuda:0")
    flag = torch.ones(size=(1, 1), dtype=torch.int32, device="cuda:0")
    
    out = torch.rand(size=(1, 9, 80, 128, 128), dtype=torch.float32, device="cuda:0")
    
    bevdet_ops_ext.gatherbev_forward(adj_feat, curr_feat, flag, out)