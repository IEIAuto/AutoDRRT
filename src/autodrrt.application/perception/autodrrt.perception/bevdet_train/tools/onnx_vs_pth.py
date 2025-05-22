import torch 
import numpy as np 
import numpy as np 
import random

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def to_onnx():
    dummy_input = torch.randn(1, 3, 112, 112, dtype=torch.float)
    # model = model_res()
    model = model_osnet()

    input_names = ["data"]
    output_names = ["fc"]
    torch.onnx.export(
        model,
        dummy_input,
        "./osnet.onnx",
        verbose=True,
        input_names=input_names,
        output_names=output_names,
    )
    print("转换模型成功^^")


def pytorch_out(input):
    model = model_res() #model.eval
    # input = input.cuda()
    # model.cuda()
    torch.no_grad()
    output = model(input)
    # print output[0].flatten()[70:80]
    return output

def pytorch_onnx_test():
    import onnxruntime
    from onnxruntime.datasets import get_example

    def to_numpy(tensor):
        return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

    # 测试数据
    
    # dummy_input = torch.randn(1, 3, 112, 112, device='cpu')
    
   
    img_input = torch.rand([6, 3, 256, 704], dtype=torch.float, device='cpu')
    rot = torch.rand([1, 6, 3, 3], dtype=torch.float, device='cpu')
    tran = torch.rand([1, 6, 3], dtype=torch.float, device='cpu')
    intrin = torch.rand([1, 6, 3, 3], dtype=torch.float, device='cpu')
    post_rot = torch.rand([1, 6, 3, 3], dtype=torch.float, device='cpu')
    post_tran = torch.rand([1, 6, 3], dtype=torch.float, device='cpu')
    bda = torch.rand([1, 3, 3], dtype=torch.float, device='cpu')

    example_model = get_example("/data/liuhonggang/tmp_test/BEVDet-9027/deploy_out/img_stage.onnx")
    # netron.start(example_model) 使用 netron python 包可视化网络
    sess = onnxruntime.InferenceSession(example_model)
   
    input = {
        "images":to_numpy(img_input),
        "rot":to_numpy(rot),
        "trans":to_numpy(tran),
        "intrin":to_numpy(intrin),
        "post_rot":to_numpy(post_rot),
        "post_trans":to_numpy(post_tran),
        "bda":to_numpy(bda),
    }
    
    print("==========input=========")
    print(input)

    # onnx 网络输出
    images_feat, depth = sess.run(None, input)
    print("==============>")
    print(images_feat)
    print(images_feat.shape)
    print("==============>")
    
    print("==============>")
    print(depth)
    print(depth.shape)
    print("==============>")
    
    # images_feat_np = np.array(images_feat,dtype="float32") #need to float32
    # depth_np = np.array(depth,dtype="float32")  ##need to float32
    




    # torch_out_res = pytorch_out(dummy_input).detach().numpy()   #fc输出是二维 列表
    # print(torch_out_res)
    # print(torch_out_res.shape)

    # print("===================================>")
    # print("输出结果验证小数点后五位是否正确,都变成一维np")

    # torch_out_res = torch_out_res.flatten()
    # onnx_out = onnx_out.flatten()

    # pytor = np.array(torch_out_res,dtype="float32") #need to float32
    # onn=np.array(onnx_out,dtype="float32")  ##need to float32
    # np.testing.assert_almost_equal(pytor,onn, decimal=5)  #精确到小数点后5位，验证是否正确，不正确会自动打印信息
    # print("恭喜你 ^^ ，onnx 和 pytorch 结果一致 ， Exported model has been executed decimal=5 and the result looks good!")

from mmdet3d.models import build_model
from mmcv import Config, DictAction
from mmcv.runner import (get_dist_info, init_dist, load_checkpoint,
                         wrap_fp16_model)
def pth_infer():
    
    cfg = Config.fromfile("/data/liuhonggang/tmp_test/BEVDet-export/configs/bevdet/bevdet-r50-4dlongterm-depth-cbgs.py")
    model = build_model(cfg.model, test_cfg=cfg.get('test_cfg'))
    
    checkpoint = load_checkpoint(model, "/data/liuhonggang/tmp_test/BEVDet-export/bevdet-lt-d-ft-nearest.pth", map_location='cpu')
    
    model.cpu().eval()       
    

    img_input = torch.rand([6, 3, 256, 704], dtype=torch.float, device='cpu')
    rot = torch.rand([1, 6, 3, 3], dtype=torch.float, device='cpu')
    tran = torch.rand([1, 6, 3], dtype=torch.float, device='cpu')
    intrin = torch.rand([1, 6, 3, 3], dtype=torch.float, device='cpu')
    post_rot = torch.rand([1, 6, 3, 3], dtype=torch.float, device='cpu')
    post_tran = torch.rand([1, 6, 3], dtype=torch.float, device='cpu')
    bda = torch.rand([1, 3, 3], dtype=torch.float, device='cpu')
    
    # print(model.img_backbone(**(img_input,rot, tran, intrin,post_rot,post_tran, bda)))
    x = model.img_backbone(img_input)
    y = model.img_neck(x)
    z, depth = model.img_view_transformer((y, rot, tran, intrin,post_rot,post_tran, bda))

    # with torch.no_grad():
    #     output = model(return_loss=False, rescale=True, **(img_input,rot, tran, intrin,post_rot,post_tran, bda))

import onnx
import numpy as np

def pth_extract_params(model_path):
    # 加载模型参数字典
    state_dict = torch.load(model_path, map_location=torch.device('cpu'))
    
    # print(state_dict)

    # 保存参数
    params = {}
    
    # 遍历模型参数
    for layer_name, param_tensor in state_dict["state_dict"].items():
        if isinstance(param_tensor, torch.Tensor):
            param_array = param_tensor.numpy()
            params[layer_name] = param_array
            print(f"Layer: {layer_name}")
            print(f"Shape: {param_array.shape}")
            print(f"Values: {param_array}\n")
        else:
            print(f"Layer: {layer_name}")
            print(f"Values: {param_tensor} (Not a tensor)\n")
            for key, value in param_tensor:
                print(key)
                print(value)
    
    return params


def onnx_extract_params(model_path):
    # 加载ONNX模型
    model = onnx.load(model_path)
    
    # 获取模型的图结构
    graph = model.graph
    
    # 保存参数
    params = {}
    
    # 遍历每一个节点（层）
    for node in graph.node:
        layer_params = {}
        print(f"Layer: {node.name}")
        print(f"Type: {node.op_type}")

        # 打印输入和输出名称
        print(f"Inputs: {node.input}")
        print(f"Outputs: {node.output}")

        # 打印参数（初始化的张量）
        for input_name in node.input:
            for initializer in graph.initializer:
                if initializer.name == input_name:
                    from onnx import numpy_helper
                    param_array = numpy_helper.to_array(initializer)
                    layer_params[initializer.name] = param_array
                    print(f"Parameter: {initializer.name}")
                    print(f"Shape: {param_array.shape}")
                    print(f"Values: {param_array}")

        params[node.name] = layer_params
        print("\n")
    
    return params





if __name__ == '__main__':
    # pth_infer()
    # setup_seed(20)
    # pytorch_onnx_test()
    # 使用示例
    # model_path = '/data/liuhonggang/tmp_test/BEVDet-export/deploy_ptq/bev_stage_lt_d.onnx'
    # params = onnx_extract_params(model_path)
    model_path = '/home/liheyi/BEVDet-export/checkpoint/epoch_5_ema.pth'
    params =pth_extract_params(model_path)
