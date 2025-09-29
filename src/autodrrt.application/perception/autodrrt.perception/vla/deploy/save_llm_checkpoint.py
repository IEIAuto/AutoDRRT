
import argparse
import mmcv
import numpy as np
import os
import torch
import warnings
from mmcv import Config, DictAction
from mmcv.cnn import fuse_conv_bn
from mmcv.runner import load_checkpoint, wrap_fp16_model
import io
from mmdet3d.models import build_model
from mmdet.apis import set_random_seed
from mmdet.datasets import replace_ImageToTensor
from transformers import AutoTokenizer
from llm_lib import LlavaLlamaForCausalLM
from peft import LoraConfig, get_peft_model

EGO_WAYPOINT_TOKEN = "<waypoint>"  # ✅ 你要添加的特殊token

def parse_args():
    parser = argparse.ArgumentParser(
        description='OmniDrive OOX export(Vision part).')
    parser.add_argument('--config',help='test config file path', default="/home/liry/swpld/OmniDrive/projects/configs/OmniDrive/eva_base_tinyllama.py")
    parser.add_argument('--checkpoint', help='checkpoint file', default="/home/liry/swpld/OmniDrive/ckpts/fp16.pth")
    parser.add_argument('--llm_checkpoint', help='llm_checkpoint file', default="/home/liry/swpld/OmniDrive/ckpts/tiny_llama/")
    parser.add_argument('--save_checkpoint_pth', help='llm_checkpoint file', default="./ckpts/")
    parser.add_argument(
        '--fuse-conv-bn',
        action='store_true',
        help='Whether to fuse conv and bn, this will slightly increase'
        'the inference speed')
    parser.add_argument('--show', action='store_true', help='show results')
    parser.add_argument('--seed', type=int, default=0, help='random seed')
    parser.add_argument(
        '--deterministic',
        action='store_true',
        help='whether to set deterministic options for CUDNN backend.')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. If the value to '
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        'Note that the quotation marks are necessary and that no white space '
        'is allowed.')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)
    return args


def load_model(base_model, use_lora=False, frozen=False):
    model = LlavaLlamaForCausalLM.from_pretrained(base_model, torch_dtype=torch.float16, device_map='auto')
    model.gradient_checkpointing_enable()
    
    if frozen:
        model.eval()
        for p in model.parameters():
            p.requires_grad = False
            
    if use_lora:
        peft_config = LoraConfig(
                r=16,
                lora_alpha=16,
                target_modules=("q_proj", "k_proj", "v_proj", "o_proj"),
                lora_dropout=0.05,
                bias="none",
                task_type="CAUSAL_LM")
        model = get_peft_model(model, peft_config)

        for param in filter(lambda p: p.requires_grad,model.parameters()):
            param.data = param.data.to(torch.float32)
    # model = model.half()
    return model

def add_special_token(special_token_list, tokenizer, model):
    # 给新的token添加索引并用原有embedding的平均值初始化
    num_new_tokens = tokenizer.add_tokens(special_token_list, special_tokens=True)
    model.resize_token_embeddings(len(tokenizer))
    if num_new_tokens > 0:
        input_embeddings = model.get_input_embeddings().weight.data
        output_embeddings = model.get_output_embeddings().weight.data
        input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)
        output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)
        input_embeddings[-num_new_tokens:] = input_embeddings_avg
        output_embeddings[-num_new_tokens:] = output_embeddings_avg


if __name__ == '__main__':
    torch.multiprocessing.set_start_method('fork')
    args = parse_args()

    # 加载 config
    cfg = Config.fromfile(args.config)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    # 插件模块导入（如有）
    if cfg.get('custom_imports', None):
        from mmcv.utils import import_modules_from_strings
        import_modules_from_strings(**cfg['custom_imports'])
    # import modules from plguin/xx, registry will be updated
    if hasattr(cfg, 'plugin'):
        if cfg.plugin:
            import importlib
            if hasattr(cfg, 'plugin_dir'):
                plugin_dir = cfg.plugin_dir
                _module_dir = os.path.dirname(plugin_dir)
                _module_dir = _module_dir.split('/')
                _module_path = _module_dir[0]

                for m in _module_dir[1:]:
                    _module_path = _module_path + '.' + m
                print(_module_path)
                plg_lib = importlib.import_module(_module_path)
            else:
                # import dir is the dirpath for the config file
                _module_dir = os.path.dirname(args.config)
                _module_dir = _module_dir.split('/')
                _module_path = _module_dir[0]
                for m in _module_dir[1:]:
                    _module_path = _module_path + '.' + m
                print(_module_path)
                plg_lib = importlib.import_module(_module_path)       
                

    # CUDNN 设置
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True

    # 数据集设置
    cfg.model.pretrained = None
    samples_per_gpu = 1
    if isinstance(cfg.data.test, dict):
        cfg.data.test.test_mode = True
        samples_per_gpu = cfg.data.test.pop('samples_per_gpu', 1)
        if samples_per_gpu > 1:
            cfg.data.test.pipeline = replace_ImageToTensor(cfg.data.test.pipeline)
    elif isinstance(cfg.data.test, list):
        for ds_cfg in cfg.data.test:
            ds_cfg.test_mode = True
        samples_per_gpu = max(ds_cfg.pop('samples_per_gpu', 1) for ds_cfg in cfg.data.test)
        if samples_per_gpu > 1:
            for ds_cfg in cfg.data.test:
                ds_cfg.pipeline = replace_ImageToTensor(ds_cfg.pipeline)

    # 设置随机种子
    if args.seed is not None:
        set_random_seed(args.seed, deterministic=args.deterministic)

    # 加载模型和 checkpoint
    cfg.model.train_cfg = None
    # 加载训练的模型的参数
    model = build_model(cfg.model, test_cfg=cfg.get('test_cfg'))

    import torch

    # 假设你有一个文本输入，先用tokenizer转成token id
    # input_text = "Hello, this is a test input."
    # inputs = tokenizer(input_text, return_tensors="pt")
    
    # # 取出token ids
    # input_ids = inputs["input_ids"]  # shape (1, seq_len)
    
    # device = next(model.lm_head.parameters()).device  # 获取模型设备
    # input_ids = input_ids.to(device)  
    
    input_ids_list = [1, 319, 13563,  1546,   263]

    # 构造 tensor，shape (1, 5) 表示 batch_size=1，序列长度=5
    input_ids = torch.tensor([input_ids_list], dtype=torch.long)

    # 确保 input_ids 在和模型同一个设备
    device = next(model.lm_head.parameters()).device
    input_ids = input_ids.to(device)
    
    # 调用embedding层，获取embedding结果
    embedding_output = model.lm_head.base_model.model.model.embed_tokens(input_ids)

   
    model.lm_head = load_model(args.llm_checkpoint, use_lora=False, frozen=False)
   
   
    # ✅ 加载 tokenizer 并添加特殊 token
    tokenizer = AutoTokenizer.from_pretrained(args.llm_checkpoint, use_fast=False)
    add_special_token([EGO_WAYPOINT_TOKEN], tokenizer, model.lm_head)
    model.lm_head.config.waypoint_token_idx = tokenizer(EGO_WAYPOINT_TOKEN, add_special_tokens=False).input_ids[0]
    
    ckpt = torch.load(args.checkpoint, map_location="cpu")
 

    
    fp16_cfg = cfg.get('fp16', None)
    if fp16_cfg is not None:
        wrap_fp16_model(model)
    
    load_checkpoint(model, args.checkpoint, map_location="cpu")
    
    if args.fuse_conv_bn:
        model = fuse_conv_bn(model)

    
    # ✅ 保存模型和分词器
    model.lm_head.save_pretrained(args.save_checkpoint_pth)
    tokenizer.save_pretrained(args.save_checkpoint_pth)

    input_ids_list = [1, 319, 13563,  1546,   263]

    # 构造 tensor，shape (1, 5) 表示 batch_size=1，序列长度=5
    input_ids = torch.tensor([input_ids_list], dtype=torch.long)

    # 确保 input_ids 在和模型同一个设备
    device = next(model.lm_head.parameters()).device
    input_ids = input_ids.to(device)

    # 调用embedding层，获取embedding结果
    embedding_output = model.lm_head.model.embed_tokens(input_ids)
