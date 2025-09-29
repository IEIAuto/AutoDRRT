import argparse
import os
import torch
# import mmcv
import warnings
import time
import tensorrt as trt
import pycuda.driver as cuda
import os.path as osp
import numpy as np
import tensorrt_llm
# import tensorrt_llm.profiler as profiler
# from tensorrt_llm import logger
from tensorrt_llm.runtime import ModelRunner
from transformers import AutoTokenizer
import json
import time 

class InferTrtLLM(object):
    def __init__(self, llm_engine_pth, tokenizer_pth) -> None:
        device_id = 0
        self.IMAGE_TOKEN_INDEX = -200
        self.llm_engine_pth = llm_engine_pth
        torch.cuda.set_device(device_id)
        self.device = "cuda:%d" % (device_id)
        self.stream = torch.cuda.Stream(torch.cuda.current_device())
        torch.cuda.set_stream(self.stream)
        
        # 分词器初始的设置是和模型部分一致的
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_pth, model_max_length=2048, padding_side="right", use_fast=False,)
        tokenizer.pad_token = tokenizer.unk_token
        tokenizer.padding_side = "right"
        self.tokenizer = tokenizer
        self.model_type = "llava_llama"
        self.init_llm()
    
    def init_llm(self):
        # 从engine里面读取llm模型吧
        # self.model = ModelRunner.from_dir(str(self.llm_engine_pth), rank=0, debug_mode=False, stream=self.stream)
        self.model = ModelRunner.from_dir(str(self.llm_engine_pth), rank=0, debug_mode=True, stream=self.stream)
        self.model_config = self.model.session._model_config
        self.runtime_mapping = self.model.session.mapping
    
    
    def image_to_ptuning(self, input_ids, vision_embeded):
        updated_input_ids = []
        # current_vocab_size = self.tokenizer.vocab_size
        # current_vocab_size = len(self.tokenizer)
        current_vocab_size = 32001
        
        for batch_idx, cur_input_ids in enumerate(input_ids):
            num_images = (cur_input_ids == self.IMAGE_TOKEN_INDEX).sum()
            if num_images == 0:
                updated_input_ids.append(cur_input_ids)
                continue
            im_token_ids = torch.where(cur_input_ids == self.IMAGE_TOKEN_INDEX)[0].tolist()
            im_token_ids = [-1] + im_token_ids + [cur_input_ids.shape[0]]
            im_idx = 0
            for i in range(len(im_token_ids) - 1):
                updated_input_ids.append(cur_input_ids[im_token_ids[i]+1:im_token_ids[i+1]])
                if im_idx < vision_embeded.shape[0]:
                    im = vision_embeded[im_idx]
                    im_size = im.shape[0]
                    im_indices = torch.from_numpy(np.arange(current_vocab_size, current_vocab_size + im_size)).cuda()
                    updated_input_ids.append(im_indices)
                    im_idx += 1
        return torch.cat(updated_input_ids).unsqueeze(0), vision_embeded.reshape(1, -1, vision_embeded.shape[2])

    def generate(self, input_ids, vision_embeded, img_metas):

        input_ids, prompt_table = self.image_to_ptuning(input_ids, vision_embeded)
        input_ids = input_ids.contiguous().to(dtype=torch.int32)
        prompt_table = prompt_table.cuda().contiguous().to(dtype=torch.float16)
        t_start = time.time()

        output_ids = self.model.generate(
            input_ids, 
            img_metas,
            prompt_table=prompt_table,
            end_id=self.tokenizer.eos_token_id,
            pad_id=self.tokenizer.pad_token_id,
            do_sample=True,
            temperature=0.1,
            top_p=0.75,
            num_beams=1,
            max_new_tokens=320,
            use_cache=False)
       
        output_ids = torch.masked_select(output_ids, output_ids.lt(self.tokenizer.vocab_size)).reshape([1, -1])
         
        self.stream.synchronize()
        return output_ids


class InferTrt(object):
    def __init__(self, logger, qa_save_path, LLM_engine=None, torch_ref_model=None):        
        self.cuda_ctx = cuda.Device(0).retain_primary_context()
        self.cuda_ctx.push()

        self.builder = trt.Builder(logger)
        self.logger = logger
        self.network = self.builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
        self.opt = self.builder.create_optimization_profile()

        self.config = self.builder.create_builder_config()
        self.config.add_optimization_profile(self.opt)
        # self.config.max_workspace_size = 2 << 34
        self.config.builder_optimization_level = 5
        self.config.profiling_verbosity = trt.ProfilingVerbosity.DETAILED
        # self.config.set_flag(trt.BuilderFlag.FP16)  # control this
        self.stream = cuda.Stream()
        self.cuda_ctx.pop()
        self.curr_scene_token = None
        self.start_timestamp = None
        self.bindings = {}
        # self.bbox_coder = NMSFreeCoder(
        #     pc_range=[-51.2, -51.2, -5.0, 51.2, 51.2, 3.0],
        #     voxel_size=[0.2, 0.2, 8],
        #     post_center_range=[-61.2, -61.2, -10.0, 61.2, 61.2, 10.0],
        #     max_num=300,
        #     score_threshold=None,
        #     num_classes=10
        # )
        self.LLM_engine = LLM_engine
        self.qa_save_path = qa_save_path
        self.torch_ref_model = torch_ref_model

    
    def from_onnx(self, onnx_mod):
        parser = trt.OnnxParser(self.network, self.logger)
        result = parser.parse(onnx_mod.SerializeToString())
        if not result:
            print("failed parsing onnx")
            for error in range(parser.num_errors):
                print(parser.get_error(error))
            exit(-1)
        self.buf = self.builder.build_serialized_network(self.network, self.config)
        self._build_engine()
        
    def _build_engine(self):
        self.runtime = trt.Runtime(self.logger)        
        self.engine = self.runtime.deserialize_cuda_engine(self.buf)
        self.context = self.engine.create_execution_context()
        # self.context.profiler = CustomProfiler()
        self.names = []
        n_io = self.engine.num_io_tensors
        for i in range(n_io):
            self.names.append(self.engine.get_tensor_name(i))

    def write(self, path):
        with open(path, "wb") as fp:
            fp.write(self.buf)

    def read(self, path):
        print("[TensorRT INFO] Loading engine from: ", path)
        with open(path, "rb") as fp:
            self.buf = fp.read()
        self._build_engine()

    
    def eval(self):
        if self.torch_ref_model is not None:            
            self.torch_ref_model.eval()
        if len(self.bindings) == 0:
            create_bindings_tensor = True
        else:
            create_bindings_tensor = False
        n_io = self.engine.num_io_tensors
        metas_in = []
        metas_out = []
        for i in range(n_io):
            tname = self.engine.get_tensor_name(i)
            tshape = str(self.engine.get_tensor_shape(tname))
            tdtype = str(self.engine.get_tensor_dtype(tname))
            tmode = str(self.engine.get_tensor_mode(tname))
            m = f"{i}\t{tname}\t{tshape}\t{tdtype}"
            if "INPUT" in tmode:
                metas_in.append(m)
            elif "OUTPUT" in tmode:
                metas_out.append(m)
            else:
                assert False, f"Unrecognized tensor mode: {tname}: {tmode}."
            if create_bindings_tensor:
                self.bindings[tname] = torch.zeros(list(self.engine.get_tensor_shape(tname)), 
                                        dtype=torch.float32, 
                                        device="cuda:0").contiguous()
        print("##### Input Bindings: ")
        print("\n".join(metas_in))
        print("##### Output Bindings: ")
        print("\n".join(metas_out))
        return

    def __call__(self, img_metas, input_ids, img, lidar2img, intrinsics, extrinsics, timestamp, img_timestamp, 
                ego_pose, ego_pose_inv, command, can_bus,
                return_loss=False, rescale=True):
        if self.torch_ref_model is not None:
            data_dict = {
                "img_metas": img_metas, "input_ids": input_ids, "img": img, "lidar2img": lidar2img, 
                "intrinsics": intrinsics, "extrinsics": extrinsics, "timestamp": timestamp, "img_timestamp": img_timestamp, 
                "ego_pose": ego_pose, "ego_pose_inv": ego_pose_inv, "command": command, "can_bus": can_bus,
            }
            ref_result_list = self.torch_ref_model(return_loss=False, rescale=True, **data_dict)
        else:
            ref_result_list = None
        return self.forward(img_metas=img_metas, 
                            input_ids=input_ids, 
                            img=img, 
                            lidar2img=lidar2img, 
                            intrinsics=intrinsics, 
                            timestamp=timestamp, 
                            ego_pose=ego_pose, 
                            ego_pose_inv=ego_pose_inv, 
                            command=command, 
                            can_bus=can_bus)
   
    
    def forward(self, save_name):
       
        
        data_start_time = time.time()

        save_path = os.path.join("./nus_tensor_data/", f"{save_name}.pt")
        data = torch.load(save_path)
        
        print("keys in data:")
        for k in data.keys():
            print("  ", k, type(data[k]))


        self.bindings = {
            name: tensor.to("cuda:0").contiguous()
            for name, tensor in data.items()
            if isinstance(tensor, torch.Tensor)  # 只恢复张量部分，排除 input_ids 等
        }
        
        input_ids = [[data.get("input_ids", None)]]

        data_end_time = time.time()
        
        start_time = time.time()

        # inference
        self.cuda_ctx.push()
        for i in range(len(self.names)):
           
            self.context.set_tensor_address(self.names[i], self.bindings[str(self.names[i])].data_ptr())
        self.context.execute_async_v3(stream_handle=self.stream.handle)
        self.stream.synchronize()
        self.cuda_ctx.pop()
        
        vision_time = time.time() - start_time
        print("vision_time cost", vision_time)
        
        
        output_ids_lst = []
        for q_id, input_llm_id in enumerate(input_ids[0]):
            input_llm_id = input_llm_id.unsqueeze(0).to(device="cuda:0").contiguous()
            
            img_metas = {
                "save_name":save_name
                }

            output_ids = self.LLM_engine.generate(input_llm_id, self.bindings["vision_embeded"], img_metas)
            output_ids_lst.append(output_ids)
        end_time = time.time()

def main():
    
    
    
    engine_pth = "./vision_engine/eva_base_tinyllama_mixed_precision.engine"
    qa_save_path = "./engine_save_path/"
    llm_engine_pth = "./llm_engine"
    tokenizer_pth = "./llm_ckpts"
    nus_tensor = "./nus_tensor_data/"

    # build the engine
    logger = trt.Logger(trt.Logger.VERBOSE)
    # 这个应该是构建engine的
    engine = InferTrt(logger, qa_save_path, engine_pth)
    engine.read(engine_pth)
    # build LLM engine
    engine.LLM_engine = InferTrtLLM(llm_engine_pth=llm_engine_pth, tokenizer_pth=tokenizer_pth)
    
    prefix_list = []
    
    for filename in os.listdir(nus_tensor):
        if filename.endswith(".pt"):
            start_time = time.time()
            prefix = os.path.splitext(filename)[0]  # 去掉后缀 .pt
            prefix_list.append(prefix)
            engine.forward(prefix)
            end_time = time.time()
            print("all over the time is ", end_time - start_time)
    
if __name__ == '__main__':
    torch.multiprocessing.set_start_method('fork')
    cuda.init()
    torch.cuda.init()
    main()
