import os
import torch
import tensorrt as trt
import pycuda.driver as cuda
from tensorrt_llm.runtime import ModelRunner
from transformers import AutoTokenizer


# 配置常量
DEVICE_ID = 0
IMAGE_TOKEN_INDEX = -200
MODEL_TYPE = "llava_llama"
TOKENIZER_MODEL_MAX_LENGTH = 2048
DATA_BASE_PATH = "./vla/model_result"
ENGINE_PATH = f"{DATA_BASE_PATH}/vision_onnx_opt/eva_base_tinyllama_quant_max.engine"
QA_SAVE_PATH = f"{DATA_BASE_PATH}/engine_save_path_int8/"
LLM_ENGINE_PATH = f"{DATA_BASE_PATH}/llm_engine_w4a16_opt/"
TOKENIZER_PATH = f"{DATA_BASE_PATH}/llm_ckpts"
NUS_TENSOR_PATH = f"{DATA_BASE_PATH}/nus_tensor_data/"


class InferTrtLLM(object):
    def __init__(self, llm_engine_pth, tokenizer_pth) -> None:
        self.IMAGE_TOKEN_INDEX = IMAGE_TOKEN_INDEX
        self.llm_engine_pth = llm_engine_pth
        
        torch.cuda.set_device(DEVICE_ID)
        self.device = f"cuda:{DEVICE_ID}"
        self.stream = torch.cuda.Stream(torch.cuda.current_device())
        torch.cuda.set_stream(self.stream)
        
        tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_pth, 
            model_max_length=TOKENIZER_MODEL_MAX_LENGTH, 
            padding_side="right", 
            use_fast=False
        )
        tokenizer.pad_token = tokenizer.unk_token
        tokenizer.padding_side = "right"
        self.tokenizer = tokenizer
        self.model_type = MODEL_TYPE
        self.init_llm()

    def init_llm(self):
        self.model = ModelRunner.from_dir(
            str(self.llm_engine_pth), 
            rank=0, 
            debug_mode=True, 
            stream=self.stream
        )
        self.model_config = self.model.session._model_config
        self.runtime_mapping = self.model.session.mapping

    def image_to_ptuning(self, input_ids, vision_embeded):
        input_ids = torch.tensor([[
            1, 319, 13563, 1546, 263, 12758, 1404, 322, 385, 23116,
            21082, 20255, 29889, 450, 20255, 4076, 8444, 29892, 13173, 29892,
            322, 1248, 568, 6089, 304, 278, 1404, 29915, 29879, 5155,
            29889, 3148, 1001, 29901, 29871, 32001, 32002, 32003, 32004, 32005,
            32006, 32007, 32008, 32009, 32010, 32011, 32012, 32013, 32014, 32015,
            32016, 32017, 32018, 32019, 32020, 32021, 32022, 32023, 32024, 32025,
            32026, 32027, 32028, 32029, 32030, 32031, 32032, 32033, 32034, 32035,
            32036, 32037, 32038, 32039, 32040, 32041, 32042, 32043, 32044, 32045,
            32046, 32047, 32048, 32049, 32050, 32051, 32052, 32053, 32054, 32055,
            32056, 32057, 32058, 32059, 32060, 32061, 32062, 32063, 32064, 32065,
            32066, 32067, 32068, 32069, 32070, 32071, 32072, 32073, 32074, 32075,
            32076, 32077, 32078, 32079, 32080, 32081, 32082, 32083, 32084, 32085,
            32086, 32087, 32088, 32089, 32090, 32091, 32092, 32093, 32094, 32095,
            32096, 32097, 32098, 32099, 32100, 32101, 32102, 32103, 32104, 32105,
            32106, 32107, 32108, 32109, 32110, 32111, 32112, 32113, 32114, 32115,
            32116, 32117, 32118, 32119, 32120, 32121, 32122, 32123, 32124, 32125,
            32126, 32127, 32128, 32129, 32130, 32131, 32132, 32133, 32134, 32135,
            32136, 32137, 32138, 32139, 32140, 32141, 32142, 32143, 32144, 32145,
            32146, 32147, 32148, 32149, 32150, 32151, 32152, 32153, 32154, 32155,
            32156, 32157, 32158, 32159, 32160, 32161, 32162, 32163, 32164, 32165,
            32166, 32167, 32168, 32169, 32170, 32171, 32172, 32173, 32174, 32175,
            32176, 32177, 32178, 32179, 32180, 32181, 32182, 32183, 32184, 32185,
            32186, 32187, 32188, 32189, 32190, 32191, 32192, 32193, 32194, 32195,
            32196, 32197, 32198, 32199, 32200, 32201, 32202, 32203, 32204, 32205,
            32206, 32207, 32208, 32209, 32210, 32211, 32212, 32213, 32214, 32215,
            32216, 32217, 32218, 32219, 32220, 32221, 32222, 32223, 32224, 32225,
            32226, 32227, 32228, 32229, 32230, 32231, 32232, 32233, 32234, 32235,
            32236, 32237, 32238, 32239, 32240, 32241, 32242, 32243, 32244, 32245,
            32246, 32247, 32248, 32249, 32250, 32251, 32252, 32253, 32254, 32255,
            32256, 32257, 29871, 13, 3492, 526, 19500, 297, 1809, 481,
            487, 29889, 3529, 3867, 278, 18987, 23324, 706, 363, 278,
            321, 1484, 1559, 1728, 9590, 29889, 319, 1799, 9047, 13566,
            29901, 2266, 338, 278, 18987, 23324, 706, 32000, 2
        ]], device='cuda:0', dtype=torch.int32)
        
        return input_ids, vision_embeded.reshape(1, -1, vision_embeded.shape[2])

    def generate(self, input_ids, vision_embeded, img_metas):
        input_ids, prompt_table = self.image_to_ptuning(input_ids, vision_embeded)
        
        input_ids = input_ids.contiguous().to(dtype=torch.int32)
        prompt_table = prompt_table.cuda().contiguous().to(dtype=torch.float16)

        self.model.generate(
            input_ids, 
            img_metas,
            prompt_table=prompt_table,
            end_id=self.tokenizer.eos_token_id,
            pad_id=self.tokenizer.pad_token_id,
            do_sample=True,
            temperature=0.1,
            top_p=0.75,
            num_beams=1,
            max_new_tokens=1,
            use_cache=False
        )

        self.stream.synchronize()
        return None


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
        self.config.builder_optimization_level = 5
        self.config.profiling_verbosity = trt.ProfilingVerbosity.DETAILED
        self.stream = cuda.Stream()
        self.cuda_ctx.pop()
        self.curr_scene_token = None
        self.start_timestamp = None
        self.bindings = {}
        
        self.cuda_graph = None
        self.graph_executed = None
        self.graph_captured = False
        self.warmup_done = False
        self.use_cuda_graph = True
        self.fixed_bindings = {}
        self.graph_stream = None
        
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
                self.bindings[tname] = torch.zeros(
                    list(self.engine.get_tensor_shape(tname)), 
                    dtype=torch.float32, 
                    device="cuda:0"
                ).contiguous()
        print("##### Input Bindings: ")
        print("\n".join(metas_in))
        print("##### Output Bindings: ")
        print("\n".join(metas_out))
        return
    
    def reset_cuda_graph(self):
        """重置 CUDA Graph，用于重新捕获"""
        self.cuda_graph = None
        self.graph_executed = None
        self.graph_captured = False
        self.warmup_done = False
        self.fixed_bindings = {}
        print("CUDA Graph 已重置")
    
    def set_cuda_graph_enabled(self, enabled):
        """设置是否使用 CUDA Graph"""
        self.use_cuda_graph = enabled
        if not enabled:
            self.reset_cuda_graph()
        print(f"CUDA Graph {'启用' if enabled else '禁用'}")

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
        save_path = os.path.join(NUS_TENSOR_PATH, f"{save_name}.pt")
        data = torch.load(save_path)

        self.bindings = {
            name: tensor.to("cuda:0").contiguous()
            for name, tensor in data.items()
            if isinstance(tensor, torch.Tensor)
        }
        
        input_ids = [[data.get("input_ids", None)]]
        
        self.cuda_ctx.push()
        
        for i in range(len(self.names)):
            tensor_name = str(self.names[i])
            if tensor_name in self.bindings:
                self.context.set_tensor_address(self.names[i], self.bindings[tensor_name].data_ptr())

        if self.use_cuda_graph and not self.graph_captured:
            print("初始化 CUDA Graph...")
            
            for name in self.names:
                if name in self.bindings:
                    tensor = self.bindings[name]
                    self.fixed_bindings[name] = torch.zeros_like(tensor, device="cuda:0").contiguous()
            
            self.graph_stream = torch.cuda.Stream()
            
            for i in range(len(self.names)):
                self.context.set_tensor_address(self.names[i], self.fixed_bindings[str(self.names[i])].data_ptr())
            
            self.context.execute_async_v3(stream_handle=self.graph_stream.cuda_stream)
            self.graph_stream.synchronize()
            
            torch.cuda.synchronize()
            self.cuda_graph = torch.cuda.CUDAGraph()
            
            with torch.cuda.graph(self.cuda_graph, stream=self.graph_stream):
                self.context.execute_async_v3(stream_handle=self.graph_stream.cuda_stream)
            
            self.graph_captured = True
            
        elif self.use_cuda_graph and self.graph_captured:
            for name in self.names:
                if name in self.bindings and name in self.fixed_bindings:
                    self.fixed_bindings[name].copy_(self.bindings[name])
            
            torch.cuda.synchronize()
            self.cuda_graph.replay()
            self.graph_stream.synchronize()
            torch.cuda.synchronize()
            
        else:
            torch.cuda.synchronize()
            self.context.execute_async_v3(stream_handle=self.stream.handle)
            self.stream.synchronize()
            torch.cuda.synchronize()
        
        self.cuda_ctx.pop()

        output_ids_lst = []
        for q_id, input_llm_id in enumerate(input_ids[0]):
            input_llm_id = input_llm_id.unsqueeze(0).to(device="cuda:0").contiguous()

            img_metas = {"save_name": save_name}
            
            output_ids = self.LLM_engine.generate(input_llm_id, vision_embeded, img_metas)
            output_ids_lst.append(output_ids)


def main():
    logger = trt.Logger(trt.Logger.VERBOSE)
    engine = InferTrt(logger, QA_SAVE_PATH, ENGINE_PATH)
    engine.read(ENGINE_PATH)
    engine.LLM_engine = InferTrtLLM(llm_engine_pth=LLM_ENGINE_PATH, tokenizer_pth=TOKENIZER_PATH)

    for filename in os.listdir(NUS_TENSOR_PATH):
        if filename.endswith(".pt"):
            prefix = os.path.splitext(filename)[0]
            engine.forward(prefix)
            print("=" * 50)


if __name__ == '__main__':
    torch.multiprocessing.set_start_method('fork')
    cuda.init()
    torch.cuda.init()
    main()
