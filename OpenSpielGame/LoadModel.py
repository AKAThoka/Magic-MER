'''
LoadModel.py 核心功能:
1. 负责加载 AffectGPT 模型，支持多模型、多 GPU 加载。
2. 封装了模型的初始化、权重加载和 Chat 对象创建过程。
3. 依赖文件:
    - my_affectgpt/common/config.py: 配置解析
    - my_affectgpt/common/registry.py: 模型注册表
    - my_affectgpt/conversation/conversation_video.py: Chat 类 (推理核心)
    - my_affectgpt/models/*: 模型定义                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               
    - my_affectgpt/processors/*: 数据预处理器

    - train_configs/AffectGame.yaml: 配置文件
    - game/GameCore.py: 多模态玩家类
'''

import os
import copy
import torch
import argparse
from omegaconf import OmegaConf
from my_affectgpt.common.config import Config
from my_affectgpt.common.registry import registry
from my_affectgpt.conversation.conversation_video import Chat
from my_affectgpt.models import *
from my_affectgpt.processors import *

class AffectGPTLoader:
    """
    AffectGPT 模型加载器
    负责加载、管理和分发多个 AffectGPT 模型实例到指定的 GPU 设备。
    """
    def __init__(self, cfg_path, options=None):
        """
        初始化加载器
        :param cfg_path: 配置文件路径 (yaml)
        :param options: 覆盖配置的选项列表
        """
        # 模拟 argparse 解析，以便复用 Config 类
        parser = argparse.ArgumentParser()
        parser.add_argument("--cfg-path", default=cfg_path)
        parser.add_argument("--options", nargs="+", default=options)
        args = parser.parse_args([])
        
        # 加载全局配置
        self.cfg = Config(args)
        self.model_cfg = self.cfg.model_cfg
        self.inference_cfg = self.cfg.inference_cfg
        
        # 存储已加载的模型实例 {model_name: chat_instance}
        self.models = {}
        
    def load_models(self, model_configs):
        """
        加载多个模型实例（支持多 GPU 隔离）
        
        :param model_configs: 模型配置列表，每个元素包含:
            - 'name': str - 模型名称
            - 'ckpt_path': str - 权重路径
            - 'device_id': int - GPU 设备 ID (0, 1, 2, ...)
            - 'llm_type': str (可选) - LLM 类型 (qwen, llama, etc.)
        
        注意: 每个模型使用独立的配置副本，避免多 GPU 加载时配置互相覆盖
        """
        print(f"======== Start loading {len(model_configs)} models ========")
        
        for config in model_configs:
            name = config['name']
            ckpt_path = config['ckpt_path']
            device_id = config.get('device_id', 0)
            device = f'cuda:{device_id}'
            
            print(f"Loading model: {name} to {device} ...")
            print(f"Checkpoint path: {ckpt_path}")

            # 🔧 关键修复: 为每个模型创建独立的配置副本
            # 避免多 GPU 加载时配置互相覆盖
            model_cfg_copy = copy.deepcopy(self.model_cfg)
            
            # 1. 更新当前模型的权重路径配置
            model_cfg_copy.ckpt_3 = ckpt_path
            
            # 1.1 动态更新 LLM 类型 (如果配置中指定了 llm_type)
            # 这对于同时加载 Qwen 和 Llama 至关重要，因为它们需要不同的 Tokenizer 和模型结构
            llm_type = config.get('llm_type', None)
            if llm_type:
                print(f"Dynamically updating LLM type to: {llm_type}")
                model_cfg_copy.llama_model = llm_type
            
            # 2. 初始化模型架构（使用隔离的配置）
            # 关键：根据 llm_type 选择正确的模型类
            # - Gemma3 需要使用 affectgpt_gemma 类（因为模型结构不同）
            # - Qwen/Llama 使用标准的 affectgpt 类
            if llm_type and llm_type.lower() in ['gemma3', 'gemma']:
                model_arch = 'affectgpt_gemma'
                print(f"Using AffectGPTGemma class for Gemma3 model")
            else:
                model_arch = model_cfg_copy.arch  # 默认使用配置中的 arch
            
            model_cls = registry.get_model_class(model_arch)
            model = model_cls.from_config(model_cfg_copy)
            
            # 3. 加载权重
            if os.path.exists(ckpt_path):
                ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
                msg = model.load_state_dict(ckpt['model'], strict=False)
                print(f"Checkpoint loaded: {msg}")
            else:
                print(f"Warning: Checkpoint file does not exist at {ckpt_path}, using random initialization")

            # 4. 移动到指定 GPU 并设为评估模式
            model = model.to(device).eval()
            
            # 5. 封装为 Chat 对象 (包含预处理和推理逻辑)
            chat = Chat(model, model_cfg_copy, device=device)
            
            self.models[name] = chat
            print(f"Model {name} loaded successfully on {device}!\n")
            
        print("======== All models loaded ========")

    def get_model(self, name):
        """获取指定名称的模型实例"""
        return self.models.get(name)

    def load_processors(self):
        """
        初始化并返回预处理器
        :return: (vis_processor, img_processor, n_frms)
        """
        print("======== Initializing Processors ========")
        inference_cfg = self.get_processor_cfg()
        vis_processor_cfg = inference_cfg.get("vis_processor")
        img_processor_cfg = inference_cfg.get("img_processor")
        
        vis_processor = None
        if vis_processor_cfg:
            vis_processor = load_processor(vis_processor_cfg.train.name, vis_processor_cfg.train)
            
        img_processor = None
        if img_processor_cfg:
            img_processor = load_processor(img_processor_cfg.train.name, img_processor_cfg.train)
            
        # 从模型配置中获取帧数，默认为 8
        n_frms = getattr(self.model_cfg.vis_processor.train, 'n_frms', 8)
        
        print(f"Processors initialized: vis={vis_processor is not None}, img={img_processor is not None}, n_frms={n_frms}")
        return vis_processor, img_processor, n_frms

    def get_processor_cfg(self):
        """获取预处理器配置，供外部数据加载使用"""
        return self.inference_cfg

# 示例用法 (测试)
# if __name__ == "__main__":
#     # 假设配置文件路径
#     cfg_path = "train_configs/AffectGame.yaml"
    
#     # 定义要加载的模型
#     # 这里演示加载两个模型到不同的 GPU (如果只有一个 GPU，device_id 都设为 0)
#     models_to_load = [
#         {
#             "name": "AffectGPT-Qwen",
#             "ckpt_path": "output/results/qwen_ckpt.pth", # 示例路径
#             "device_id": 0
#         },
#         {
#             "name": "AffectGPT-Llama", 
#             "ckpt_path": "output/results/llama_ckpt.pth", # 示例路径
#             "device_id": 0 # 如果有双卡可设为 1
#         }
#     ]
    
#     try:
#         loader = AffectGPTLoader(cfg_path)
#         # loader.load_models(models_to_load) # 实际运行时取消注释
#     except Exception as e:
#         print(f"初始化失败 (预期内，因为路径可能不存在): {e}")
