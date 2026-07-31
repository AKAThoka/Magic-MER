"""
DataLoader.py - CFR 训练专用数据集加载器

核心设计理念（严格对齐 inference_hybird.py）:
1. 使用与推理完全相同的方式加载和处理数据
2. 支持 MERCaptionPlus 训练集（CFR 训练需要的样本）
3. 提供标准化的多模态数据读取接口
4. 与 GameCore.py 解耦，专注于数据加载职责

使用方式:
    loader = MERDataLoader(cfg_path="train_configs/cfr_training.yaml")
    loader.load_dataset("MERCaptionPlus", split="train")
    for sample in loader.iter_samples():
        sample_data = loader.read_sample(sample['name'])
        prompt = loader.get_prompt(sample, user_message="What is the emotion?")
"""

import os
import sys
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Iterator

import numpy as np
import torch

# 添加项目根目录到路径
SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))

# 导入 AffectGPT 组件
from my_affectgpt.common.config import Config
from my_affectgpt.common.registry import registry
from my_affectgpt.processors import BaseProcessor

# 导入数据集类（与 inference_hybird.py 第28行一致）
from my_affectgpt.datasets.builders.image_text_pair_builder import *

# 导入根目录的 config（包含 PATH_TO_RAW_VIDEO 等路径配置）
import config as path_config

logger = logging.getLogger("MERDataLoader")


class MERDataLoader:
    """
    MER 数据集加载器（CFR 训练专用）
    
    严格对齐 inference_hybird.py 的数据处理流程，确保：
    1. 使用相同的 face_or_frame 配置
    2. 使用相同的 processor 初始化方式
    3. 使用相同的多模态数据读取方法
    4. 使用相同的 prompt 构造方法
    
    职责边界：
    - 只负责数据加载和预处理
    - 不负责模型加载（由 LoadModel.py 负责）
    - 不负责推理调用（由 GameCore.py 负责）
    """
    
    def __init__(self, cfg_path: str, model_cfg=None):
        """
        初始化数据加载器
        
        :param cfg_path: 配置文件路径（如 train_configs/cfr_training.yaml）
        :param model_cfg: 模型配置（用于 Gemma3 等特殊模型的 Prompt 格式）
        """
        self.cfg_path = cfg_path
        self.model_cfg = model_cfg
        
        # 解析配置
        self._parse_config()
        
        # 数据集状态
        self.dataset_cls = None
        self.face_or_frame = None
        self.samples: List[Dict] = []
        
        logger.info(f"MERDataLoader 初始化完成: {cfg_path}")
    
    def _parse_config(self):
        """
        解析配置文件
        与 inference_hybird.py 第177-180行一致
        """
        # 模拟 argparse 参数
        class Args:
            def __init__(self, cfg_path):
                self.cfg_path = cfg_path
                self.options = None
        
        # 解析配置文件路径
        config_path = self.cfg_path
        if not os.path.isabs(config_path):
            possible_paths = [
                config_path,
                os.path.join(SCRIPT_DIR, config_path),
                os.path.join(PROJECT_ROOT, config_path),
            ]
            for p in possible_paths:
                if os.path.exists(p):
                    config_path = p
                    break
        
        args = Args(config_path)
        self.cfg = Config(args)
        self.datasets_cfg = self.cfg.datasets_cfg
        
        # 如果没有传入 model_cfg，使用配置文件中的
        if self.model_cfg is None:
            self.model_cfg = self.cfg.model_cfg
        
        # 获取 inference 配置（用于 processor）
        self.inference_cfg = getattr(self.cfg, 'inference_cfg', None)
    
    def _get_face_or_frame(self, outside_face_or_frame: Optional[str] = None) -> str:
        """
        获取 face_or_frame 配置
        与 inference_hybird.py 第92-100行完全一致
        """
        if outside_face_or_frame is not None:
            return outside_face_or_frame
        
        face_or_frame_candidates = []
        if 'mercaptionplus' in self.datasets_cfg:
            face_or_frame_candidates.append(self.datasets_cfg['mercaptionplus'].face_or_frame)
        if 'ovmerd' in self.datasets_cfg:
            face_or_frame_candidates.append(self.datasets_cfg['ovmerd'].face_or_frame)
        
        if len(set(face_or_frame_candidates)) == 1:
            return list(set(face_or_frame_candidates))[0]
        elif len(face_or_frame_candidates) > 0:
            logger.warning(f"多个 face_or_frame 配置不一致: {face_or_frame_candidates}，使用第一个")
            return face_or_frame_candidates[0]
        else:
            # 默认值
            return 'multiface_audio_face_text'
    
    def load_dataset(
        self, 
        dataset_name: str = "MERCaptionPlus",
        split: str = "train",
        face_or_frame: Optional[str] = None
    ):
        """
        加载数据集
        与 inference_hybird.py 第236-252行对齐
        
        :param dataset_name: 数据集名称（支持 MERCaptionPlus, MER2023, MER2024 等）
        :param split: 数据划分 ("train" 或 "test")
        :param face_or_frame: 多模态特征组合（如果为 None 则从配置读取）
        """
        logger.info(f"加载数据集: {dataset_name} (split={split})")
        
        # 1. 获取 face_or_frame 配置
        self.face_or_frame = self._get_face_or_frame(face_or_frame)
        logger.info(f"face_or_frame: {self.face_or_frame}")
        
        # 2. 获取数据集类（与 inference_hybird.py 第236行一致）
        self.dataset_cls = self._get_dataset_cls(dataset_name)
        if self.dataset_cls is None:
            raise ValueError(f"不支持的数据集: {dataset_name}")
        
        # 3. 设置必要属性（与 inference_hybird.py 第237-246行一致）
        self.dataset_cls.needed_data = self.dataset_cls.get_needed_data(self.face_or_frame)
        self.dataset_cls.vis_processor = BaseProcessor()
        self.dataset_cls.img_processor = BaseProcessor()
        
        # 4. 设置 processor（如果配置中指定）
        if self.inference_cfg is not None:
            vis_processor_cfg = self.inference_cfg.get("vis_processor")
            img_processor_cfg = self.inference_cfg.get("img_processor")
            if vis_processor_cfg is not None:
                self.dataset_cls.vis_processor = registry.get_processor_class(
                    vis_processor_cfg.train.name
                ).from_config(vis_processor_cfg.train)
            if img_processor_cfg is not None:
                self.dataset_cls.img_processor = registry.get_processor_class(
                    img_processor_cfg.train.name
                ).from_config(img_processor_cfg.train)
        
        # 5. 设置帧数
        self.dataset_cls.n_frms = self.model_cfg.vis_processor.train.n_frms
        
        # 6. 读取样本列表
        self._load_samples(split)
        
        logger.info(f"成功加载 {len(self.samples)} 个样本")
    
    def _get_dataset_cls(self, dataset_name: str):
        """
        获取数据集类实例
        与 inference_hybird.py 第104-116行对齐
        """
        # 标准化名称
        name_upper = dataset_name.upper()
        
        if name_upper == 'MERCAPTIONPLUS':
            return MERCaptionPlus_Dataset(model_cfg=self.model_cfg)
        elif name_upper == 'MER2023':
            return MER2023_Dataset(model_cfg=self.model_cfg)
        elif name_upper == 'MER2024':
            return MER2024_Dataset(model_cfg=self.model_cfg)
        elif name_upper == 'MELD':
            return MELD_Dataset(model_cfg=self.model_cfg)
        elif name_upper == 'IEMOCAPFOUR':
            return IEMOCAPFour_Dataset(model_cfg=self.model_cfg)
        elif name_upper == 'CMUMOSI':
            return CMUMOSI_Dataset(model_cfg=self.model_cfg)
        elif name_upper == 'CMUMOSEI':
            return CMUMOSEI_Dataset(model_cfg=self.model_cfg)
        elif name_upper == 'SIMS':
            return SIMS_Dataset(model_cfg=self.model_cfg)
        elif name_upper == 'SIMSV2':
            return SIMSv2_Dataset(model_cfg=self.model_cfg)
        elif name_upper == 'MER2025OV':
            return MER2025OV_Dataset(model_cfg=self.model_cfg)
        elif name_upper == 'OVMERDPLUS':
            return OVMERDPlus_Dataset(model_cfg=self.model_cfg)
        else:
            logger.error(f"未知的数据集类型: {dataset_name}")
            return None
    
    def _load_samples(self, split: str = "train"):
        """
        加载样本列表
        
        对于 MERCaptionPlus：使用 annotation 属性（训练集）
        对于其他数据集：使用 read_test_names()
        """
        self.samples = []
        
        if hasattr(self.dataset_cls, 'annotation') and split == "train":
            # MERCaptionPlus 等训练数据集使用 annotation
            for sample in self.dataset_cls.annotation:
                sample_info = {
                    'name': sample.get('name', ''),
                    'subtitle': sample.get('subtitle', ''),
                    'label': sample.get('ovlabel') or sample.get('emotion', 'unknown'),
                    'description': sample.get('description', ''),
                }
                self.samples.append(sample_info)
        
        elif hasattr(self.dataset_cls, 'read_test_names'):
            # 处理 MER2025OV 等数据集（支持 train/test split）
            test_names = self.dataset_cls.read_test_names()
            name2subtitle = getattr(self.dataset_cls, 'name2subtitle', {})
            
            # 尝试获取真值标签（供 CFR 训练使用）
            name2gt = {}
            if hasattr(self.dataset_cls, 'get_test_name2gt'):
                name2gt = self.dataset_cls.get_test_name2gt()
            
            for name in test_names:
                sample_info = {
                    'name': name,
                    'subtitle': name2subtitle.get(name, ''),
                    'label': name2gt.get(name, 'unknown'),
                }
                self.samples.append(sample_info)
        
        else:
            logger.warning(f"无法加载样本: dataset_cls 不符合预期的加载模式")
    
    def read_sample(self, sample_name: str) -> Dict[str, Any]:
        """
        读取单个样本的多模态数据
        与 inference_hybird.py 第282行完全一致
        
        :param sample_name: 样本名称
        :return: sample_data 字典（包含 frame, face, audio 等）
        """
        if self.dataset_cls is None:
            raise RuntimeError("数据集未加载，请先调用 load_dataset()")
        
        # 构造样本字典
        sample = {'name': sample_name}
        
        # 获取各模态数据的路径
        video_path, image_path, audio_path, face_npy = None, None, None, None
        
        if hasattr(self.dataset_cls, '_get_video_path'):
            video_path = self.dataset_cls._get_video_path(sample)
        if hasattr(self.dataset_cls, '_get_audio_path'):
            audio_path = self.dataset_cls._get_audio_path(sample)
        if hasattr(self.dataset_cls, '_get_face_path'):
            face_npy = self.dataset_cls._get_face_path(sample)
        if hasattr(self.dataset_cls, '_get_image_path'):
            image_path = self.dataset_cls._get_image_path(sample)
        
        # 调用数据集类的读取方法（与 inference_hybird.py 第282行完全一致）
        sample_data = self.dataset_cls.read_frame_face_audio_text(
            video_path, face_npy, audio_path, image_path
        )
        
        return sample_data
    
    def get_sample_paths(self, sample_name: str) -> Tuple[str, str, str, str]:
        """
        获取样本的多模态文件路径
        
        :param sample_name: 样本名称
        :return: (video_path, audio_path, face_path, image_path)
        """
        sample = {'name': sample_name}
        
        video_path = self.dataset_cls._get_video_path(sample) if hasattr(self.dataset_cls, '_get_video_path') else None
        audio_path = self.dataset_cls._get_audio_path(sample) if hasattr(self.dataset_cls, '_get_audio_path') else None
        face_path = self.dataset_cls._get_face_path(sample) if hasattr(self.dataset_cls, '_get_face_path') else None
        image_path = self.dataset_cls._get_image_path(sample) if hasattr(self.dataset_cls, '_get_image_path') else None
        
        return video_path, audio_path, face_path, image_path
    
    def get_prompt(
        self, 
        sample: Dict[str, Any],
        user_message: str,
        zeroshot: bool = True
    ) -> str:
        """
        构造多模态 Prompt
        与 inference_hybird.py 第368-369行完全一致
        
        :param sample: 样本信息字典
        :param user_message: 用户问题
        :param zeroshot: 是否使用 zeroshot 问题
        :return: 完整的多模态 prompt
        """
        if self.dataset_cls is None:
            raise RuntimeError("数据集未加载，请先调用 load_dataset()")
        
        subtitle = sample.get('subtitle', '')
        
        # 与 inference_hybird.py 第368行一致
        prompt = self.dataset_cls.get_prompt_for_multimodal(
            self.face_or_frame,
            subtitle,
            user_message
        )
        
        return prompt
    
    def get_default_user_message(self, zeroshot: bool = True) -> str:
        """
        获取默认的用户问题
        与 inference_hybird.py 第120-129行对齐
        
        :param zeroshot: 是否使用 zeroshot 问题
        :return: 用户问题字符串
        """
        if self.dataset_cls is None:
            return "Please analyze the emotional content of this video."
        
        if zeroshot:
            return self.dataset_cls.func_get_qa_ovlabel(sample=None, question_only=True)
        else:
            # 可以根据需要返回其他默认问题
            return self.dataset_cls.func_get_qa_ovlabel(sample=None, question_only=True)
    
    def iter_samples(self) -> Iterator[Dict[str, Any]]:
        """
        迭代所有样本
        
        :yield: 样本信息字典
        """
        for sample in self.samples:
            yield sample
    
    def get_sample_by_name(self, name: str) -> Optional[Dict[str, Any]]:
        """
        根据名称获取样本
        
        :param name: 样本名称
        :return: 样本信息字典，如果不存在返回 None
        """
        for sample in self.samples:
            if sample['name'] == name:
                return sample
        return None
    
    def get_prompt_for_llm_type(
        self,
        sample: Dict[str, Any],
        user_message: str,
        llm_type: str = "Qwen25"
    ) -> str:
        """
        根据 LLM 类型构造 prompt（解决 Gemma 格式问题）
        
        不同 LLM 需要不同的 prompt 格式：
        - Qwen/Llama: ###Human: ... ###Assistant: ...
        - Gemma: <start_of_turn>user\n ... <end_of_turn>\n<start_of_turn>model\n
        
        :param sample: 样本信息字典
        :param user_message: 用户问题
        :param llm_type: LLM 类型 ("Qwen25", "Llama31", "Gemma3" 等)
        :return: 完整的多模态 prompt
        """
        if self.dataset_cls is None:
            raise RuntimeError("数据集未加载，请先调用 load_dataset()")
        
        subtitle = sample.get('subtitle', '')
        
        # 根据 llm_type 确定 prompt 格式
        is_gemma = llm_type and llm_type.lower() in ['gemma3', 'gemma', 'gemma2']
        
        if is_gemma:
            prefix = "<start_of_turn>user\n"
            suffix = "<end_of_turn>\n<start_of_turn>model\n"
        else:
            prefix = "###Human: "
            suffix = " ###Assistant: "
        
        # 构造 prompt（复用 face_or_frame 逻辑）
        prompt = self._build_prompt_with_format(
            self.face_or_frame, subtitle, user_message, prefix, suffix
        )
        
        return prompt
    
    def _build_prompt_with_format(
        self,
        face_or_frame: str,
        subtitle: str,
        user_message: str,
        prefix: str,
        suffix: str
    ) -> str:
        """
        使用指定格式构造 prompt
        
        复用 base_dataset.get_prompt_for_multimodal 的逻辑，但允许自定义 prefix/suffix
        """
        # 与 base_dataset.py get_prompt_for_multimodal 保持一致
        if face_or_frame == 'faceframe':
            prompt = f"{prefix}The audio content is as follows: <Audio><AudioHere></Audio>. " \
                    + f"Meanwhile, we uniformly sample raw frames from the video: <Video><FrameHere></Video>. "  \
                    + f"Additionally, we uniformly sample raw frames from the video and extract faces from these frames: <Video><FaceHere></Video>. "  \
                    + f"The subtitle of this video is: <Subtitle>{subtitle}</Subtitle>. " \
                    + f"Now, please answer my question based on all the provided information. {user_message}{suffix}"
        elif face_or_frame == 'face':
            prompt = f"{prefix}The audio content is as follows: <Audio><AudioHere></Audio>. " \
                    + f"Meanwhile, we uniformly sample raw frames from the video and extract faces from these frames: <Video><FaceHere></Video>. "  \
                    + f"The subtitle of this video is: <Subtitle>{subtitle}</Subtitle>. " \
                    + f"Now, please answer my question based on all the provided information. {user_message}{suffix}"
        elif face_or_frame == 'frame':
            prompt = f"{prefix}The audio content is as follows: <Audio><AudioHere></Audio>. " \
                    + f"Meanwhile, we uniformly sample raw frames from the video: <Video><FrameHere></Video>. "  \
                    + f"The subtitle of this video is: <Subtitle>{subtitle}</Subtitle>. " \
                    + f"Now, please answer my question based on all the provided information. {user_message}{suffix}"
        elif face_or_frame.startswith('multiface'):
            prompt = f"{prefix}The audio content is as follows: <Audio><AudioHere></Audio>. " \
                    + f"Meanwhile, we uniformly sample raw frames from the video and extract faces from these frames: <Video><FaceHere></Video>. "  \
                    + f"<Multi><MultiHere></Multi>. "  \
                    + f"The subtitle of this video is: <Subtitle>{subtitle}</Subtitle>. " \
                    + f"Now, please answer my question based on all the provided information. {user_message}{suffix}"
        elif face_or_frame.startswith('multiframe'):
            prompt = f"{prefix}The audio content is as follows: <Audio><AudioHere></Audio>. " \
                    + f"Meanwhile, we uniformly sample raw frames from the video: <Video><FrameHere></Video>. "  \
                    + f"<Multi><MultiHere></Multi>. "  \
                    + f"The subtitle of this video is: <Subtitle>{subtitle}</Subtitle>. " \
                    + f"Now, please answer my question based on all the provided information. {user_message}{suffix}"
        else:
            # 默认格式
            prompt = f"{prefix}The audio content is as follows: <Audio><AudioHere></Audio>. " \
                    + f"Meanwhile, we uniformly sample raw frames from the video and extract faces from these frames: <Video><FaceHere></Video>. "  \
                    + f"The subtitle of this video is: <Subtitle>{subtitle}</Subtitle>. " \
                    + f"Now, please answer my question based on all the provided information. {user_message}{suffix}"
        
        return prompt
    
    def __len__(self) -> int:
        """返回样本数量"""
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """根据索引获取样本"""
        return self.samples[idx]


# ==========================================
# 便捷工厂函数
# ==========================================

def create_mer_dataloader(
    cfg_path: str,
    dataset_name: str = "MERCaptionPlus",
    split: str = "train",
    model_cfg=None
) -> MERDataLoader:
    """
    创建 MER 数据加载器的便捷函数
    
    :param cfg_path: 配置文件路径
    :param dataset_name: 数据集名称
    :param split: 数据划分
    :param model_cfg: 模型配置
    :return: MERDataLoader 实例
    """
    loader = MERDataLoader(cfg_path, model_cfg=model_cfg)
    loader.load_dataset(dataset_name, split)
    return loader


# ==========================================
# 测试代码
# ==========================================

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    print("=" * 60)
    print("MERDataLoader 测试")
    print("=" * 60)
    
    # 测试配置文件路径
    cfg_path = "train_configs/cfr_training.yaml"
    
    try:
        # 创建数据加载器
        loader = MERDataLoader(cfg_path)
        print(f"✓ 配置加载成功")
        
        # 加载数据集
        loader.load_dataset("MERCaptionPlus", split="train")
        print(f"✓ 数据集加载成功: {len(loader)} 个样本")
        
        # 获取第一个样本
        if len(loader) > 0:
            sample = loader[0]
            print(f"✓ 第一个样本: {sample['name']}")
            
            # 获取 prompt
            user_message = loader.get_default_user_message()
            prompt = loader.get_prompt(sample, user_message)
            print(f"✓ Prompt 长度: {len(prompt)} 字符")
            print(f"  Prompt 前100字符: {prompt[:100]}...")
        
    except Exception as e:
        print(f"✗ 测试失败: {e}")
        import traceback
        traceback.print_exc()
