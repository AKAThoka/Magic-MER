"""
train_cfr.py - MER 博弈 CFR 训练主脚本

功能：
1. 加载多个 AffectGPT 模型（支持多 GPU）
2. 加载 MER 数据集（复用 AffectGPT 的数据集加载机制）
3. 运行 CFR 训练循环
4. 保存训练策略和日志

使用方式：
    python train_cfr.py --config train_configs/AffectGame.yaml --num_epochs 10

"""

import os
import sys
import json
import time
import random
import logging
import argparse
import warnings
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import torch

# 抑制 transformers 的警告信息
warnings.filterwarnings("ignore", message=".*do_sample.*")
warnings.filterwarnings("ignore", message=".*pad_token_id.*")
warnings.filterwarnings("ignore", message=".*Setting.*pad_token_id.*")
warnings.filterwarnings("ignore", message=".*open-end generation.*")

# 抑制 transformers 的 logging 输出
import logging as std_logging
std_logging.getLogger("transformers").setLevel(std_logging.ERROR)
std_logging.getLogger("transformers.generation").setLevel(std_logging.ERROR)
std_logging.getLogger("transformers.generation.utils").setLevel(std_logging.ERROR)

# 添加项目根目录到路径
SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))

# 导入 AffectGPT 数据集和任务系统
import my_affectgpt.tasks as tasks
from my_affectgpt.common.config import Config
from my_affectgpt.common.registry import registry
from my_affectgpt.tasks import *
from my_affectgpt.models import *
from my_affectgpt.runners import *
from my_affectgpt.processors import *
from my_affectgpt.datasets.builders import *

# 导入 OpenSpielGame 模块
from OpenSpielGame.LoadModel import AffectGPTLoader
from OpenSpielGame.MERgame import MERPlayer, Referee, MERGameDriver, GameTrajectory
from OpenSpielGame.solver import MERCFRSolver, create_mer_solver, MERAction
from OpenSpielGame.confidence_utils import extract_confidence
from OpenSpielGame.DataLoader import MERDataLoader, create_mer_dataloader
from OpenSpielGame.LLMInference import LLMInferenceWrapper, create_llm_wrapper
from OpenSpielGame.prompt import (
    CFRGamePromptFactory, 
    create_cfr_prompt_functions,
    get_perception_prompt,
    get_reevaluation_prompt,
    get_decision_prompt
)
from OpenSpielGame.LabelExtractor import (
    LabelExtractor,
    OVLabelExtractor,
    create_label_extractor,
)

# 配置日志
class _StreamToLogger:
    """将 stdout/stderr 重定向到 logger"""

    def __init__(self, logger: logging.Logger, level: int):
        self.logger = logger
        self.level = level
        self._buffer = ""

    def write(self, message: str):
        if not message:
            return
        self._buffer += message
        while "\n" in self._buffer:
            line, self._buffer = self._buffer.split("\n", 1)
            if line.strip():
                self.logger.log(self.level, line.rstrip())

    def flush(self):
        if self._buffer.strip():
            self.logger.log(self.level, self._buffer.rstrip())
        self._buffer = ""


def setup_logging(log_dir: str, log_level: str = "INFO", redirect_stdio: bool = True) -> logging.Logger:
    """配置日志系统（可捕获 stdout/stderr）"""
    os.makedirs(log_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"train_cfr_{timestamp}.log")

    # 创建 logger
    logger = logging.getLogger("CFRTrainer")
    logger.setLevel(getattr(logging, log_level.upper()))

    if not logger.handlers:
        # 文件 handler
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(logging.DEBUG)
        file_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(file_formatter)

        # 控制台 handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s'
        )
        console_handler.setFormatter(console_formatter)

        logger.addHandler(file_handler)
        logger.addHandler(console_handler)

        # 同时配置子模块的日志（添加控制台和文件输出）
        for module_name in ["MERGame", "MERSolver", "ConfidenceUtils"]:
            module_logger = logging.getLogger(module_name)
            module_logger.setLevel(logging.INFO)
            module_logger.addHandler(file_handler)
            module_logger.addHandler(console_handler)  # 添加控制台输出

        logger.info(f"日志文件: {log_file}")

    if redirect_stdio:
        sys.stdout = _StreamToLogger(logger, logging.INFO)
        sys.stderr = _StreamToLogger(logger, logging.ERROR)

    return logger


class CFRTrainer:
    """
    CFR 训练器主类
    
    负责协调模型加载、数据加载、训练循环和策略保存
    """
    
    def __init__(
        self,
        config_path: str,
        output_dir: str = "output/cfr_training",
        num_epochs: int = 1,
        warmup_iterations: int = 1000,
        save_every: int = 100,
        log_level: str = "INFO",
        seed: int = 42,
        max_samples: int = None,
        run_dir: Optional[str] = None,
        log_dir: Optional[str] = None,
        logger: Optional[logging.Logger] = None
    ):
        """
        初始化训练器
        
        :param config_path: 配置文件路径 (yaml)
        :param output_dir: 输出目录
        :param num_epochs: 训练轮数
        :param warmup_iterations: CFR 热身迭代次数
        :param save_every: 每隔多少样本保存一次检查点
        :param log_level: 日志级别
        :param seed: 随机种子
        :param max_samples: 每个 epoch 最多使用的样本数（None 表示使用全部）
        """
        self.config_path = config_path
        self.output_dir = output_dir
        self.num_epochs = num_epochs
        self.warmup_iterations = warmup_iterations
        self.save_every = save_every
        self.max_samples = max_samples
        
        # 设置随机种子
        self.seed = seed
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        
        # 创建带时间戳的输出目录
        if run_dir is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.run_dir = os.path.join(output_dir, f"cfr_{timestamp}")
        else:
            self.run_dir = run_dir
        os.makedirs(self.run_dir, exist_ok=True)
        self.checkpoint_dir = os.path.join(self.run_dir, "checkpoints")
        self.log_dir = log_dir or os.path.join(self.run_dir, "logs")
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        os.makedirs(self.log_dir, exist_ok=True)

        # 设置日志（允许外部注入）
        self.logger = logger or setup_logging(self.log_dir, log_level)
        
        # 训练状态
        self.models: Dict[str, Any] = {}
        self.players: List[MERPlayer] = []
        self.cfr_solver: Optional[MERCFRSolver] = None
        self.driver: Optional[MERGameDriver] = None
        self.dataset: List[Dict] = []
        
        # 统计信息
        self.stats = {
            "total_samples": 0,
            "total_games": 0,
            "wins_by_player": {0: 0, 1: 0, 2: 0},
            "correct_predictions": 0,
            "epoch_stats": []
        }
        
        # ε-CCE 遗憾曲线数据（每 log_interval 步记录一次）
        self.regret_history: List[Dict] = []

        # 四维收敛性指标并行历史
        # 每条记录包含：raw_epsilon / norm_epsilon / strategy_stability / strategy_entropy / lazy_exploration
        self.conv_metrics_history: List[Dict] = []
        # 策略稳定性辅助状态
        self._prev_strategy_snapshot: Dict = {}
        self._prev_infoset_count: int = 0
        # payoff 量程（用于归一化 ε-CCE，即 u_max - u_min）
        self._payoff_range: float = 100.0

        # 【新增】数据加载器和推理封装器（使用 inference 风格）
        self.data_loader: Optional[MERDataLoader] = None
        self.llm_wrappers: Dict[str, LLMInferenceWrapper] = {}
        
        # 【新增】标签提取器（用于从 LLM 输出中提取情感标签）
        self.label_extractor: Optional[LabelExtractor] = None
        
        self.logger.info("=" * 60)
        self.logger.info("CFR Trainer 初始化")
        self.logger.info(f"配置文件: {config_path}")
        self.logger.info(f"输出目录: {output_dir}")
        self.logger.info(f"训练轮数: {num_epochs}")
        self.logger.info(f"热身迭代: {warmup_iterations}")
        self.logger.info("=" * 60)
    
    def load_config(self) -> Dict:
        """加载配置文件"""
        from omegaconf import OmegaConf
        
        # 解析配置文件路径：支持相对路径和绝对路径
        config_path = self.config_path
        if not os.path.isabs(config_path):
            # 如果是相对路径，尝试多个可能的位置
            possible_paths = [
                config_path,  # 当前目录
                os.path.join(SCRIPT_DIR, config_path),  # OpenSpielGame 目录
                os.path.join(PROJECT_ROOT, config_path),  # 项目根目录
            ]
            for p in possible_paths:
                if os.path.exists(p):
                    config_path = p
                    break
            else:
                raise FileNotFoundError(
                    f"配置文件未找到: {self.config_path}\n"
                    f"尝试的路径: {possible_paths}"
                )
        
        self.logger.info(f"加载配置: {config_path}")
        cfg = OmegaConf.load(config_path)
        
        # 转换为字典
        config = OmegaConf.to_container(cfg, resolve=True)
        self.config = config
        
        return config
    
    def load_models(self, model_configs: List[Dict]):
        """
        加载 AffectGPT 模型
        
        :param model_configs: 模型配置列表
        """
        self.logger.info("=" * 40)
        self.logger.info("开始加载模型...")
        self.logger.info("=" * 40)
        
        # 创建模型加载器
        loader = AffectGPTLoader(self.config_path)
        
        # 加载模型
        loader.load_models(model_configs)
        
        # 保存模型引用
        self.models = loader.models
        self.loader = loader
        
        self.logger.info(f"成功加载 {len(self.models)} 个模型")
        for name in self.models.keys():
            self.logger.info(f"  - {name}")
    
    def create_llm_callable(self, chat_instance, player_name: str, llm_type: str = "Qwen25"):
        """
        创建 LLM 可调用包装器
        
        【重写】使用 LLMInferenceWrapper 和 MERDataLoader
        与 inference_hybird.py 完全对齐
        
        【修复】添加 llm_type 参数，用于为不同模型（Qwen/Llama/Gemma）
        生成对应格式的 prompt
        
        :param chat_instance: Chat 对象
        :param llm_type: LLM 类型（"Qwen25", "Llama31", "Gemma3" 等）
        :param player_name: 玩家名称（用于日志）
        :return: 符合 MERPlayer 接口的 callable
        """
        logger = self.logger
        
        # 确保数据加载器已初始化
        if self.data_loader is None:
            raise RuntimeError("数据加载器未初始化！请先调用 load_dataset()")
        
        # 获取 face_or_frame 配置
        face_or_frame = self.data_loader.face_or_frame
        
        # 创建 LLM 推理封装器
        wrapper = LLMInferenceWrapper(
            chat_instance=chat_instance,
            face_or_frame=face_or_frame,
            name=player_name
        )
        
        # 保存引用
        self.llm_wrappers[player_name] = wrapper
        
        # 获取数据加载器引用
        data_loader = self.data_loader
        
        # 根据 llm_type 确定 prompt 格式前后缀
        is_gemma = llm_type and llm_type.lower() in ['gemma3', 'gemma', 'gemma2']
        if is_gemma:
            llm_prefix = "<start_of_turn>user\n"
            llm_suffix = "<end_of_turn>\n<start_of_turn>model\n"
        else:
            llm_prefix = "###Human: "
            llm_suffix = " ###Assistant: "
        
        def llm_callable(
            user_message: str,
            video_path: Optional[str] = None,
            audio_path: Optional[str] = None,
            face_path: Optional[str] = None,
            sample_name: Optional[str] = None,
            subtitle: Optional[str] = None
        ) -> Tuple[str, np.ndarray]:
            """
            调用 AffectGPT 模型
            
            【关键设计】只有初始感知阶段提供多模态数据：
            
            1. 感知阶段 (Perception)：
               - 需要完整多模态 prompt（音频、视频、字幕描述）
               - 需要传入多模态特征（sample_data）
               
            2. 重评估阶段 (Re-evaluation)：
               - 只需要 LLM 格式包装
               - 不需要多模态描述和特征（模型已有记忆）
               
            3. 决策阶段 (Decision)：
               - 只需要 LLM 格式包装
               - 不需要多模态描述和特征
            
            检测逻辑：通过关键词判断当前阶段
            
            :param user_message: 博弈问题或决策 prompt
            :param video_path: 视频路径（未使用，保留兼容性）
            :param audio_path: 音频路径（未使用，保留兼容性）
            :param face_path: 人脸路径（未使用，保留兼容性）
            :param sample_name: 样本名称（用于读取数据）
            :param subtitle: 字幕（用于 prompt 构造）
            :return: (response, logits)
            """
            try:
                # 【关键修复】检测当前阶段
                # 非感知阶段的关键词列表
                non_perception_keywords = [
                    # 决策阶段
                    "Texas Hold'em", "Strategy Advice", "Your Decision", 
                    "FOLD", "CHECK", "CALL", "RAISE", "Nash Equilibrium",
                    # 重评估阶段
                    "re-evaluate", "Re-evaluate", "other players", "Other players",
                    "revealed", "predictions", "confidence"
                ]
                
                is_perception_phase = not any(
                    keyword in user_message for keyword in non_perception_keywords
                )
                
                if is_perception_phase:
                    # ========== 感知阶段 ==========
                    # 需要多模态数据 + 多模态 prompt 模板
                    # logger.info(f"  │  📹 [{player_name}] 感知阶段：使用多模态输入")
                    
                    # 读取多模态数据
                    if sample_name is not None:
                        sample_data = data_loader.read_sample(sample_name)
                    else:
                        sample_data = {
                            'frame': None, 'raw_frame': None,
                            'face': None, 'raw_face': None,
                            'audio': None, 'raw_audio': None,
                            'image': None, 'raw_image': None,
                        }
                    
                    # 构造完整多模态 prompt
                    sample = {'name': sample_name or '', 'subtitle': subtitle or ''}
                    prompt = data_loader.get_prompt_for_llm_type(sample, user_message, llm_type)
                    
                else:
                    # ========== 重评估/决策阶段 ==========
                    # 不需要多模态数据，只需要 LLM 格式包装
                    # stage = "决策" if "Texas Hold'em" in user_message else "重评估"
                    # logger.info(f"  │  📝 [{player_name}] {stage}阶段：纯文本输入（无多模态）")
                    
                    # 空的多模态数据（不传入视频/音频特征）
                    sample_data = {
                        'frame': None, 'raw_frame': None,
                        'face': None, 'raw_face': None,
                        'audio': None, 'raw_audio': None,
                        'image': None, 'raw_image': None,
                    }
                    
                    # 只添加 LLM 格式前后缀，不添加多模态描述
                    prompt = f"{llm_prefix}{user_message}{llm_suffix}"
                
                # 【调试】打印实际发送给 LLM 的 prompt（前300字符）
                logger.debug(f"  │  🔍 [{player_name}] 实际LLM输入 (前300字符):")
                logger.debug(f"  │  {prompt[:300]}...")
                
                # Step 3: 执行推理
                response, logits = wrapper.inference(sample_data, prompt)
                
                # 如果没有 logits，返回模拟值
                if logits is None:
                    logits = np.random.randn(4) * 0.5 + 1.0
                
                return (response, logits)
                
            except Exception as e:
                logger.error(f"[{player_name}] LLM 调用失败: {e}")
                import traceback
                traceback.print_exc()
                return ("neutral", np.array([0.25, 0.25, 0.25, 0.25]))
        
        return llm_callable
    
    def setup_players(self, model_configs: List[Dict]):
        """
        设置玩家
        
        :param model_configs: 模型配置
        """
        self.logger.info("创建玩家...")
        
        self.players = []
        for i, config in enumerate(model_configs):
            name = config['name']
            chat = self.models.get(name)
            
            if chat is None:
                raise ValueError(
                    f"模型 {name} 未找到！请检查:\n"
                    f"  1. 模型权重路径是否正确: {config.get('ckpt_path', '未指定')}\n"
                    f"  2. LLM 类型是否匹配: {config.get('llm_type', '未指定')}"
                )
            
            # 【修复】从配置中获取 llm_type，用于生成对应格式的 prompt
            llm_type = config.get('llm_type', 'Qwen25')
            llm_callable = self.create_llm_callable(chat, name, llm_type=llm_type)
            
            player = MERPlayer(
                player_id=i,
                name=name,
                llm_callable=llm_callable
            )
            self.players.append(player)
            self.logger.info(f"  玩家 {i}: {name}")
    
    def setup_cfr_solver(self):
        """初始化 CFR 求解器"""
        self.logger.info("初始化 CFR 求解器...")
        
        # 从配置读取参数
        cfr_config = self.config.get("cfr", {})
        sqrt_max_raw = cfr_config.get("sqrt_max_raw", 1e4)
        blind_mode = cfr_config.get("blind_mode", False)
        infoset_mode = cfr_config.get("infoset_mode", "blind" if blind_mode else "simple")
        if blind_mode and infoset_mode != "blind":
            self.logger.warning(f"blind_mode=True，自动将 infoset_mode 从 {infoset_mode} 切换为 blind")
            infoset_mode = "blind"
        
        self.cfr_solver = create_mer_solver(
            warmup_iterations=self.warmup_iterations,
            perception_mode="deterministic",
            num_players=len(self.players),
            sqrt_max_raw=sqrt_max_raw,
            infoset_mode=infoset_mode
        )
        
        self.logger.info(f"信息集模式: {infoset_mode} | blind_mode={blind_mode}")
        
        # 【修改】Warmup 现在是"延迟记录"模式，不再瞬间完成
        # 前 warmup_iterations 个真实样本只更新遗憾值，不记录平均策略
        if self.warmup_iterations > 0:
            self.logger.info(f"CFR Warmup 已配置: 前 {self.warmup_iterations} 个样本仅更新遗憾值")
        else:
            self.logger.info(f"CFR 初始化完成 (无 Warmup), sqrt_max_raw={sqrt_max_raw}")
    
    def setup_label_extractor(
        self,
        use_llm: bool = None,
        model_name: str = None,
        device: str = None,
        temperature: float = None,
        max_tokens: int = None,
        use_vllm: bool = None
    ):
        """
        设置标签提取器
        
        使用独立的 LLM (Qwen2.5) 从 AffectGPT 输出中提取情感标签
        优先使用传入参数，否则从配置文件读取，最后使用默认值
        
        :param use_llm: 是否使用 LLM 提取（False 则使用规则匹配）
        :param model_name: LLM 模型名称
        :param device: GPU 设备（建议与 AffectGPT 使用不同 GPU）
        :param temperature: 生成温度
        :param max_tokens: 最大生成 token 数
        :param use_vllm: 是否使用 vLLM
        """
        # 从配置文件读取默认值
        extractor_cfg = self.config.get('label_extractor', {}) if hasattr(self, 'config') and self.config else {}
        
        # 优先级：传入参数 > 配置文件 > 默认值
        use_llm = use_llm if use_llm is not None else extractor_cfg.get('use_llm', True)
        model_name = model_name or extractor_cfg.get('model_name', 'Qwen25')
        device = device or extractor_cfg.get('device', 'cuda:2')
        temperature = temperature if temperature is not None else extractor_cfg.get('temperature', 0.3)
        max_tokens = max_tokens if max_tokens is not None else extractor_cfg.get('max_tokens', 128)
        use_vllm = use_vllm if use_vllm is not None else extractor_cfg.get('use_vllm', False)  # 默认使用 transformers
        
        self.logger.info(f"设置标签提取器: use_llm={use_llm}, model={model_name}, device={device}, use_vllm={use_vllm}")
        
        if use_llm:
            self.label_extractor = LabelExtractor(
                model_name=model_name,
                device=device,
                use_vllm=use_vllm,
                temperature=temperature,
                max_tokens=max_tokens
            )
            # 预加载模型
            self.label_extractor.initialize()
        else:
            raise ValueError(
                "必须使用 LLM 进行标签提取！请设置 use_llm=True。"
                "RuleBasedLabelExtractor 已被移除。"
            )
        
        self.logger.info("✓ LLM 标签提取器设置完成")
    
    def setup_driver(self, mode: str = "training"):
        """
        设置博弈驱动器
        
        :param mode: 运行模式 ("training" 或 "inference")
        """
        self.logger.info(f"创建博弈驱动器 (mode={mode})...")
        
        # 创建 Referee，必须使用 LLM 标签提取器
        if self.label_extractor is None:
            raise RuntimeError(
                "必须设置 LLM 标签提取器！请先调用 setup_label_extractor()。"
                "RuleBasedLabelExtractor 已被移除。"
            )
        
        referee = Referee(
            label_extractor=self.label_extractor.create_label_extractor_fn(),
            score_calculator=self.label_extractor.create_score_calculator_fn()
        )
        self.logger.info("  使用 LLM 标签提取器进行评分")
        
        cfr_cfg = self.config.get('cfr', {}) if hasattr(self, 'config') and self.config else {}
        blind_mode = cfr_cfg.get('blind_mode', False)

        self.driver = MERGameDriver(
            players=self.players,
            cfr_solver=self.cfr_solver,
            referee=referee,
            mode=mode,
            blind_mode=blind_mode
        )
        
        # 【新增】从配置文件读取并应用筹码/下注配置
        if hasattr(self, 'config') and self.config:
            cfr_cfg = self.config.get('cfr', {})
            betting_cfg = cfr_cfg.get('betting_amounts', {})
            if betting_cfg:
                self.driver.configure_betting(
                    ante=betting_cfg.get('ante'),
                    raise_amount=betting_cfg.get('raise_amount'),
                    initial_chips=betting_cfg.get('initial_chips')
                )
    
    def load_dataset(self, dataset_name: str = None, split: str = "train"):
        """
        加载数据集 - 使用 inference 风格的数据加载
        
        【重写】使用 MERDataLoader，与 inference_hybird.py 完全对齐
        
        :param dataset_name: 数据集名称（None 则从配置文件读取，默认 MERCaptionPlus）
        :param split: 数据集划分 ("train", "val", "test")
        """
        # 如果未指定数据集名称，从配置文件的 datasets 部分读取
        if dataset_name is None:
            if hasattr(self, 'config') and self.config:
                datasets_cfg = self.config.get('datasets', {})
                if datasets_cfg:
                    # 取第一个数据集的名称
                    dataset_name = list(datasets_cfg.keys())[0]
                    self.logger.info(f"从配置文件读取数据集: {dataset_name}")
            
            # 如果还是 None，使用默认值
            if dataset_name is None:
                dataset_name = "MERCaptionPlus"
                self.logger.info(f"使用默认数据集: {dataset_name}")
        
        self.logger.info(f"加载数据集: {dataset_name} (split={split})...")
        
        # 使用 MERDataLoader（与 inference_hybird.py 对齐）
        self.data_loader = MERDataLoader(self.config_path)
        self.data_loader.load_dataset(dataset_name, split)
        
        # 将样本列表复制到 self.dataset（保持兼容性）
        self.dataset = list(self.data_loader.samples)
        
        self.logger.info(f"✓ 通过 MERDataLoader 加载了 {len(self.dataset)} 个样本")
        self.logger.info(f"  face_or_frame: {self.data_loader.face_or_frame}")
        self.stats["total_samples"] = len(self.dataset)
    
    def create_prompt_functions(self):
        """
        创建 Prompt 生成函数
        
        【重要】所有博弈 Prompt 均从 prompt.py 获取，确保解耦性
        """
        # 直接使用 prompt.py 提供的工厂方法
        return create_cfr_prompt_functions()
    
    def train_epoch(self, epoch: int) -> Dict:
        """
        训练一个 epoch
        
        :param epoch: 当前 epoch 编号
        :return: epoch 统计信息
        """
        self.logger.info(f"\n{'='*60}")
        self.logger.info(f"Epoch {epoch + 1}/{self.num_epochs}")
        self.logger.info(f"{'='*60}")
        
        # 获取 Prompt 函数
        perception_fn, reevaluation_fn, decision_fn = self.create_prompt_functions()
        
        # 打乱数据集
        samples = self.dataset.copy()
        random.shuffle(samples)
        
        # 应用 max_samples 限制
        if self.max_samples is not None and self.max_samples < len(samples):
            samples = samples[:self.max_samples]
            self.logger.info(f"📌 使用前 {self.max_samples} 个样本（快速验证模式）")
        
        epoch_stats = {
            "epoch": epoch + 1,
            "num_samples": len(samples),
            "wins": {0: 0, 1: 0, 2: 0},
            "correct": 0,
            "total_reward": 0.0,
            "avg_trajectory_length": 0.0
        }
        
        trajectory_lengths = []
        epoch_start_time = time.time()  # 记录epoch开始时间，用于ETA计算
        
        for i, sample in enumerate(samples):
            sample_name = sample.get("name", f"sample_{i}")
            video_path = sample.get("video_path", "")
            audio_path = sample.get("audio_path")
            face_path = sample.get("face_path")
            subtitle = sample.get("subtitle", "")  # 获取字幕
            ground_truth = sample.get("label", "unknown")
            
            # 运行一局博弈
            try:
                trajectory = self.driver.play_round(
                    sample_name=sample_name,
                    ground_truth=ground_truth,
                    video_path=video_path,
                    audio_path=audio_path,
                    face_path=face_path,
                    subtitle=subtitle,  # 传递字幕
                    perception_prompt_fn=perception_fn,
                    reevaluation_prompt_fn=reevaluation_fn,
                    decision_prompt_fn=decision_fn
                )
                
                # 更新统计
                self.stats["total_games"] += 1
                trajectory_lengths.append(len(trajectory.decision_points))
                
                if trajectory.winner_id is not None:
                    epoch_stats["wins"][trajectory.winner_id] = \
                        epoch_stats["wins"].get(trajectory.winner_id, 0) + 1
                    self.stats["wins_by_player"][trajectory.winner_id] += 1
                    
                    # 检查预测是否正确
                    if trajectory.winner_prediction:
                        if ground_truth.lower() in trajectory.winner_prediction.lower():
                            epoch_stats["correct"] += 1
                            self.stats["correct_predictions"] += 1
                
                epoch_stats["total_reward"] += sum(trajectory.final_payoffs)
                
            except Exception as e:
                self.logger.error(f"样本 {sample_name} 处理失败: {e}")
                import traceback
                traceback.print_exc()
                continue
            
            # 进度日志
            if (i + 1) % 10 == 0:
                # 计算剩余时间
                elapsed = time.time() - epoch_start_time
                samples_done = i + 1
                samples_remaining = len(samples) - samples_done
                if samples_done > 0:
                    time_per_sample = elapsed / samples_done
                    eta_seconds = time_per_sample * samples_remaining
                    # 格式化剩余时间
                    if eta_seconds > 3600:
                        eta_str = f"{eta_seconds/3600:.1f}h"
                    elif eta_seconds > 60:
                        eta_str = f"{eta_seconds/60:.1f}min"
                    else:
                        eta_str = f"{eta_seconds:.0f}s"
                else:
                    eta_str = "--"
                
                # 获取 Warmup 状态
                warmup_status = ""
                if self.cfr_solver.is_in_warmup():
                    warmup_status = f" | 🔥 Warmup ({self.cfr_solver.current_sample_count}/{self.cfr_solver.warmup_iterations})"
                
                # 记录 ε-CCE 间隙（原始值，保持兼容）
                cce = self.cfr_solver.compute_cce_gap()
                self.regret_history.append({
                    "step": self.cfr_solver.current_sample_count,
                    "epsilon": cce["epsilon"],
                    "per_player": cce["per_player"]
                })

                # ---- 收集五维收敛指标 ----
                # 1. 归一化 ε-CCE
                norm_cce = self.cfr_solver.compute_cce_gap_normalized(self._payoff_range)
                # 2. 策略稳定性（平均 L1 距离）
                new_snap = self.cfr_solver.snapshot_average_strategy()
                stability = self._compute_strategy_stability(new_snap)
                # 3. Jensen-Shannon 散度（对平均策略的平滑变化衡量）
                jsd_info = self.cfr_solver.compute_strategy_jsd(self._prev_strategy_snapshot)
                self._prev_strategy_snapshot = new_snap
                # 4. 策略熵
                entropy_info = self.cfr_solver.compute_strategy_entropy()
                # 5. 惰性探索率（与上次记录相比新增信息集数 / 间隔步数）
                cur_infosets = len(self.cfr_solver.infoset_records)
                new_infosets = cur_infosets - self._prev_infoset_count
                self._prev_infoset_count = cur_infosets
                self.conv_metrics_history.append({
                    "step": self.cfr_solver.current_sample_count,
                    "raw_epsilon": {
                        "per_player": cce["per_player"],
                        "global": cce["epsilon"]
                    },
                    "norm_epsilon": {
                        "per_player": norm_cce["per_player"],
                        "global": norm_cce["epsilon"],
                        "payoff_range": self._payoff_range
                    },
                    "strategy_stability": stability,
                    "strategy_jsd": jsd_info,
                    "strategy_entropy": entropy_info,
                    "lazy_exploration": {
                        "total_infosets": cur_infosets,
                        "new_infosets": new_infosets,
                        "discovery_rate": new_infosets / 10
                    }
                })

                self.logger.info(
                    f"进度: {i+1}/{len(samples)} | "
                    f"CFR迭代: {self.cfr_solver.total_iterations} | "
                    f"信息集: {len(self.cfr_solver.infoset_records)}{warmup_status} | "
                    f"ε-CCE: {cce['epsilon']:.4f} (norm={norm_cce['epsilon']:.4f}) | "
                    f"稳定性 L1: {stability['global_max_mean']:.4f} (max={stability['max_l1_per_infoset']:.4f}) | "
                    f"JSD: {jsd_info['global_mean']:.4f} | "
                    f"ETA: {eta_str}"
                )
            
            # 定期保存检查点
            if (i + 1) % self.save_every == 0:
                self.save_checkpoint(f"epoch{epoch+1}_step{i+1}")
        
        # 计算 epoch 统计
        if trajectory_lengths:
            epoch_stats["avg_trajectory_length"] = np.mean(trajectory_lengths)
        
        self.stats["epoch_stats"].append(epoch_stats)
        
        # 打印 epoch 总结
        self.logger.info(f"\nEpoch {epoch + 1} 完成:")
        self.logger.info(f"  样本数: {epoch_stats['num_samples']}")
        self.logger.info(f"  获胜统计: {epoch_stats['wins']}")
        self.logger.info(f"  正确预测: {epoch_stats['correct']}")
        self.logger.info(f"  平均轨迹长度: {epoch_stats['avg_trajectory_length']:.2f}")
        self.logger.info(f"  总奖励: {epoch_stats['total_reward']:.2f}")
        
        return epoch_stats

    def _compute_strategy_stability(self, new_snapshot: Dict) -> Dict:
        """
        计算相邻两次记录之间的策略变化幅度（L1 距离）

        同时输出两种度量，含义不同，建议画图时都展示：

        【度量A：均值 L1（mean_l1）】
            per_player_mean[i] = (1/|S_i|) × Σ_{I∈S_i} ||σ̄^T(I) - σ̄^{T-Δ}(I)||_1
            - 对玩家 i 的所有信息集求 L1 距离的算术平均值
            - 优点：平均掉随机噪声，曲线平滑，视觉效果好
            - 缺点：若某个信息集剧烈波动，会被其他信息集稀释掉

            global_max_mean = max_i(per_player_mean[i])
            - 三个玩家的均值 L1 中取最大值，代表"最不稳定的玩家"
            - 适合作为单条收敛曲线画在论文里

        【度量B：严格最大值（max_l1_per_infoset）】
            max_l1_per_infoset = max_{I∈所有信息集} ||σ̄^T(I) - σ̄^{T-Δ}(I)||_1
            - 对应论文公式 Δ_T = max_I ||σ̄^T(I) - σ̄^{T-Δ}(I)||_1
            - 只要有「一个」信息集的策略仍在抖动，此值就不会降为 0
            - 优点：收敛标准最严格，审稿人无法挑剔
            - 缺点：对噪声敏感，曲线可能有毛刺

        【共同含义】
            两个指标收敛后都趋近 0，代表"策略不再变化"。
            若曲线下降并稳定在接近 0 的水平，即可声明策略已收敛到均衡。

        【附属字段】
            infosets_compared : 参与比较的信息集数量（与上一快照共同存在的信息集数）
                                 若该值持续增长，说明状态空间仍未完全探索

        :param new_snapshot: cfr_solver.snapshot_average_strategy() 返回的当前策略快照
                             格式：{infoset_key: {action_id: prob, ...}, ...}
        :return: 含以下字段的字典：
                 - per_player_mean     : {pid: 均值L1}，每位玩家的均值变化幅度
                 - global_max_mean     : 所有玩家均值L1的最大值（度量A，平滑）
                 - max_l1_per_infoset  : 所有信息集上L1的最大值（度量B，严格）
                 - infosets_compared   : 参与比较的信息集总数
        """
        n = self.cfr_solver.num_players
        if not self._prev_strategy_snapshot:
            # 第一次调用时无历史快照，所有距离初始化为 0
            return {
                "per_player_mean": {pid: 0.0 for pid in range(n)},
                "global_max_mean": 0.0,
                "max_l1_per_infoset": 0.0,
                "infosets_compared": 0
            }

        per_player_l1_sum: Dict[int, float] = {}   # 各玩家的 L1 累加值
        per_player_count: Dict[int, int] = {}       # 各玩家参与比较的信息集计数
        global_max_l1: float = 0.0                  # 跨所有信息集的全局最大 L1（度量B）

        for key, new_sigma in new_snapshot.items():
            if key not in self._prev_strategy_snapshot:
                # 新出现的信息集在上一快照中不存在，跳过（不计入比较）
                continue
            old_sigma = self._prev_strategy_snapshot[key]
            pid = self.cfr_solver.infoset_records[key].player_id
            all_actions = set(new_sigma) | set(old_sigma)
            # L1 距离：||σ_new(I) - σ_old(I)||_1 = Σ_a |σ_new(a) - σ_old(a)|
            l1 = sum(abs(new_sigma.get(a, 0.0) - old_sigma.get(a, 0.0)) for a in all_actions)
            per_player_l1_sum[pid] = per_player_l1_sum.get(pid, 0.0) + l1
            per_player_count[pid] = per_player_count.get(pid, 0) + 1
            if l1 > global_max_l1:
                global_max_l1 = l1  # 度量B：维护全局最大值

        # 度量A：各玩家均值 L1
        per_player_mean: Dict[int, float] = {
            pid: per_player_l1_sum.get(pid, 0.0) / max(per_player_count.get(pid, 1), 1)
            for pid in range(n)
        }

        return {
            "per_player_mean": per_player_mean,            # 每位玩家的均值 L1（度量A，平滑）
            "global_max_mean": max(per_player_mean.values()) if per_player_mean else 0.0,  # 度量A全局汇总
            "max_l1_per_infoset": global_max_l1,           # 严格 max_I L1（度量B，论文公式）
            "infosets_compared": sum(per_player_count.values())  # 参与比较的信息集总数
        }

    def save_checkpoint(self, name: str):
        """
        保存检查点

        :param name: 检查点名称
        """
        # 检查 CFR 求解器是否已初始化
        if self.cfr_solver is None:
            self.logger.warning(f"CFR 求解器未初始化，跳过保存检查点: {name}")
            return
        
        checkpoint_path = os.path.join(self.checkpoint_dir, f"cfr_{name}.json")
        self.cfr_solver.save_strategy(checkpoint_path)
        
        # 保存统计信息（含 ε-CCE）
        cce = self.cfr_solver.compute_cce_gap()
        self.stats["cce_gap"] = cce
        stats_path = os.path.join(self.checkpoint_dir, f"stats_{name}.json")
        with open(stats_path, 'w', encoding='utf-8') as f:
            json.dump(self.stats, f, indent=2, ensure_ascii=False)
        
        # 保存遗憾曲线数据（原始 ε-CCE，保持向后兼容）
        regret_path = os.path.join(self.checkpoint_dir, f"regret_history_{name}.json")
        with open(regret_path, 'w', encoding='utf-8') as f:
            json.dump(self.regret_history, f, indent=2, ensure_ascii=False)

        # 保存四维收敛指标（raw ε / norm ε / 策略稳定性 / 策略熵 / 惰性探索率）
        conv_metrics_path = os.path.join(self.checkpoint_dir, f"conv_metrics_{name}.json")
        with open(conv_metrics_path, 'w', encoding='utf-8') as f:
            json.dump(self.conv_metrics_history, f, indent=2, ensure_ascii=False)

        self.logger.info(
            f"检查点已保存: {checkpoint_path} | "
            f"ε-CCE={cce['epsilon']:.4f} (T={cce['T']})"
        )
    
    def train(
        self,
        model_configs: List[Dict],
        dataset_path: str = None,
        extractor_device: str = None,
        extractor_model: str = None,
        use_rule_extractor: bool = None
    ):
        """
        完整训练流程
        
        :param model_configs: 模型配置
        :param dataset_path: 数据集路径（可选，如果为空则从配置文件加载）
        :param extractor_device: 标签提取器使用的 GPU 设备（None 则从配置文件读取）
        :param extractor_model: 标签提取器使用的模型名称（None 则从配置文件读取）
        :param use_rule_extractor: 是否使用规则匹配提取器（None 则从配置文件读取）
        """
        start_time = time.time()
        
        try:
            # 1. 加载配置
            self.load_config()
            
            # 2. 加载数据集（需要先加载，以便后续创建 LLM callable 时使用）
            self.load_dataset(dataset_path)
            
            if len(self.dataset) == 0:
                self.logger.error("数据集为空，无法训练")
                return
            
            # 3. 加载模型
            self.load_models(model_configs)
            
            # 4. 设置玩家（需要在 load_dataset 之后，因为 llm_callable 需要 torch_datasets）
            self.setup_players(model_configs)
            
            # 5. 初始化 CFR
            self.setup_cfr_solver()
            
            # 6. 设置标签提取器（用于从 LLM 输出中提取情感标签）
            # 优先使用命令行参数，否则从配置文件读取
            self.setup_label_extractor(
                use_llm=None if use_rule_extractor is None else not use_rule_extractor,
                model_name=extractor_model,
                device=extractor_device
            )
            
            # 7. 设置驱动器
            self.setup_driver(mode="training")
            
            # 8. 训练循环
            self.logger.info("\n" + "=" * 60)
            self.logger.info("开始 CFR 训练")
            self.logger.info("=" * 60)
            
            for epoch in range(self.num_epochs):
                self.train_epoch(epoch)
                
                # 每个 epoch 结束保存
                self.save_checkpoint(f"epoch{epoch+1}_final")
            
            # 8. 保存最终模型
            self.save_checkpoint("final")
            
            # 训练完成
            elapsed = time.time() - start_time
            cce = self.cfr_solver.compute_cce_gap()
            self.logger.info("\n" + "=" * 60)
            self.logger.info("训练完成!")
            self.logger.info(f"总耗时: {elapsed/3600:.2f} 小时")
            self.logger.info(f"总博弈局数: {self.stats['total_games']}")
            self.logger.info(f"CFR 迭代次数: {self.cfr_solver.total_iterations}")
            self.logger.info(f"信息集数量: {len(self.cfr_solver.infoset_records)}")
            self.logger.info(f"最终 ε-CCE: {cce['epsilon']:.4f} (per_player={cce['per_player']})")
            self.logger.info("=" * 60)
            
        except KeyboardInterrupt:
            self.logger.warning("\n训练被中断，保存当前状态...")
            self.save_checkpoint("interrupted")
        
        except Exception as e:
            self.logger.error(f"训练失败: {e}")
            import traceback
            traceback.print_exc()
            self.save_checkpoint("error")
            raise


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="MER 博弈 CFR 训练")
    
    parser.add_argument(
        "--config", 
        type=str, 
        default="train_configs/AffectGame.yaml",
        help="配置文件路径（包含模型和数据集配置）"
    )
    
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="数据集路径（可选，默认从配置文件中读取）"
    )
    
    parser.add_argument(
        "--output_dir",
        type=str,
        default="output/cfr_training",
        help="输出目录"
    )
    
    parser.add_argument(
        "--num_epochs",
        type=int,
        default=10,
        help="训练轮数"
    )
    
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="每个 epoch 最多使用的样本数（用于快速验证，如 --max_samples 2000）"
    )
    
    parser.add_argument(
        "--warmup_iterations",
        type=int,
        default=1000,
        help="CFR 热身迭代次数"
    )
    
    parser.add_argument(
        "--save_every",
        type=int,
        default=100,
        help="每隔多少样本保存检查点"
    )
    
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="随机种子"
    )
    
    parser.add_argument(
        "--log_level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="日志级别"
    )
    
    # 模型配置（可以通过命令行覆盖）
    parser.add_argument(
        "--model_ckpts",
        type=str,
        nargs="+",
        default=None,
        help="模型检查点路径列表（按玩家顺序）"
    )
    
    parser.add_argument(
        "--device_ids",
        type=int,
        nargs="+",
        default=[0, 1, 2],
        help="GPU 设备 ID 列表"
    )
    
    parser.add_argument(
        "--llm_types",
        type=str,
        nargs="+",
        default=None,
        help="LLM 类型列表（Qwen25, Llama31, Gemma3 等），按玩家顺序"
    )
    
    parser.add_argument(
        "--use_config_players",
        action="store_true",
        help="从配置文件的 cfr.players 读取模型配置"
    )
    
    # 标签提取器配置
    parser.add_argument(
        "--extractor_device",
        type=str,
        default="cuda:1",
        help="标签提取器使用的 GPU 设备（建议与 AffectGPT 使用不同 GPU）"
    )
    
    parser.add_argument(
        "--extractor_model",
        type=str,
        default="Qwen25",
        help="标签提取器使用的模型名称（对应 config.py 中的 PATH_TO_LLM）"
    )
    
    parser.add_argument(
        "--use_rule_extractor",
        action="store_true",
        help="使用规则匹配提取器（不需要额外 LLM，但准确率较低）"
    )
    
    return parser.parse_args()


def main():
    """主函数"""
    args = parse_args()

    # 先初始化日志，尽早捕获所有 stdout/stderr
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(args.output_dir, f"cfr_{timestamp}")
    log_dir = os.path.join(run_dir, "logs")
    logger = setup_logging(log_dir, args.log_level, redirect_stdio=True)

    print("=" * 60)
    print("MER 博弈 CFR 训练")
    print("=" * 60)
    print(f"配置文件: {args.config}")
    print(f"数据集: {args.dataset if args.dataset else '从配置文件读取'}")
    print(f"输出目录: {args.output_dir}")
    print(f"训练轮数: {args.num_epochs}")
    if args.max_samples:
        print(f"最大样本数: {args.max_samples} (快速验证模式)")
    print("-" * 60)
    print("标签提取器配置:")
    if args.use_rule_extractor:
        print("  模式: 规则匹配（无需额外 GPU，命令行指定）")
    elif args.extractor_device != "cuda:1" or args.extractor_model != "Qwen25":
        print(f"  模式: LLM 提取（命令行指定）")
        print(f"  模型: {args.extractor_model}")
        print(f"  设备: {args.extractor_device}")
    else:
        print("  模式: 从配置文件读取（label_extractor 部分）")
    print("=" * 60)
    
    # 构建模型配置
    model_configs = None
    
    # 方式 1: 从命令行参数读取
    if args.model_ckpts:
        model_configs = []
        for i, ckpt in enumerate(args.model_ckpts):
            device_id = args.device_ids[i] if i < len(args.device_ids) else 0
            llm_type = args.llm_types[i] if args.llm_types and i < len(args.llm_types) else "Qwen25"
            model_configs.append({
                "name": f"AffectGPT_Player{i}",
                "ckpt_path": ckpt,
                "device_id": device_id,
                "llm_type": llm_type
            })
        print(f"\n✓ 从命令行加载 {len(model_configs)} 个模型配置")
    
    # 方式 2: 从配置文件读取
    elif args.use_config_players:
        from omegaconf import OmegaConf
        
        # 解析配置文件路径
        config_path = args.config
        if not os.path.isabs(config_path):
            for p in [config_path, 
                      os.path.join(SCRIPT_DIR, config_path), 
                      os.path.join(PROJECT_ROOT, config_path)]:
                if os.path.exists(p):
                    config_path = p
                    break
        
        cfg = OmegaConf.load(config_path)
        
        # 从配置文件读取 CFR 相关参数
        if hasattr(cfg, 'cfr'):
            cfr_cfg = cfg.cfr
            # 如果命令行没有显式指定，则使用配置文件中的值
            if args.warmup_iterations == 1000:  # 默认值
                args.warmup_iterations = cfr_cfg.get('warmup_iterations', 1000)
            if args.save_every == 100:  # 默认值
                args.save_every = cfr_cfg.get('save_every', 100)
            if args.output_dir == "output/cfr_training":  # 默认值
                args.output_dir = cfr_cfg.get('output_dir', 'output/cfr_training')
            # 从配置文件读取 max_samples（命令行优先）
            if args.max_samples is None:
                config_max_samples = cfr_cfg.get('max_samples', None)
                if config_max_samples is not None:
                    args.max_samples = config_max_samples
                    print(f"\n✓ 从配置文件读取 max_samples: {args.max_samples}")
            print(f"\n✓ 从配置文件读取 CFR 参数:")
            print(f"  warmup_iterations: {args.warmup_iterations}")
            print(f"  save_every: {args.save_every}")
            print(f"  output_dir: {args.output_dir}")
            if args.max_samples:
                print(f"  max_samples: {args.max_samples}")
        
        # 从 run 部分读取训练轮数
        if hasattr(cfg, 'run') and args.num_epochs == 10:  # 默认值
            args.num_epochs = cfg.run.get('max_epoch', 10)
            print(f"  num_epochs: {args.num_epochs}")
        
        if hasattr(cfg, 'cfr') and hasattr(cfg.cfr, 'players'):
            model_configs = []
            for player_cfg in cfg.cfr.players:
                model_configs.append({
                    "name": player_cfg.get('name', f"Player_{len(model_configs)}"),
                    "ckpt_path": player_cfg.get('ckpt_path', ''),
                    "device_id": player_cfg.get('device_id', 0),
                    "llm_type": player_cfg.get('llm_type', 'Qwen25')
                })
            print(f"\n✓ 从配置文件加载 {len(model_configs)} 个模型配置:")
            for mc in model_configs:
                print(f"  - {mc['name']}: {mc['ckpt_path']} (GPU {mc['device_id']}, {mc['llm_type']})")
        else:
            print("\n⚠️ 配置文件中未找到 cfr.players")
    
    # 检查模型配置是否有效
    if model_configs is None:
        print("\n❌ 错误：未指定模型检查点！")
        print("\n请使用以下方式之一加载模型:")
        print("\n方式 1: 命令行参数")
        print("  python train_cfr.py \\")
        print("      --config train_configs/cfr_training.yaml \\")
        print("      --model_ckpts /path/to/model1.pth /path/to/model2.pth /path/to/model3.pth \\")
        print("      --llm_types Qwen25 Llama31 Gemma3 \\")
        print("      --device_ids 0 1 2")
        print("\n方式 2: 配置文件")
        print("  1. 编辑 cfr_training.yaml 中的 cfr.players 和 datasets 部分")
        print("  2. python train_cfr.py --config train_configs/cfr_training.yaml --use_config_players")
        print("\n注意：数据集会自动从配置文件的 datasets 部分读取（与 train.py 相同）")
        print("")
        sys.exit(1)
    
    # 创建训练器
    trainer = CFRTrainer(
        config_path=args.config,
        output_dir=args.output_dir,
        num_epochs=args.num_epochs,
        warmup_iterations=args.warmup_iterations,
        save_every=args.save_every,
        log_level=args.log_level,
        seed=args.seed,
        max_samples=args.max_samples,
        run_dir=run_dir,
        log_dir=log_dir,
        logger=logger
    )
    
    # 开始训练
    # 命令行参数只在显式指定时才覆盖配置文件，否则传 None
    trainer.train(
        model_configs=model_configs,
        dataset_path=args.dataset,
        extractor_device=args.extractor_device if args.extractor_device != "cuda:1" else None,
        extractor_model=args.extractor_model if args.extractor_model != "Qwen25" else None,
        use_rule_extractor=args.use_rule_extractor if args.use_rule_extractor else None
    )


if __name__ == "__main__":
    main()
