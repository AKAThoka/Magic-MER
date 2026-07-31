"""
inference_cfr.py - MER 博弈 CFR 推理脚本

功能：
1. 加载预训练的 CFR 策略表
2. 在测试集上进行冻结策略博弈
3. 使用 EV 判定胜者（无需 GT）
4. 输出 evaluation.py 兼容的格式 (checkpoint_*.npz)

使用方式：
    python inference_cfr.py \\
        --config train_configs/AffectGame.yaml \\
        --strategy path/to/cfr_strategy.json \\
        --output output/results-{dataset}/cfr_inference
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
from OpenSpielGame.DataLoader import MERDataLoader
from OpenSpielGame.LLMInference import LLMInferenceWrapper, create_llm_wrapper
from OpenSpielGame.prompt import (
    create_cfr_prompt_functions,
    get_perception_prompt,
    get_reevaluation_prompt,
    get_decision_prompt
)
from OpenSpielGame.LabelExtractor import (
    LabelExtractor,
    OVLabelExtractor,
)


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
    log_file = os.path.join(log_dir, f"inference_cfr_{timestamp}.log")

    logger = logging.getLogger("CFRInference")
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

        # 同时配置子模块的日志
        for module_name in ["MERGame", "MERSolver", "ConfidenceUtils"]:
            module_logger = logging.getLogger(module_name)
            module_logger.setLevel(logging.INFO)
            module_logger.addHandler(file_handler)
            module_logger.addHandler(console_handler)

        logger.info(f"日志文件: {log_file}")

    if redirect_stdio:
        sys.stdout = _StreamToLogger(logger, logging.INFO)
        sys.stderr = _StreamToLogger(logger, logging.ERROR)

    return logger


class CFRInferenceRunner:
    """
    CFR 推理运行器
    
    复用 CFRTrainer 的核心逻辑，但运行在 inference 模式：
    1. 加载预训练策略（不更新）
    2. 使用 EV 判定胜者
    3. 输出 evaluation.py 兼容格式
    """
    
    def __init__(
        self,
        config_path: str,
        strategy_path: str,
        output_dir: str = "output/cfr_inference",
        log_level: str = "INFO",
        seed: int = 42,
        max_samples: int = None,
        log_dir: Optional[str] = None,
        logger: Optional[logging.Logger] = None
    ):
        """
        初始化推理运行器
        
        :param config_path: 配置文件路径 (yaml)
        :param strategy_path: 预训练策略 JSON 文件路径
        :param output_dir: 输出目录
        :param log_level: 日志级别
        :param seed: 随机种子
        :param max_samples: 最多处理的样本数（None 表示全部）
        """
        self.config_path = config_path
        self.strategy_path = strategy_path
        self.output_dir = output_dir
        self.max_samples = max_samples
        self.log_level = log_level  # 保存日志级别以便后续使用
        self.log_dir = log_dir
        self.logger = logger
        
        # 设置随机种子（完全可复现）
        self.seed = seed
        self._set_seed_for_reproducibility(seed)
    
    def _set_seed_for_reproducibility(self, seed: int):
        """
        设置所有随机种子以确保完全可复现性
        
        包含:
        - Python random
        - NumPy random
        - PyTorch CPU/GPU
        - cuDNN 确定性设置
        - PYTHONHASHSEED
        - transformers (如果可用)
        """
        import os
        import random
        import numpy as np
        import torch
        
        # 1. 基础随机种子
        random.seed(seed)
        np.random.seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)
        
        # 2. PyTorch 种子
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
        
        # 3. cuDNN 确定性设置
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        
        # 4. transformers 种子（如果可用）
        try:
            import transformers
            transformers.set_seed(seed)
        except ImportError:
            pass
        
        # 5. 设置环境变量以确保 CUDA 操作确定性
        os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
        
        # 6. PyTorch 2.0+ 确定性算法
        if hasattr(torch, 'use_deterministic_algorithms'):
            try:
                torch.use_deterministic_algorithms(True)
            except Exception:
                # 某些操作可能不支持确定性模式
                pass
        
        print(f" 随机种子已设置为 {seed}，启用完全可复现模式")

        # 创建输出目录
        os.makedirs(self.output_dir, exist_ok=True)
        if self.log_dir is None:
            self.log_dir = os.path.join(self.output_dir, "logs")
        os.makedirs(self.log_dir, exist_ok=True)

        # 设置日志（允许外部注入）
        if self.logger is None:
            self.logger = setup_logging(self.log_dir, self.log_level, redirect_stdio=True)
        
        # 运行状态
        self.config = None
        self.models: Dict[str, Any] = {}
        self.players: List[MERPlayer] = []
        self.cfr_solver: Optional[MERCFRSolver] = None
        self.driver: Optional[MERGameDriver] = None
        self.data_loader: Optional[MERDataLoader] = None
        self.llm_wrappers: Dict[str, LLMInferenceWrapper] = {}
        self.label_extractor: Optional[LabelExtractor] = None
        
        # 推理结果存储
        self.predictions: Dict[str, str] = {}  # sample_name -> winner_prediction
        
        # 获胜统计（用于打印汇总）
        self.win_counts: Dict[int, int] = {}  # player_id -> win_count
        self.total_games: int = 0
        
        self.logger.info("=" * 60)
        self.logger.info("CFR 推理运行器初始化")
        self.logger.info(f"  配置文件: {self.config_path}")
        self.logger.info(f"  策略文件: {self.strategy_path}")
        self.logger.info(f"  输出目录: {self.output_dir}")
        self.logger.info("=" * 60)
    
    def load_config(self):
        """加载配置文件"""
        from omegaconf import OmegaConf
        
        self.logger.info(f"加载配置: {self.config_path}")
        
        # 解析配置文件路径
        config_path = self.config_path
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
            else:
                raise FileNotFoundError(f"配置文件未找到: {self.config_path}")
        
        cfg = OmegaConf.load(config_path)
        self.config = OmegaConf.to_container(cfg, resolve=True)
        self.logger.info("✓ 配置加载完成")
    
    def load_strategy(self):
        """加载预训练的 CFR 策略"""
        self.logger.info(f"加载 CFR 策略: {self.strategy_path}")
        
        if not os.path.exists(self.strategy_path):
            raise FileNotFoundError(f"策略文件不存在: {self.strategy_path}")
        
        # 从配置读取 infoset_mode（load_strategy 会根据策略文件自动覆盖）
        infoset_mode = self.config.get('cfr', {}).get('infoset_mode', 'simple')
        
        # 创建空的 CFR 求解器
        self.cfr_solver = create_mer_solver(
            warmup_iterations=0,  # 推理模式不需要热身
            perception_mode="deterministic",
            infoset_mode=infoset_mode
        )
        
        # 加载策略（会自动检测并切换 infoset_mode）
        self.cfr_solver.load_strategy(self.strategy_path)
        
        self.logger.info(f"✓ 策略加载完成: {len(self.cfr_solver.infoset_records)} 个信息集, 模式: {self.cfr_solver.infoset_mode}")
    
    def _parse_inactive_players(self, config_section: str = 'cfr_inference') -> List[int]:
        """解析消融实验配置，返回非活跃玩家索引列表"""
        cfg = self.config.get(config_section, {})
        inactive_player = cfg.get('inactive_player', None)
        if inactive_player is None:
            return []
        if isinstance(inactive_player, list):
            return inactive_player
        return [inactive_player]

    def load_models(self, model_configs: List[Dict], skip_players: List[int] = None):
        """加载 LLM 模型（复用 AffectGPTLoader 的逻辑）"""
        self.logger.info("=" * 40)
        self.logger.info("开始加载模型...")
        self.logger.info("=" * 40)
        
        # 过滤掉非活跃玩家的配置
        if skip_players:
            active_configs = [
                cfg for i, cfg in enumerate(model_configs)
                if i not in skip_players
            ]
            skipped_names = [model_configs[i].get('name', f'Player_{i}') for i in skip_players if i < len(model_configs)]
            self.logger.info(f"⚡ 消融模式：跳过 {skipped_names} 的模型加载")
        else:
            active_configs = model_configs
        
        # 创建模型加载器
        loader = AffectGPTLoader(self.config_path)
        
        # 只加载活跃玩家的模型
        loader.load_models(active_configs)
        
        # 保存模型引用（models 字典中的值是 Chat 实例）
        self.models = loader.models
        self.loader = loader
        
        self.logger.info(f"成功加载 {len(self.models)} 个模型")
        for name in self.models.keys():
            self.logger.info(f"  - {name}")
        
        self.logger.info(f"✓ 模型加载完成: {list(self.models.keys())}")
    
    def load_dataset(self, dataset_name: str = None, split: str = "test"):
        """
        加载数据集
        
        :param dataset_name: 数据集名称
        :param split: 数据集划分（推理默认使用 test）
        """
        # 优先从 cfr_inference 读取数据集配置
        if dataset_name is None:
            cfr_inf_cfg = self.config.get('cfr_inference', {})
            dataset_name = cfr_inf_cfg.get('dataset', None)
            
            if dataset_name:
                self.logger.info(f"从 cfr_inference.dataset 读取数据集: {dataset_name}")
        
        # 其次从 datasets 配置读取
        if dataset_name is None:
            if self.config:
                datasets_cfg = self.config.get('datasets', {})
                if datasets_cfg:
                    dataset_name = list(datasets_cfg.keys())[0]
                    self.logger.info(f"从 datasets 配置读取数据集: {dataset_name}")
        
        # 默认数据集
        if dataset_name is None:
            dataset_name = "MERCaptionPlus"
            self.logger.info(f"使用默认数据集: {dataset_name}")
        
        self.logger.info(f"加载数据集: {dataset_name} (split={split})...")
        
        self.data_loader = MERDataLoader(self.config_path)
        self.data_loader.load_dataset(dataset_name, split)
        
        self.logger.info(f"✓ 加载了 {len(self.data_loader.samples)} 个样本")
    
    def setup_players(self, model_configs: List[Dict], skip_players: List[int] = None):
        """设置玩家（复用 train_cfr.py 的逻辑）"""
        self.logger.info("设置玩家...")
        self.players = []
        
        for i, config in enumerate(model_configs):
            model_name = config.get("name", f"Player_{i}")
            llm_type = config.get("llm_type", "Qwen25")
            
            # 非活跃玩家：创建 dummy，不需要真实模型
            if skip_players and i in skip_players:
                def _dummy_llm(*args, **kwargs):
                    return ("INACTIVE", np.zeros(4))
                player = MERPlayer(
                    player_id=i,
                    name=model_name,
                    llm_callable=_dummy_llm
                )
                self.players.append(player)
                self.logger.info(f"  玩家 {i}: {model_name} (⚡ 非活跃，未加载模型)")
                continue
            
            # self.models[model_name] 直接是 Chat 实例
            chat_instance = self.models.get(model_name)
            
            if chat_instance is not None:
                # 创建 llm_callable（完全复用 train_cfr.py 的实现）
                llm_callable = self._create_llm_callable(
                    chat_instance=chat_instance,
                    player_name=model_name,
                    llm_type=llm_type
                )
            else:
                self.logger.warning(f"  模型 {model_name} 未找到，使用 fallback")
                def fallback_llm(user_message, video_path=None, audio_path=None, 
                                face_path=None, sample_name=None, subtitle=None):
                    return (f"[FALLBACK] {user_message[:50]}...", np.random.randn(4))
                llm_callable = fallback_llm
            
            player = MERPlayer(
                player_id=i,
                name=model_name,
                llm_callable=llm_callable
            )
            self.players.append(player)
            self.logger.info(f"  玩家 {i}: {model_name}")
        
        self.logger.info(f"✓ 创建了 {len(self.players)} 个玩家")
    
    def _create_llm_callable(
        self,
        chat_instance,
        player_name: str,
        llm_type: str = "Qwen25"
    ):
        """
        创建 LLM 可调用函数（完全复用 train_cfr.py 的阶段检测逻辑）
        
        这是推理流程的核心：
        - 感知阶段：使用多模态输入
        - 决策/重评估阶段：纯文本输入
        """
        from OpenSpielGame.LLMInference import LLMInferenceWrapper
        
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
        
        # 非感知阶段的关键词
        non_perception_keywords = [
            "Texas Hold'em", "Strategy Advice", "Your Decision",
            "FOLD", "CHECK", "CALL", "RAISE", "Nash Equilibrium",
            "re-evaluate", "Re-evaluate", "other players", "Other players",
            "revealed", "predictions", "confidence"
        ]
        
        def llm_callable(
            user_message: str,
            video_path: Optional[str] = None,
            audio_path: Optional[str] = None,
            face_path: Optional[str] = None,
            sample_name: Optional[str] = None,
            subtitle: Optional[str] = None
        ) -> Tuple[str, np.ndarray]:
            """
            调用 LLM 模型（与 train_cfr.py 完全一致的逻辑）
            """
            try:
                # 检测当前阶段
                is_perception_phase = not any(
                    keyword in user_message for keyword in non_perception_keywords
                )
                
                if is_perception_phase:
                    # 感知阶段：需要多模态数据
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
                    # 非感知阶段：纯文本输入
                    sample_data = {
                        'frame': None, 'raw_frame': None,
                        'face': None, 'raw_face': None,
                        'audio': None, 'raw_audio': None,
                        'image': None, 'raw_image': None,
                    }
                    
                    # 只添加 LLM 格式前后缀
                    prompt = f"{llm_prefix}{user_message}{llm_suffix}"
                
                # 执行推理
                response, logits = wrapper.inference(sample_data, prompt)
                
                if logits is None:
                    logits = np.random.randn(4) * 0.5 + 1.0
                
                return (response, logits)
                
            except Exception as e:
                self.logger.error(f"[{player_name}] LLM 调用失败: {e}")
                import traceback
                traceback.print_exc()
                return ("neutral", np.array([0.25, 0.25, 0.25, 0.25]))
        
        return llm_callable

    def _resolve_label_extractor_device(self, preferred_device: Optional[str] = None) -> str:
        """解析标签提取器设备，自动回退到可用设备。"""
        preferred = preferred_device or "cuda:2"

        if preferred == "cpu":
            return "cpu"

        if not torch.cuda.is_available():
            self.logger.warning("CUDA 不可用，标签提取器回退到 CPU")
            return "cpu"

        gpu_count = torch.cuda.device_count()

        if preferred.startswith("cuda:"):
            try:
                gpu_id = int(preferred.split(":", 1)[1])
                if 0 <= gpu_id < gpu_count:
                    return preferred
                self.logger.warning(
                    f"标签提取器设备 {preferred} 不可用（可用 GPU 数: {gpu_count}），回退到 cuda:0"
                )
                return "cuda:0"
            except Exception:
                self.logger.warning(f"标签提取器设备格式无效: {preferred}，回退到 cuda:0")
                return "cuda:0"

        # 其他写法统一回退到第一张卡
        return "cuda:0"

    def _as_yaml_bool(self, value: Any, default: bool = False, field_name: str = "") -> bool:
        """将 YAML 字段稳健解析为布尔值。"""
        if isinstance(value, bool):
            return value
        if value is None:
            return default
        if isinstance(value, (int, float)):
            return bool(value)
        if isinstance(value, str):
            v = value.strip().lower()
            if v in {"true", "1", "yes", "y", "on"}:
                return True
            if v in {"false", "0", "no", "n", "off", "flase"}:  # 兼容常见拼写错误
                if v == "flase":
                    self.logger.warning(f"配置项 {field_name or '<unknown>'} 写成了 'flase'，按 false 处理")
                return False
        self.logger.warning(f"配置项 {field_name or '<unknown>'}={value!r} 无法解析为布尔值，回退默认值 {default}")
        return default
    
    def setup_label_extractor(self):
        """按 YAML 配置设置标签提取器（YAML 为准）。"""
        cfr_inference_cfg = self.config.get('cfr_inference', {}) if self.config else {}
        label_cfg = self.config.get('label_extractor', {}) if self.config else {}

        # YAML 主开关（推荐）
        skip_scoring_model = self._as_yaml_bool(
            cfr_inference_cfg.get('skip_scoring_model', False),
            default=False,
            field_name='cfr_inference.skip_scoring_model'
        )
        use_llm = self._as_yaml_bool(
            label_cfg.get('use_llm', True),
            default=True,
            field_name='label_extractor.use_llm'
        )
        skip_score_calculation = self._as_yaml_bool(
            cfr_inference_cfg.get('skip_score_calculation', False),
            default=False,
            field_name='cfr_inference.skip_score_calculation'
        )

        # 兼容旧字段（不作为主路径）
        legacy_skip_label_extractor = self._as_yaml_bool(
            cfr_inference_cfg.get('skip_label_extractor', False),
            default=False,
            field_name='cfr_inference.skip_label_extractor'
        )

        # 推理态允许完全跳过标签提取大模型（例如仅做 winner 输出时）
        if skip_scoring_model or (not use_llm) or skip_score_calculation or legacy_skip_label_extractor:
            reasons = []
            if skip_scoring_model:
                reasons.append("cfr_inference.skip_scoring_model=True")
            if not use_llm:
                reasons.append("label_extractor.use_llm=False")
            if skip_score_calculation:
                reasons.append("cfr_inference.skip_score_calculation=True")
            if legacy_skip_label_extractor:
                reasons.append("cfr_inference.skip_label_extractor=True(legacy)")
            self.label_extractor = None
            self.logger.info(f"✓ 跳过 LLM 标签提取模型加载（{', '.join(reasons)}）")
            return

        requested_device = label_cfg.get('device', cfr_inference_cfg.get('label_extractor_device', 'cuda:0'))
        resolved_device = self._resolve_label_extractor_device(requested_device)

        model_name = label_cfg.get('model_name', 'Qwen25')
        use_vllm = self._as_yaml_bool(
            label_cfg.get('use_vllm', False),
            default=False,
            field_name='label_extractor.use_vllm'
        )
        try:
            temperature = float(label_cfg.get('temperature', 0.0))
        except Exception:
            temperature = 0.0
            self.logger.warning("label_extractor.temperature 非法，回退为 0.0")
        try:
            max_tokens = int(label_cfg.get('max_tokens', 512))
        except Exception:
            max_tokens = 512
            self.logger.warning("label_extractor.max_tokens 非法，回退为 512")

        try:
            self.label_extractor = OVLabelExtractor(
                model_name=model_name,
                device=resolved_device,
                use_vllm=use_vllm,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            self.label_extractor.initialize()
            self.logger.info(
                f"✓ 使用 LLM 标签提取器 (OVLabelExtractor), model={model_name}, device={resolved_device}, use_vllm={use_vllm}"
            )
        except Exception as e:
            # 不阻塞推理：回退到 Referee 默认 extractor/scorer
            self.label_extractor = None
            self.logger.warning(f"⚠️ LLM 标签提取模型加载失败 ({e})，回退到默认提取器")
    
    def setup_driver(self):
        """设置博弈驱动器（inference 模式）"""
        self.logger.info("创建博弈驱动器 (mode=inference)...")

        # 若未加载 LLM 标签提取器，则使用 Referee 默认 extractor/scorer
        if self.label_extractor is not None:
            referee = Referee(
                label_extractor=self.label_extractor.create_label_extractor_fn(),
                score_calculator=self.label_extractor.create_score_calculator_fn()
            )
        else:
            referee = Referee()
            self.logger.info("✓ 使用 Referee 默认提取器/评分器（未加载 LLM 标签提取模型）")
        
        self.driver = MERGameDriver(
            players=self.players,
            cfr_solver=self.cfr_solver,
            referee=referee,
            mode="inference",  # 冻结模式
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
            
            # 【新增】读取推理态是否跳过评分计算的配置
            cfr_inference_cfg = self.config.get('cfr_inference', {})
            skip_score = cfr_inference_cfg.get('skip_score_calculation', False)
            self.driver.skip_score_calculation = skip_score
            if skip_score:
                self.logger.info("✓ 推理态已启用「跳过评分计算」模式（适用于整数标签数据集如 MELD）")
        
        self.logger.info("✓ 博弈驱动器创建完成 (推理模式，策略冻结)")
    
    def run_inference(self) -> Dict[str, str]:
        """
        执行推理
        
        :return: 预测结果字典 {sample_name: prediction}
        """
        self.logger.info("\n" + "=" * 60)
        self.logger.info("开始 CFR 推理")
        self.logger.info("=" * 60)
        
        samples = list(self.data_loader.samples)
        if self.max_samples is not None:
            samples = samples[:self.max_samples]
        
        total = len(samples)
        self.logger.info(f"处理 {total} 个样本...")
        
        # create_cfr_prompt_functions 返回元组 (perception_fn, reevaluation_fn, decision_fn)
        perception_fn, reevaluation_fn, decision_fn = create_cfr_prompt_functions()
        
        for idx, sample in enumerate(samples):
            sample_name = sample.get("name", f"sample_{idx}")
            subtitle = sample.get("subtitle", "")
            gt = sample.get("label", "")  # 推理时可能为空
            
            self.logger.info(f"\n[{idx+1}/{total}] 样本: {sample_name}")
            
            try:
                # 获取样本的多模态文件路径
                video_path, audio_path, face_path, _ = self.data_loader.get_sample_paths(sample_name)
                
                # 运行博弈（使用 play_round 方法）
                trajectory = self.driver.play_round(
                    sample_name=sample_name,
                    ground_truth=gt,
                    video_path=video_path or "",
                    audio_path=audio_path,
                    face_path=face_path,
                    subtitle=subtitle,
                    perception_prompt_fn=perception_fn,
                    reevaluation_prompt_fn=reevaluation_fn,
                    decision_prompt_fn=decision_fn
                )
                
                # 记录获胜统计
                self.total_games += 1
                if trajectory and trajectory.winner_id is not None:
                    winner_id = trajectory.winner_id
                    self.win_counts[winner_id] = self.win_counts.get(winner_id, 0) + 1
                
                # 记录胜者预测
                if trajectory and trajectory.winner_prediction:
                    self.predictions[sample_name] = trajectory.winner_prediction
                    self.logger.info(f"  ✓ 预测: {trajectory.winner_prediction}")
                else:
                    self.predictions[sample_name] = "unknown"
                    self.logger.warning(f"  ⚠ 无有效预测")
                    
            except Exception as e:
                self.logger.error(f"  ✗ 样本处理失败: {e}")
                import traceback
                traceback.print_exc()
                self.predictions[sample_name] = "error"
        
        self.logger.info(f"\n✓ 推理完成: {len(self.predictions)} 个预测")
        
        # 打印获胜统计
        self.logger.info("\n" + "=" * 50)
        self.logger.info("🏆 玩家获胜统计")
        self.logger.info("=" * 50)
        for player in self.players:
            wins = self.win_counts.get(player.player_id, 0)
            win_rate = wins / self.total_games * 100 if self.total_games > 0 else 0
            self.logger.info(f"  {player.name}: {wins} 次获胜 ({win_rate:.1f}%)")
        self.logger.info(f"  总局数: {self.total_games}")
        self.logger.info("=" * 50)
        
        return self.predictions
    
    def save_results(self, checkpoint_name: str = "checkpoint_000001"):
        """
        保存结果为 evaluation.py 兼容格式，仅保留主输出。

        输出文件：
        1. checkpoint_000001_winner.npz  - 博弈胜者预测
        2. checkpoint_000001_winner.json - 博弈胜者预测（便于查看）

        文件格式与 inference_hybird.py 一致:
        - 使用 np.savez_compressed 保存
        - 字段为 name2reason（字典：sample_name -> prediction_text）
        """
        # 【修复】确保输出目录存在
        os.makedirs(self.output_dir, exist_ok=True)

        # 保存博弈胜者的预测（主输出）
        winner_name2reason = dict(self.predictions)
        winner_npz_path = os.path.join(self.output_dir, f"{checkpoint_name}_winner.npz")
        np.savez_compressed(winner_npz_path, name2reason=winner_name2reason)
        
        self.logger.info(f"✓ [胜者预测] {winner_npz_path}")
        self.logger.info(f"  样本数: {len(winner_name2reason)}")
        
        # 同时保存 JSON 格式（便于查看）
        winner_json_path = os.path.join(self.output_dir, f"{checkpoint_name}_winner.json")
        with open(winner_json_path, 'w', encoding='utf-8') as f:
            json.dump(winner_name2reason, f, indent=2, ensure_ascii=False)

        self.logger.info("✓ 已禁用每玩家原始预测文件保存")
        self.logger.info("✓ 当前仅保存 winner.npz 和 winner.json")
        
        # 返回主输出路径（胜者预测）
        return winner_npz_path
    
    def run(
        self,
        model_configs: List[Dict],
        dataset_name: str = None,
        split: str = "test"
    ):
        """
        完整推理流程
        
        :param model_configs: 模型配置列表
        :param dataset_name: 数据集名称
        :param split: 数据集划分
        """
        start_time = time.time()
        
        try:
            # 1. 加载配置
            self.load_config()
            
            # 2. 加载策略
            self.load_strategy()
            
            # 3. 加载数据集
            self.load_dataset(dataset_name, split)
            
            if len(self.data_loader.samples) == 0:
                self.logger.error("数据集为空，无法推理")
                return
            
            # 4. 解析消融配置
            self.inactive_players = self._parse_inactive_players('cfr_inference')
            if self.inactive_players:
                self.logger.info(f"⚡ 消融实验：非活跃玩家 = {self.inactive_players}")
            
            # 5. 加载模型（跳过非活跃玩家）
            self.load_models(model_configs, skip_players=self.inactive_players)
            
            # 6. 设置玩家（非活跃玩家使用 dummy）
            self.setup_players(model_configs, skip_players=self.inactive_players)
            
            # 7. 设置标签提取器
            self.setup_label_extractor()
            
            # 8. 设置驱动器
            self.setup_driver()
            
            # 9. 运行推理
            self.run_inference()
            
            # 10. 保存结果
            self.save_results()
            
            elapsed = time.time() - start_time
            self.logger.info(f"\n推理完成，耗时: {elapsed/60:.2f} 分钟")
            
        except Exception as e:
            self.logger.exception(f"推理失败: {e}")
            raise


def main():
    parser = argparse.ArgumentParser(description="MER 博弈 CFR 推理")
    
    parser.add_argument(
        "--config",
        type=str,
        default="OpenSpielGame/train_configs/cfr_training.yaml",
        help="配置文件路径（包含所有参数）"
    )
    
    # 以下参数可选，如果不指定则从配置文件读取
    parser.add_argument(
        "--strategy",
        type=str,
        default=None,
        help="预训练策略 JSON 文件路径（覆盖配置文件）"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="输出目录（覆盖配置文件）"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="数据集名称（默认从配置文件读取）"
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        choices=["train", "val", "test"],
        help="数据集划分"
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="最多处理的样本数"
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
    
    # 模型配置（支持多个玩家）- 命令行覆盖配置文件
    parser.add_argument(
        "--model_ckpts",
        nargs="+",
        type=str,
        default=None,
        help="模型权重路径列表（覆盖配置文件）"
    )
    parser.add_argument(
        "--device_ids",
        nargs="+",
        type=int,
        default=None,
        help="GPU 设备 ID 列表（覆盖配置文件）"
    )
    parser.add_argument(
        "--llm_types",
        nargs="+",
        type=str,
        default=None,
        help="LLM 类型列表（覆盖配置文件）"
    )
    
    args = parser.parse_args()
    
    # ========== 从配置文件读取参数 ==========
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
    
    if not os.path.exists(config_path):
        print(f"错误: 配置文件不存在: {config_path}")
        return
    
    cfg = OmegaConf.load(config_path)
    print(f"✓ 加载配置文件: {config_path}")
    
    # 读取 cfr_inference 配置（命令行参数优先）
    cfr_inf = cfg.get('cfr_inference', {})
    
    strategy_path = args.strategy or cfr_inf.get('strategy_path')
    split = args.split or cfr_inf.get('split', 'test')
    max_samples = args.max_samples if args.max_samples is not None else cfr_inf.get('max_samples')
    seed = args.seed if args.seed != 42 else cfr_inf.get('seed', 42)
    log_level = args.log_level if args.log_level != "INFO" else cfr_inf.get('log_level', 'INFO')
    
    # 读取数据集名称（用于自动生成输出目录）
    dataset_name = args.dataset or cfr_inf.get('dataset')
    if not dataset_name:
        # 从 datasets 配置读取
        datasets_cfg = cfg.get('datasets', {})
        if datasets_cfg:
            dataset_name = list(datasets_cfg.keys())[0]
    dataset_name = dataset_name or "mercaptionplus"
    
    if not strategy_path:
        print("错误: 请指定策略文件路径 (--strategy 或配置文件 cfr_inference.strategy_path)")
        return
    
    # 从策略路径提取策略表标识（如 cfr_20260108_160707）
    # 策略路径格式: output/cfr_training/cfr_20260108_160707/checkpoints/cfr_xxx.json
    import re
    strategy_id_match = re.search(r'(cfr_\d{8}_\d{6})', strategy_path)
    strategy_id = strategy_id_match.group(1) if strategy_id_match else "cfr_unknown"
    
    # 自动生成输出目录：output/results-{dataset}/cfr_inference/{strategy_id}
    if args.output:
        output_dir = args.output
    elif cfr_inf.get('output_dir'):
        output_dir = cfr_inf.get('output_dir')
    else:
        # 根据数据集名称和策略表标识自动生成
        dataset_slug = dataset_name.lower().replace("_", "").replace("-", "")
        output_dir = f"output/results-{dataset_slug}/cfr_inference/{strategy_id}"
    
    # 初始化日志，尽早捕获 stdout/stderr
    log_dir = os.path.join(output_dir, "logs")
    logger = setup_logging(log_dir, log_level, redirect_stdio=True)

    print(f"✓ 数据集: {dataset_name}")
    print(f"✓ 策略表: {strategy_id}")
    print(f"✓ 输出目录: {output_dir}")
    
    # ========== 读取玩家配置 ==========
    model_configs = []
    
    # 方式 1: 从命令行参数读取
    if args.model_ckpts:
        device_ids = args.device_ids or [0, 1, 2]
        llm_types = args.llm_types or ["Qwen25", "Llama31", "Gemma3"]
        player_names = ["Qwen", "Llama", "Gemma"]
        
        for i, ckpt in enumerate(args.model_ckpts):
            device_id = device_ids[i] if i < len(device_ids) else 0
            llm_type = llm_types[i] if i < len(llm_types) else "Qwen25"
            player_name = player_names[i] if i < len(player_names) else f"Player_{i}"
            model_configs.append({
                "name": player_name,
                "ckpt_path": ckpt,
                "device_id": device_id,
                "llm_type": llm_type
            })
        print(f"✓ 从命令行加载 {len(model_configs)} 个模型配置")
    
    # 方式 2: 从配置文件读取（默认）
    else:
        # 从 cfr.players 读取玩家配置
        if hasattr(cfg, 'cfr') and cfg.cfr.get('players'):
            for player_cfg in cfg.cfr.players:
                model_configs.append({
                    "name": player_cfg.get('name', 'Player'),
                    "ckpt_path": player_cfg.get('ckpt_path', ''),
                    "device_id": player_cfg.get('device_id', 0),
                    "llm_type": player_cfg.get('llm_type', 'Qwen25')
                })
            print(f"✓ 从配置文件加载 {len(model_configs)} 个玩家配置")
        else:
            print("错误: 配置文件中没有 cfr.players 配置")
            return
    
    if len(model_configs) == 0:
        print("错误: 没有找到模型配置")
        return
    
    # ========== 显示配置摘要 ==========
    print("\n" + "=" * 50)
    print("CFR 推理配置摘要")
    print("=" * 50)
    print(f"  策略文件: {strategy_path}")
    print(f"  输出目录: {output_dir}")
    print(f"  数据划分: {split}")
    print(f"  最大样本: {max_samples or '全部'}")
    print(f"  玩家数量: {len(model_configs)}")
    for i, mc in enumerate(model_configs):
        print(f"    [{i}] {mc['name']} ({mc['llm_type']}) @ cuda:{mc['device_id']}")
    print("=" * 50 + "\n")
    
    # ========== 运行推理 ==========
    runner = CFRInferenceRunner(
        config_path=config_path,
        strategy_path=strategy_path,
        output_dir=output_dir,
        log_level=log_level,
        seed=seed,
        max_samples=max_samples,
        log_dir=log_dir,
        logger=logger
    )
    
    runner.run(
        model_configs=model_configs,
        dataset_name=args.dataset,
        split=split
    )


if __name__ == "__main__":
    main()
