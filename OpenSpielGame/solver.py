"""
solver.py - MER博弈的OpenSpiel集成模块

核心设计理念（对齐 chat_game_cfr_example.py）:
1. MERPokerGame: 纯博弈逻辑，不包含任何LLM调用
2. MERPokerState: 状态管理，支持Chance Node感知概率
3. MERObserver: 信息集字符串生成器
4. MERCFRSolver: OS-MCCFR求解器封装，支持手动遗憾更新
5. PerceptionSampler: LLM感知概率的抽象层（启发1的实现）

"""

import logging
import math
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Any

import numpy as np

# 从 constants.py 导入（单一数据源）
try:
    from .constants import (
        MERAction, ACTION_NAMES, ACTION_SHORT,
        NUM_CONFIDENCE_LEVELS, DEFAULT_NUM_PLAYERS,
        DEFAULT_NUM_ROUNDS, DEFAULT_EPSILON, DEFAULT_WEIGHT_MAX,
        DEFAULT_SQRT_MAX_RAW
    )
except ImportError:
    from constants import (
        MERAction, ACTION_NAMES, ACTION_SHORT,
        NUM_CONFIDENCE_LEVELS, DEFAULT_NUM_PLAYERS,
        DEFAULT_NUM_ROUNDS, DEFAULT_EPSILON, DEFAULT_WEIGHT_MAX,
        DEFAULT_SQRT_MAX_RAW
    )

logger = logging.getLogger("MERSolver")


# ==========================================
# 2. 感知采样器（启发1的实现）
# ==========================================

class PerceptionSampler:
    """
    感知采样器：将LLM的置信度输出建模为Chance Node概率
    
    核心思想（来自Steering论文的启发1）:
    - 将LLM的感知结果视为"自然"的随机选择
    - q_llm表示LLM选择该感知结果的概率
    - 在确定性模式下，q_llm = 1.0（假设LLM总是给出相同结果）
    
    支持的模式:
    - deterministic: q = 1.0（默认，用于初期实验）
    - entropy_based: q = softmax(logits)（基于LLM输出的logits）
    - temperature_scaled: q = softmax(logits/T)（温度缩放）
    """
    
    def __init__(self, mode: str = "deterministic", temperature: float = 1.0):
        """
        初始化感知采样器
        
        :param mode: 采样模式 ("deterministic", "entropy_based", "temperature_scaled")
        :param temperature: 温度参数（仅用于temperature_scaled模式）
        """
        self.mode = mode
        self.temperature = temperature
        logger.info(f"PerceptionSampler初始化: mode={mode}, temperature={temperature}")
    
    def get_probability(
        self, 
        confidence_level: int,
        logits: Optional[np.ndarray] = None
    ) -> float:
        """
        获取感知概率 q_llm
        
        :param confidence_level: 离散化后的置信度档位 (0-4)
        :param logits: LLM输出的原始logits（可选，用于非确定性模式）
        :return: 感知概率 q_llm ∈ (0, 1]
        """
        if self.mode == "deterministic":
            # 确定性模式：假设LLM总是给出相同结果
            return 1.0
        
        elif self.mode == "entropy_based" and logits is not None:
            # 基于熵的模式：使用softmax计算概率
            probs = self._softmax(logits)
            return float(probs[confidence_level])
        
        elif self.mode == "temperature_scaled" and logits is not None:
            # 温度缩放模式
            scaled_logits = logits / self.temperature
            probs = self._softmax(scaled_logits)
            return float(probs[confidence_level])
        
        # 默认返回1.0
        return 1.0
    
    @staticmethod
    def _softmax(x: np.ndarray) -> np.ndarray:
        """数值稳定的softmax"""
        exp_x = np.exp(x - np.max(x))
        return exp_x / exp_x.sum()


# ==========================================
# 3. 信息集记录（对齐chat_game_cfr_example.py的InfoStateRecord）
# ==========================================

@dataclass
class InfoStateRecord:
    """
    信息集记录：存储单个信息集的所有相关数据
    
    设计目的:
    - 提供信息集到策略的映射
    - 支持手动遗憾更新
    - 追踪访问历史用于调试
    """
    infoset_key: str                          # 信息集字符串（唯一标识符）
    player_id: int                            # 所属玩家ID
    legal_actions: List[int] = field(default_factory=list)  # 合法动作列表
    cumulative_regret: Dict[int, float] = field(default_factory=dict)  # 累积遗憾
    cumulative_strategy: Dict[int, float] = field(default_factory=dict)  # 累积策略
    visit_count: int = 0                      # 访问次数
    
    def get_current_strategy(self, epsilon: float = DEFAULT_EPSILON) -> Dict[int, float]:
        """
        获取当前策略（遗憾匹配）
        
        算法: σ(a) = max(r(a), 0) / Σ max(r(a'), 0)
        若所有遗憾为负，则均匀分布
        
        :param epsilon: 最小动作概率（保证探索）
        :return: 动作概率分布
        """
        if not self.legal_actions:
            return {}
        
        # 计算正遗憾和
        positive_regrets = {a: max(self.cumulative_regret.get(a, 0.0), 0.0) 
                          for a in self.legal_actions}
        total_positive = sum(positive_regrets.values())
        
        # 遗憾匹配策略
        if total_positive > 0:
            strategy = {a: positive_regrets[a] / total_positive for a in self.legal_actions}
        else:
            # 未知信息集：优先选择 CHECK (保守策略)
            # CHECK 概率 70%，其他动作均分剩余 30%
            check_action = 1  # MERAction.CHECK
            if check_action in self.legal_actions:
                other_actions = [a for a in self.legal_actions if a != check_action]
                check_prob = 0.7
                other_prob = 0.3 / len(other_actions) if other_actions else 0.0
                strategy = {a: other_prob for a in self.legal_actions}
                strategy[check_action] = check_prob
            else:
                # 如果 CHECK 不合法，均匀分布
                uniform_prob = 1.0 / len(self.legal_actions)
                strategy = {a: uniform_prob for a in self.legal_actions}
        
        # 应用epsilon下界（保证探索）
        for a in self.legal_actions:
            strategy[a] = max(strategy[a], epsilon)
        
        # 重新归一化
        total = sum(strategy.values())
        strategy = {a: p / total for a, p in strategy.items()}
        
        return strategy
    
    def get_average_strategy(self) -> Dict[int, float]:
        """
        获取平均策略（用于最终输出）
        
        算法: σ_avg(a) = Σ σ(a) / T
        """
        if not self.legal_actions:
            return {}
        
        total = sum(self.cumulative_strategy.get(a, 0.0) for a in self.legal_actions)
        
        if total > 0:
            return {a: self.cumulative_strategy.get(a, 0.0) / total 
                   for a in self.legal_actions}
        else:
            # 未知信息集：优先选择 CHECK (保守策略)
            # CHECK 概率 70%，其他动作均分剩余 30%
            check_action = 1  # MERAction.CHECK
            if check_action in self.legal_actions:
                other_actions = [a for a in self.legal_actions if a != check_action]
                check_prob = 0.7
                other_prob = 0.3 / len(other_actions) if other_actions else 0.0
                strategy = {a: other_prob for a in self.legal_actions}
                strategy[check_action] = check_prob
                return strategy
            else:
                # 如果 CHECK 不合法，均匀分布
                uniform_prob = 1.0 / len(self.legal_actions)
                return {a: uniform_prob for a in self.legal_actions}
    
    def update_regret(self, action: int, regret_value: float):
        """更新单个动作的累积遗憾"""
        self.cumulative_regret[action] = self.cumulative_regret.get(action, 0.0) + regret_value
    
    def update_strategy(self, strategy: Dict[int, float], weight: float = 1.0):
        """
        更新累积策略
        
        【OS-MCCFR 标准】使用到达概率加权
        
        :param strategy: 当前策略分布
        :param weight: 策略累加权重（标准 OS-MCCFR 使用 π_i(z)/q(z)）
        """
        for a, prob in strategy.items():
            self.cumulative_strategy[a] = self.cumulative_strategy.get(a, 0.0) + weight * prob
        self.visit_count += 1


# ==========================================
# 4. 博弈状态（简化版，专注于信息集生成）
# ==========================================

@dataclass
class MERGameState:
    """
    MER博弈状态：纯数据结构，不包含任何逻辑
    
    设计理念:
    - 最小化状态，仅保留生成信息集所需的数据
    - 与OpenSpiel的State概念解耦（不继承pyspiel.State）
    - 便于序列化和调试
    """
    # 玩家置信度历史：confidence_history[player][round] = level (0-4)
    confidence_history: List[List[int]] = field(default_factory=list)
    
    # 动作历史：action_history[i] = (player_id, action_id)
    action_history: List[Tuple[int, int]] = field(default_factory=list)
    
    # 当前轮次（0=Preflop, 1-N=揭示轮）
    current_round: int = 0
    
    # 当前行动玩家
    current_player: int = 0
    
    # 弃牌标记
    folded: List[bool] = field(default_factory=list)
    
    # 是否为终局
    is_terminal: bool = False
    
    # 最终收益（仅终局时有效）
    payoffs: List[float] = field(default_factory=list)
    
    def __post_init__(self):
        """初始化默认值"""
        if not self.confidence_history:
            self.confidence_history = [[] for _ in range(DEFAULT_NUM_PLAYERS)]
        if not self.folded:
            self.folded = [False] * DEFAULT_NUM_PLAYERS
        if not self.payoffs:
            self.payoffs = [0.0] * DEFAULT_NUM_PLAYERS


# ==========================================
# 5. 信息集生成器（MERObserver）
# ==========================================

class MERObserver:
    """
    信息集观察器：根据游戏状态生成信息集字符串
    
    支持五种模式：
    1. 简化模式 (simple): P{id}_C{conf}_R{round}
       - 只记录当前置信度和轮次
       - 状态数：3 × 10 × 4 = 120 种
       - 极速收敛，适合快速调试
       
    2. 紧凑模式 (compact): P{id}_C{conf}_R{round}_Thr{threat}_Δ{trend}
       - 记录置信度、威胁等级、抽象趋势变化
       - trend: 3档（δ<0→下降, δ=0→持平, δ>0→上升）
       - 状态数：3 × 10 × 4 × 3 × 3 = 1,080 种
       
    3. 序列模式 (sequence): P{id}_C{conf}_R{round}_Thr{threat}_Seq{seq}
       - 记录置信度变化序列（每轮变化方向编码）
       - seq: 3^3 = 27 种组合（↓/→/↑ 三值，最多3次变化）
       - 状态数：3 × 10 × 4 × 3 × 27 = 9,720 种
       
    4. 中等模式 (medium): P{id}_H[c0,c1,...]_R{round}
       - 记录玩家自身的完整置信度历史
       - 状态数：~30,000 种（10 + 100 + 1000 + 10000 per player per round）
       
    5. 完整模式 (full): P{id}_H[{h0},{h1},...,{hN}]_R{round}_Act:{p0:R|p1:C|...}
       - 记录完整置信度历史和动作历史
       - 状态数：万亿级，理论最优但难收敛
    """
    
    # 信息集模式
    mode: str = "simple"  # 默认使用简化模式
    
    @classmethod
    def set_mode(cls, mode: str):
        """设置信息集生成模式"""
        valid_modes = ("blind", "simple", "compact", "sequence", "medium", "full")
        if mode not in valid_modes:
            raise ValueError(f"模式必须是 {valid_modes} 之一，收到: {mode}")
        cls.mode = mode
        mode_states = {
            "blind": "~12",
            "simple": "~120",
            "compact": "~1,080",
            "sequence": "~9,720",
            "medium": "~30,000",
            "full": "万亿级"
        }
        logger.info(f"📋 信息集模式设置为: {mode} (状态数: {mode_states.get(mode, '未知')})")
    
    @classmethod
    def get_infoset_string(cls, state: MERGameState, player_id: int) -> str:
        """
        生成玩家的信息集字符串
        
        :param state: 当前游戏状态
        :param player_id: 观察者玩家ID
        :return: 信息集字符串
        """
        if cls.mode == "blind":
            return cls._get_blind_infoset(state, player_id)
        elif cls.mode == "simple":
            return cls._get_simple_infoset(state, player_id)
        elif cls.mode == "compact":
            return cls._get_compact_infoset(state, player_id)
        elif cls.mode == "sequence":
            return cls._get_sequence_infoset(state, player_id)
        elif cls.mode == "medium":
            return cls._get_medium_infoset(state, player_id)
        else:  # full
            return cls._get_full_infoset(state, player_id)
    
    @staticmethod
    def _get_blind_infoset(state: MERGameState, player_id: int) -> str:
        """
        盲式最小信息集：P{id}_R{round}

        仅保留玩家 ID 与回合数，不包含置信度与动作历史。
        用于“取消交互设计”的盲式训练。
        """
        return f"P{player_id}_R{state.current_round}"

    @staticmethod
    def _get_simple_infoset(state: MERGameState, player_id: int) -> str:
        """
        简化信息集：P{id}_C{conf}_R{round}
        
        状态空间：3 × 10 × 3 = 90 种
        - 3 玩家
        - 10 置信度级别 (0-9)
        - 3 轮次 (0, 1, 2)
        """
        # 获取当前置信度（最新的一个）
        hist = state.confidence_history[player_id]
        current_conf = hist[-1] if hist else 2  # 默认置信度 LEVEL_2（LLM 很少触及的独立档位）
        
        return f"P{player_id}_C{current_conf}_R{state.current_round}"
    
    @staticmethod
    def _get_full_infoset(state: MERGameState, player_id: int) -> str:
        """
        完整信息集：P{id}_H[{h0},{h1},...,{hN}]_R{round}_Act:{p0:R|p1:C|...}
        
        状态空间：指数级（取决于历史长度）
        """
        # 1. 玩家前缀
        prefix = f"P{player_id}"
        
        # 2. 置信度历史（只包含到当前轮次，避免泄露未来信息）
        hist = state.confidence_history[player_id][:state.current_round + 1]
        hist_str = ",".join(str(h) for h in hist) if hist else "-1"
        hand_part = f"H[{hist_str}]"
        
        # 3. 当前轮次
        round_part = f"R{state.current_round}"
        
        # 4. 动作历史
        if not state.action_history:
            action_part = "Act:empty"
        else:
            actions = "|".join(f"p{p}:{ACTION_SHORT[a]}" for p, a in state.action_history)
            action_part = f"Act:{actions}"
        
        return f"{prefix}_{hand_part}_{round_part}_{action_part}"
    
    # ==========================================
    # 辅助函数：从状态提取局势特征
    # ==========================================
    
    @staticmethod
    def _compute_threat_level(action_history: List[Tuple[int, int]]) -> int:
        """
        计算威胁等级：统计 RAISE 次数
        
        :param action_history: 动作历史 [(player_id, action_id), ...]
        :return: 威胁等级 (0=无RAISE, 1=单次RAISE, 2=多次RAISE)
        """
        raise_count = sum(1 for _, action in action_history if action == MERAction.RAISE)
        if raise_count == 0:
            return 0  # 低威胁
        elif raise_count == 1:
            return 1  # 中威胁
        else:
            return 2  # 高威胁
    
    @staticmethod
    def _compute_confidence_trend(confidence_history: List[int]) -> int:
        """
        计算置信度趋势：当前置信度 vs 初始置信度（抽象3档，阈值=2）
        
        【设计理念】
        Compact 模式追求状态压缩，小幅波动视为持平
        只有显著变化（|δ|≥2）才认为趋势改变
        
        :param confidence_history: 置信度历史 [level0, level1, ...]
        :return: 趋势值 (0=下降, 1=持平, 2=上升)
        """
        if len(confidence_history) < 2:
            return 1  # 只有一个数据点，视为持平
        
        initial = confidence_history[0]
        current = confidence_history[-1]
        delta = current - initial
        
        if delta <= -2:
            return 0  # 显著下降
        elif delta >= 2:
            return 2  # 显著上升
        else:
            return 1  # 持平（包括 -1, 0, +1）
    
    @staticmethod
    def _compute_change_sequence(confidence_history: List[int]) -> int:
        """
        计算置信度变化序列编码（精确捕捉每轮变化）
        
        【设计理念】
        Sequence 模式追求精确捕捉各回合间的变化，任何变化都有意义
        阈值=1：δ<0 → ↓，δ=0 → →，δ>0 → ↑
        
        记录每轮的变化方向，编码为3进制数（最多3次变化）
        
        示例:
        - [4,5,7,8] → [↑,↑,↑] → 2*9 + 2*3 + 2*1 = 26
        - [4,4,4,6] → [→,→,↑] → 1*9 + 1*3 + 2*1 = 14
        - [7,5,3,2] → [↓,↓,↓] → 0*9 + 0*3 + 0*1 = 0
        - [5,4,5,4] → [↓,↑,↓] → 0*9 + 2*3 + 0*1 = 6  (震荡模式)
        
        :param confidence_history: 置信度历史
        :return: 序列编码 (0-26)
        """
        if len(confidence_history) <= 1:
            return 13  # 无变化，返回中间值 (1*9 + 1*3 + 1*1)
        
        # 计算每轮变化方向（精确，任何变化都有意义）
        changes = []
        for i in range(1, min(len(confidence_history), 4)):  # 最多3次变化
            delta = confidence_history[i] - confidence_history[i-1]
            if delta < 0:
                changes.append(0)  # 下降（任何程度）
            elif delta > 0:
                changes.append(2)  # 上升（任何程度）
            else:
                changes.append(1)  # 持平（仅 δ=0）
        
        # 填充到3位
        while len(changes) < 3:
            changes.append(1)  # 未发生的变化视为持平
        
        # 编码为3进制数
        seq_code = changes[0] * 9 + changes[1] * 3 + changes[2]
        return seq_code
    
    # ==========================================
    # Compact 模式：抽象趋势变化
    # ==========================================
    
    @staticmethod
    def _get_compact_infoset(state: MERGameState, player_id: int) -> str:
        """
        紧凑信息集：P{id}_C{conf}_R{round}_Thr{threat}_Δ{trend}
        
        状态空间：3 × 10 × 4 × 3 × 3 = 1,080 种
        
        核心思想：用抽象趋势（δ<0/=0/>0）替代完整历史
        """
        hist = state.confidence_history[player_id]
        current_conf = hist[-1] if hist else 2
        
        threat = MERObserver._compute_threat_level(state.action_history)
        trend = MERObserver._compute_confidence_trend(hist)
        
        return f"P{player_id}_C{current_conf}_R{state.current_round}_Thr{threat}_Δ{trend}"
    
    # ==========================================
    # Sequence 模式：变化序列编码
    # ==========================================
    
    @staticmethod
    def _get_sequence_infoset(state: MERGameState, player_id: int) -> str:
        """
        序列信息集：P{id}_C{conf}_R{round}_Thr{threat}_Seq{seq}
        
        状态空间：3 × 10 × 4 × 3 × 27 = 9,720 种
        
        核心思想：记录完整的变化序列（每轮↓/→/↑），捕获变化模式
        """
        hist = state.confidence_history[player_id]
        current_conf = hist[-1] if hist else 2
        
        threat = MERObserver._compute_threat_level(state.action_history)
        seq = MERObserver._compute_change_sequence(hist)
        
        return f"P{player_id}_C{current_conf}_R{state.current_round}_Thr{threat}_Seq{seq}"
    
    # ==========================================
    # Medium 模式：完整置信度历史
    # ==========================================
    
    @staticmethod
    def _get_medium_infoset(state: MERGameState, player_id: int) -> str:
        """
        中等信息集：P{id}_H[c0,c1,...]_R{round}
        
        状态空间：
        - R0: 3 × 10 = 30
        - R1: 3 × 100 = 300
        - R2: 3 × 1000 = 3,000
        - R3: 3 × 10000 = 30,000
        - 总计：~33,330 种
        
        核心思想：保留玩家自身的完整置信度历史，但不记录动作历史
        """
        # 置信度历史（只包含到当前轮次）
        hist = state.confidence_history[player_id][:state.current_round + 1]
        hist_str = ",".join(str(h) for h in hist) if hist else "2"  # 默认 LEVEL_2
        
        return f"P{player_id}_H[{hist_str}]_R{state.current_round}"
    
    @staticmethod
    def get_legal_actions(state: MERGameState, player_id: int) -> List[int]:
        """
        获取合法动作列表
        
        规则:
        - 已弃牌玩家无合法动作
        - FOLD: 总是合法
        - CHECK: 无人加注时合法
        - CALL: 有人加注时合法
        - RAISE: 总是合法
        
        :param state: 当前游戏状态
        :param player_id: 玩家ID
        :return: 合法动作ID列表
        """
        if state.folded[player_id]:
            return []
        
        # 简化版本：所有动作都合法
        # 实际可以根据下注状态细化
        return [MERAction.FOLD, MERAction.CHECK, MERAction.CALL, MERAction.RAISE]


# ==========================================
# 6. 路径概率计算器（OS-MCCFR配置容器）
# ==========================================

class PathProbabilityCalculator:
    """
    路径概率计算器：存储 OS-MCCFR 权重裁剪配置
    
    【设计说明】
    实际的权重计算和裁剪逻辑已移至 MERCFRSolver.update_from_trajectory()
    此类保留用于：
    1. 存储配置参数 (epsilon, max_weight)
    2. 保持向后兼容性（部分测试代码可能依赖此类）
    3. 存储 perception_sampler 引用
    """
    
    def __init__(
        self,
        perception_sampler: PerceptionSampler,
        epsilon: float = DEFAULT_EPSILON,
        max_weight: float = DEFAULT_WEIGHT_MAX
    ):
        """
        初始化配置容器
        
        :param perception_sampler: 感知采样器实例
        :param epsilon: 最小动作概率（探索保证）
        :param max_weight: 最大权重（裁剪阈值，用于 update_from_trajectory）
        """
        self.perception_sampler = perception_sampler
        self.epsilon = epsilon
        self.max_weight = max_weight
        self.log_max_weight = math.log(max_weight)


# ==========================================
# 7. MER CFR求解器（核心类）
# ==========================================

class MERCFRSolver:
    """
    MER博弈的CFR求解器
    
    核心功能:
    1. 策略查询: get_action_probabilities()
    2. 手动遗憾更新: update_from_trajectory()（标准 OS-MCCFR）
    3. 热身训练: warmup_train()
    
    设计特点:
    - 实现标准 Outcome Sampling MCCFR
    - 完全可控的遗憾更新逻辑
    - 支持外部轨迹的重要性采样学习
    - 使用 ε-贪婪采样策略进行探索
    """
    
    def __init__(
        self,
        num_players: int = DEFAULT_NUM_PLAYERS,
        perception_sampler: Optional[PerceptionSampler] = None,
        epsilon: float = DEFAULT_EPSILON,
        max_weight: float = DEFAULT_WEIGHT_MAX,
        sqrt_max_raw: float = DEFAULT_SQRT_MAX_RAW,  # 开根号前的原始权重上限
        infoset_mode: str = "simple"
    ):
        """
        初始化CFR求解器
        
        :param num_players: 玩家数量
        :param perception_sampler: 感知采样器（用于计算q_llm）
        :param epsilon: 最小动作概率（ε-贪婪探索率）
        :param max_weight: 最终权重硬上限
        :param sqrt_max_raw: 开根号前的原始权重上限（W = sqrt(min(W_raw, sqrt_max_raw))）
        :param infoset_mode: 信息集模式，'simple'（45种状态）或 'full'（指数级）
        """
        self.num_players = num_players
        self.epsilon = epsilon
        self.sqrt_max_raw = sqrt_max_raw  # 保存开根号前的权重上限
        self.infoset_mode = infoset_mode
        
        # 设置信息集生成模式
        MERObserver.set_mode(infoset_mode)
        
        # 信息集记录存储
        self.infoset_records: Dict[str, InfoStateRecord] = {}
        
        # 感知采样器（默认确定性模式）
        if perception_sampler is None:
            perception_sampler = PerceptionSampler(mode="deterministic")
        self.perception_sampler = perception_sampler
        
        # 路径概率计算器
        self.path_calculator = PathProbabilityCalculator(
            perception_sampler=perception_sampler,
            epsilon=epsilon,
            max_weight=max_weight
        )
        
        # 训练统计
        self.total_iterations = 0
        
        # 【Warmup 控制】
        # warmup 期间只更新遗憾值，不更新平均策略
        # 这样可以让模型在初期找到正确方向，但不把初期的混乱表现固化到最终策略中
        self.warmup_iterations = 0  # 由外部设置
        self.current_sample_count = 0  # 当前已处理的样本数
        
        # 计算理论状态数
        mode_info = {
            "blind": ("盲式最小模式", "~12"),
            "simple": ("简化模式", "~120"),
            "compact": ("紧凑模式", "~1,080"),
            "sequence": ("序列模式", "~9,720"),
            "medium": ("中等模式", "~30,000"),
            "full": ("完整模式", "万亿级")
        }
        mode_desc, max_states = mode_info.get(infoset_mode, ("未知模式", "未知"))
        
        logger.info(f"MERCFRSolver初始化: {mode_desc} (OS-MCCFR), 理论状态数={max_states}, epsilon={epsilon}")
    
    def get_or_create_infoset(
        self, 
        infoset_key: str,
        player_id: int,
        legal_actions: Optional[List[int]] = None
    ) -> InfoStateRecord:
        """
        获取或创建信息集记录
        
        :param infoset_key: 信息集字符串
        :param player_id: 玩家ID
        :param legal_actions: 合法动作列表（仅创建时使用）
        :return: InfoStateRecord实例
        """
        if infoset_key not in self.infoset_records:
            if legal_actions is None:
                legal_actions = list(MERAction)  # 默认所有动作
            
            self.infoset_records[infoset_key] = InfoStateRecord(
                infoset_key=infoset_key,
                player_id=player_id,
                legal_actions=legal_actions
            )
            logger.debug(f"创建新信息集: {infoset_key}")
        
        return self.infoset_records[infoset_key]
    
    def get_action_probabilities(
        self,
        player_id: int,
        confidence_history: List[int],
        action_history: List[Tuple[int, int]],
        current_round: int = 0
    ) -> Dict[int, float]:
        """
        获取当前状态下的动作概率分布
        
        :param player_id: 玩家ID (0-2)
        :param confidence_history: 该玩家的置信度历史 [h0, h1, ...]
        :param action_history: 动作历史 [(player_id, action_id), ...]
        :param current_round: 当前轮次
        :return: {action_id: probability} 字典
        """
        # 构造状态对象
        state = MERGameState(
            current_round=current_round,
            current_player=player_id
        )
        state.confidence_history[player_id] = confidence_history
        state.action_history = action_history
        
        # 生成信息集字符串
        infoset_key = MERObserver.get_infoset_string(state, player_id)
        legal_actions = MERObserver.get_legal_actions(state, player_id)
        
        # 获取或创建信息集记录
        record = self.get_or_create_infoset(infoset_key, player_id, legal_actions)
        
        # 返回当前策略
        return record.get_current_strategy(self.epsilon)
    
    def sample_action_with_probs(
        self,
        player_id: int,
        confidence_history: List[int],
        action_history: List[Tuple[int, int]],
        current_round: int = 0,
        legal_actions: Optional[List[int]] = None
    ) -> Tuple[int, float, float, Dict[int, float]]:
        """
        【OS-MCCFR 核心】带概率追踪的动作采样
        
        使用 ε-贪婪策略进行探索采样，同时返回：
        - 策略概率 σ(a): 遗憾匹配得到的当前策略下选该动作的概率
        - 采样概率 q(a): 实际采样时选该动作的概率（包含探索）
        
        公式: q(a) = ε × (1/|A|) + (1-ε) × σ(a)
        
        :param player_id: 玩家ID
        :param confidence_history: 置信度历史
        :param action_history: 动作历史
        :param current_round: 当前轮次
        :param legal_actions: 合法动作列表（None则使用默认）
        :return: (action, sigma_a, q_a, full_strategy)
                 - action: 采样选中的动作
                 - sigma_a: 该动作的策略概率 σ(I, a)
                 - q_a: 该动作的采样概率 q(I, a)
                 - full_strategy: 完整策略分布（用于遗憾更新）
        """
        if legal_actions is None:
            legal_actions = list(MERAction)
        
        n_actions = len(legal_actions)
        
        # 1. 获取遗憾匹配策略 σ(I)
        sigma = self.get_action_probabilities(
            player_id, confidence_history, action_history, current_round
        )
        
        # 2. 构建采样策略 q(I) = ε × uniform + (1-ε) × σ
        q = {}
        for a in legal_actions:
            sigma_a = sigma.get(a, 1.0 / n_actions)
            q[a] = self.epsilon * (1.0 / n_actions) + (1.0 - self.epsilon) * sigma_a
        
        # 归一化（理论上已归一化，但为安全起见）
        total_q = sum(q.values())
        q = {a: p / total_q for a, p in q.items()}
        
        # 3. 按采样策略 q 随机选择动作
        actions = list(q.keys())
        probs = [q[a] for a in actions]
        chosen_action = int(np.random.choice(actions, p=probs))
        
        # 4. 返回选中动作及其概率
        sigma_a = sigma.get(chosen_action, 1.0 / n_actions)
        q_a = q[chosen_action]
        
        return chosen_action, sigma_a, q_a, sigma
    
    def get_average_strategy(
        self,
        player_id: int,
        confidence_history: List[int],
        action_history: List[Tuple[int, int]],
        current_round: int = 0
    ) -> Dict[int, float]:
        """
        获取平均策略（用于最终推理）
        
        :param player_id: 玩家ID
        :param confidence_history: 置信度历史
        :param action_history: 动作历史
        :param current_round: 当前轮次
        :return: 平均策略分布
        """
        # 构造状态对象
        state = MERGameState(
            current_round=current_round,
            current_player=player_id
        )
        state.confidence_history[player_id] = confidence_history
        state.action_history = action_history
        
        # 生成信息集字符串
        infoset_key = MERObserver.get_infoset_string(state, player_id)
        
        if infoset_key in self.infoset_records:
            return self.infoset_records[infoset_key].get_average_strategy()
        else:
            # 未见过的信息集，返回均匀分布
            return {a: 0.25 for a in MERAction}
    
    def update_from_trajectory(
        self,
        trajectory: List[Dict],
        final_payoffs: List[float],
        reach_probs: Dict[int, float] = None,
        sample_prob: float = 1.0
    ):
        """
        【标准 OS-MCCFR】根据一局博弈的轨迹更新遗憾值
        
        算法公式 (Outcome Sampling MCCFR):
        
        对于玩家 i 在信息集 I 处：
        1. 重要性采样权重: W = π_{-i}(z) / q(z)
           - π_{-i}(z): 对手到达终局的策略概率乘积
           - q(z): 采样到该路径的概率
        
        2. 遗憾更新:
           regret_i(I, a) += W × u_i(z) × (I[a==a*] - σ(a))
           其中:
           - u_i(z): 玩家 i 的终局收益
           - a*: 实际采样的动作
           - σ(a): 当前策略下选动作 a 的概率
           - I[a==a*]: 指示函数，a==a* 时为1，否则为0
        
        3. 平均策略更新:
           cumulative_strategy(a) += (π_i(z) / q(z)) × σ(a)
        
        :param trajectory: 决策轨迹（包含 sigma_a, q_a, full_strategy）
        :param final_payoffs: 终局收益 [p0, p1, p2]
        :param reach_probs: π_i(z) 每个玩家的到达概率
        :param sample_prob: q(z) 采样概率
        """
        if not trajectory:
            logger.warning("空轨迹，跳过更新")
            return
        
        # 如果没有传入概率，从轨迹中计算
        if reach_probs is None:
            reach_probs = self._compute_reach_probs_from_trajectory(trajectory)
        if sample_prob <= 0:
            sample_prob = self._compute_sample_prob_from_trajectory(trajectory)
        
        # 防止除零
        sample_prob = max(sample_prob, 1e-10)
        
        logger.info(f"  📊 OS-MCCFR更新: π={reach_probs}, q(z)={sample_prob:.6f}, 决策点数={len(trajectory)}")
        
        # 【权重裁剪】防止重要性采样权重爆炸
        # 参考 OpenSpiel 的实现，使用软裁剪 + 硬上限
        MAX_WEIGHT = self.path_calculator.max_weight  # 默认 10000
        
        # 对每个决策点进行遗憾更新
        for step in trajectory:
            player = step["player"]
            infoset = step["infoset"]
            action_taken = step["action"]
            legal_actions = step.get("legal_actions", list(MERAction))
            
            # 获取或创建信息集记录
            record = self.get_or_create_infoset(infoset, player, legal_actions)
            
            # === 计算重要性采样权重 W = π_{-i}(z) / q(z) ===
            # π_{-i}(z) = 所有对手的到达概率乘积
            pi_minus_i = 1.0
            for other_player, pi_p in reach_probs.items():
                if other_player != player:
                    pi_minus_i *= pi_p
            
            W_raw = pi_minus_i / sample_prob
            
            # 【关键】权重裁剪：平方根压缩 + 硬上限
            # 步骤1: 先限制原始权重（防止极端值）
            # 步骤2: 对限制后的值取平方根（保留区分度）
            # 公式: W = sqrt(min(W_raw, sqrt_max_raw))
            # 效果: 1000倍差异 → ~31倍差异（优于对数的~2倍）
            W_clamped = min(W_raw, self.sqrt_max_raw)  # 开根号前限制
            W = math.sqrt(W_clamped)  # 平方根压缩
            W_final = min(W, MAX_WEIGHT)  # 最终硬上限（通常不会触发）
            
            # 打印权重变换过程
            logger.info(f"[遗憾权重] P{player} infoset={infoset[:30]}... | W_raw={W_raw:.2f} → sqrt({W_clamped:.2f})={W:.2f} → W_final={W_final:.2f}")
            if W_raw > self.sqrt_max_raw:
                logger.info(f"  ⚠️ [裁剪触发] W_raw={W_raw:.2f} > 上限={self.sqrt_max_raw}")
            
            W = W_final  # 使用最终权重
            
            # 获取当前策略（从轨迹记录或重新计算）
            current_strategy = step.get("full_strategy", record.get_current_strategy(self.epsilon))
            
            # === 遗憾更新（标准 OS-MCCFR 公式）===
            # regret(a) += W × u × (I[a==a*] - σ(a))
            u = final_payoffs[player]  # 玩家 i 的终局收益
            
            for a in legal_actions:
                sigma_a = current_strategy.get(a, 1.0 / len(legal_actions))
                
                # I[a==a*] - σ(a)
                indicator = 1.0 if a == action_taken else 0.0
                regret_term = indicator - sigma_a
                
                # W × u × (I[a==a*] - σ(a))
                weighted_regret = W * u * regret_term
                
                record.update_regret(a, weighted_regret)
            
            # === 平均策略更新（带权重，同样需要裁剪）===
            # 【关键】Warmup 期间跳过平均策略更新
            # 只有过了预热期，才把当前策略固化进最终策略表
            if self.current_sample_count > self.warmup_iterations:
                # cumulative_strategy(a) += (π_i(z) / q(z)) × σ(a)
                pi_i = reach_probs.get(player, 1.0)
                strategy_weight_raw = pi_i / sample_prob
                
                # 【关键】策略权重也需要裁剪（与遗憾权重保持一致）
                # 使用相同的平方根压缩策略
                strategy_weight_clamped = min(strategy_weight_raw, self.sqrt_max_raw)
                strategy_weight = math.sqrt(strategy_weight_clamped)
                strategy_weight_final = min(strategy_weight, MAX_WEIGHT)
                
                # 打印策略权重变换过程
                logger.info(f"[策略权重] P{player} infoset={infoset[:30]}... | W_raw={strategy_weight_raw:.2f} → sqrt({strategy_weight_clamped:.2f})={strategy_weight:.2f} → W_final={strategy_weight_final:.2f}")
                if strategy_weight_raw > self.sqrt_max_raw:
                    logger.info(f"  ⚠️ [裁剪触发] W_raw={strategy_weight_raw:.2f} > 上限={self.sqrt_max_raw}")
                
                record.update_strategy(current_strategy, weight=strategy_weight_final)
        
        # 更新计数器
        self.current_sample_count += 1
        self.total_iterations += 1
        
        # 日志区分 Warmup 和正式训练
        if self.current_sample_count <= self.warmup_iterations:
            logger.debug(f"Warmup 更新完成 ({self.current_sample_count}/{self.warmup_iterations}): 仅更新遗憾值")
        else:
            logger.debug(f"OS-MCCFR更新完成，总迭代次数: {self.total_iterations}")
    
    def _compute_reach_probs_from_trajectory(self, trajectory: List[Dict]) -> Dict[int, float]:
        """从轨迹中计算各玩家的到达概率 π_i(z)"""
        reach_probs = {}
        for step in trajectory:
            player = step["player"]
            sigma_a = step.get("sigma_a", 1.0)
            if player not in reach_probs:
                reach_probs[player] = 1.0
            reach_probs[player] *= sigma_a
        return reach_probs
    
    def _compute_sample_prob_from_trajectory(self, trajectory: List[Dict]) -> float:
        """从轨迹中计算采样概率 q(z)"""
        sample_prob = 1.0
        for step in trajectory:
            q_a = step.get("q_a", 1.0)
            sample_prob *= q_a
        return max(sample_prob, 1e-10)  # 防止为零
    
    def set_warmup_iterations(self, iterations: int):
        """
        设置 Warmup 迭代次数
        
        Warmup 期间（前 N 个样本）：
        - 遗憾值（Regret）：正常更新（学习方向）
        - 平均策略（Average Strategy）：不更新（不固化到最终策略）
        
        这样设计的好处：
        1. 让 CFR 在初期找到正确的博弈方向
        2. 避免初期的混乱表现污染最终策略
        3. Warmup 使用真实样本，不再使用合成轨迹
        
        :param iterations: 预热样本数（建议为总样本的 10%-20%）
        """
        self.warmup_iterations = iterations
        self.current_sample_count = 0  # 重置计数器
        logger.info(f"Warmup 设置完成: 前 {iterations} 个样本仅更新遗憾值，不更新平均策略")
    
    def is_in_warmup(self) -> bool:
        """检查当前是否在 Warmup 阶段"""
        return self.current_sample_count <= self.warmup_iterations
    
    def get_warmup_progress(self) -> str:
        """获取 Warmup 进度字符串"""
        if self.warmup_iterations == 0:
            return "Warmup 未启用"
        elif self.current_sample_count <= self.warmup_iterations:
            return f"Warmup 进行中: {self.current_sample_count}/{self.warmup_iterations}"
        else:
            return f"Warmup 已完成，正式训练中: {self.current_sample_count - self.warmup_iterations} 个样本"
    
    def format_advice_for_prompt(self, probs: Dict[int, float]) -> str:
        """将策略概率格式化为Prompt建议
        
        【重要】动作显示顺序：CHECK, CALL, RAISE, FOLD
        将 FOLD 放最后，避免 LLM 在概率相同时倾向选择列表首位的动作
        """
        # 【修改】按 KCRF 顺序显示（FOLD最后），而不是按概率排序
        # 这是因为 LLM 在概率相同时倾向选择前面的选项
        action_order = [1, 2, 3, 0]  # CHECK, CALL, RAISE, FOLD
        
        lines = ["Nash Equilibrium Strategy:"]
        for action_id in action_order:
            prob = probs.get(action_id, 0.0)
            if prob > 0.01:
                action_name = ACTION_NAMES[action_id]
                lines.append(f"  - {action_name}: {prob*100:.2f}%")  # 2位小数精度
        
        # 推荐动作选择逻辑：
        # 1. 按概率降序排列
        # 2. 概率相同时，按 KCRF 顺序优先（FOLD 最后）
        # 这是因为 sorted() 是稳定排序，相同概率会保持原序
        action_priority_order = [1, 2, 3, 0]  # K, C, R, F
        sorted_by_priority = sorted(
            probs.items(), 
            key=lambda x: (-x[1], action_priority_order.index(x[0]) if x[0] in action_priority_order else 99)
        )
        best_action = sorted_by_priority[0][0]
        lines.append(f"\n**Recommended**: {ACTION_NAMES[best_action]}")
        
        return "\n".join(lines)
    
    def get_confidence_weighted_ev(
        self,
        player_id: int,
        confidence_history: List[int],
        action_history: List[Tuple[int, int]],
        current_round: int = 0
    ) -> float:
        """
        计算置信度加权的 EV 值（用于推理态判定获胜者）
        
        【简化版设计】纯粹使用 CFR 策略和置信度，无人工偏置
        
        设计思路：
        1. 置信度本身就是预测质量的直接反映
        2. CFR 策略的「激进度」反映玩家对自身预测的博弈信心：
           - 愿意 RAISE 的玩家更可能有高质量预测
           - 倾向 FOLD 的玩家对预测信心不足
        
        公式：EV = normalized_conf × (1 + aggression_factor)
        
        其中 aggression_factor = π(RAISE) - π(FOLD)
        - 范围 [-1, +1]
        - 完全激进 (100% RAISE): +1
        - 完全保守 (100% FOLD): -1
        - 均衡策略: ≈ 0
        
        :param player_id: 玩家ID
        :param confidence_history: 置信度历史 [0-4]
        :param action_history: 动作历史 [(player_id, action), ...]
        :param current_round: 当前轮次
        :return: EV 值 ∈ [0, 2]
        """
        # 获取 CFR 策略概率
        probs = self.get_action_probabilities(
            player_id, confidence_history, action_history, current_round
        )
        
        # 当前置信度（使用最新值）
        current_conf = confidence_history[-1] if confidence_history else 2
        
        # 归一化置信度到 [0, 1]
        normalized_conf = current_conf / 4.0
        
        # 激进度因子：π(RAISE) - π(FOLD)
        # 这完全来自 CFR 学习到的策略，无人工偏置
        aggression_factor = probs.get(MERAction.RAISE, 0.25) - probs.get(MERAction.FOLD, 0.25)
        
        # 最终 EV：置信度 × (1 + 激进度因子)
        # 范围 [0, 2]，中点 1.0
        # - 高置信度 + 愿意 RAISE → 最高 EV (接近 2.0)
        # - 低置信度 + 倾向 FOLD → 最低 EV (接近 0)
        ev = normalized_conf * (1.0 + aggression_factor)
        
        return ev

    # ==================== 收敛性诊断 ====================

    def compute_cce_gap(self) -> Dict[str, Any]:
        """
        计算当前 ε-CCE 间隙（收敛性诊断）

        ε_i = (1/T) × Σ_I Σ_a max(0, R_i^T(I,a))
        ε   = max_i ε_i

        当 ε → 0 时，平均策略为 CCE。
        """
        T = max(self.current_sample_count, 1)

        player_positive_regret: Dict[int, float] = {}
        for record in self.infoset_records.values():
            pid = record.player_id
            pos_regret = sum(max(0.0, v) for v in record.cumulative_regret.values())
            player_positive_regret[pid] = player_positive_regret.get(pid, 0.0) + pos_regret

        per_player = {pid: total / T for pid, total in player_positive_regret.items()}
        for pid in range(self.num_players):
            per_player.setdefault(pid, 0.0)

        global_eps = max(per_player.values()) if per_player else 0.0

        return {"per_player": per_player, "epsilon": global_eps, "T": T}

    def compute_cce_gap_normalized(self, payoff_range: float = 100.0) -> Dict[str, Any]:
        """
        归一化 ε-CCE 间隙（无量纲，除以收益范围 Δu）

        ε_i^norm = ε_i / Δu，其中 Δu = u_max - u_min
        当 ε^norm < 0.05 时可声明近似收敛

        :param payoff_range: 收益范围（默认 100，即 payoff ∈ [-50, +50]）
        """
        raw = self.compute_cce_gap()
        denom = max(payoff_range, 1e-10)
        per_player_norm = {pid: v / denom for pid, v in raw["per_player"].items()}
        return {
            "per_player": per_player_norm,
            "epsilon": raw["epsilon"] / denom,
            "T": raw["T"],
            "payoff_range": payoff_range
        }

    def compute_strategy_entropy(self) -> Dict[str, Any]:
        """
        计算各玩家平均策略的 Shannon 熵

        H(σ̄(I)) = -Σ_a σ̄(a) log σ̄(a)
        熵稳定 → 策略已定型；熵持续下降 → 趋向确定性策略
        返回每位玩家的均值熵和全局均值
        """
        player_entropy_sum: Dict[int, float] = {}
        player_count: Dict[int, int] = {}

        for record in self.infoset_records.values():
            pid = record.player_id
            avg_sigma = record.get_average_strategy()
            if not avg_sigma:
                continue
            h = -sum(p * math.log(p + 1e-12) for p in avg_sigma.values() if p > 0)
            player_entropy_sum[pid] = player_entropy_sum.get(pid, 0.0) + h
            player_count[pid] = player_count.get(pid, 0) + 1

        per_player: Dict[int, float] = {
            pid: player_entropy_sum.get(pid, 0.0) / max(player_count.get(pid, 1), 1)
            for pid in range(self.num_players)
        }
        global_mean = sum(per_player.values()) / self.num_players if self.num_players > 0 else 0.0
        return {
            "per_player": per_player,
            "global_mean": global_mean,
            "total_infosets": sum(player_count.values())
        }

    def snapshot_average_strategy(self) -> Dict[str, Dict[int, float]]:
        """
        快照当前所有信息集的平均策略（用于策略稳定性 L1 距离计算）

        返回 {infoset_key: {action_id: prob, ...}, ...}
        调用者需保存前一份快照以与下一份对比
        """
        return {
            key: dict(record.get_average_strategy())
            for key, record in self.infoset_records.items()
        }

    def compute_strategy_jsd(self, prev_snapshot: Dict[str, Dict[int, float]]) -> Dict[str, Any]:
        """
        计算当前平均策略与上一次快照之间的平均 Jensen-Shannon 散度（JSD）

        JSD(σ_new ‖ σ_old) = H(0.5*σ_new + 0.5*σ_old) - 0.5*H(σ_new) - 0.5*H(σ_old)

        范围 [0, ln2 ≈ 0.693]（自然对数）
        0 表示策略完全不变（收敛）；比 L1 对小概率偏移更敏感，曲线更平滑

        :param prev_snapshot: 上一次 snapshot_average_strategy() 返回的快照
        :return: {per_player: {pid: mean_jsd}, global_mean: float, infosets_compared: int}
        """
        if not prev_snapshot:
            return {
                "per_player": {pid: 0.0 for pid in range(self.num_players)},
                "global_mean": 0.0,
                "infosets_compared": 0
            }

        def _h(p: float) -> float:
            return -p * math.log(p + 1e-12) if p > 1e-12 else 0.0

        player_jsd_sum: Dict[int, float] = {}
        player_count: Dict[int, int] = {}

        for key, record in self.infoset_records.items():
            if key not in prev_snapshot:
                continue
            new_sigma = record.get_average_strategy()
            old_sigma = prev_snapshot[key]
            pid = record.player_id
            all_actions = set(new_sigma) | set(old_sigma)
            jsd = 0.0
            for a in all_actions:
                p = new_sigma.get(a, 0.0)
                q = old_sigma.get(a, 0.0)
                m = 0.5 * (p + q)
                jsd += _h(m) - 0.5 * _h(p) - 0.5 * _h(q)
            player_jsd_sum[pid] = player_jsd_sum.get(pid, 0.0) + jsd
            player_count[pid] = player_count.get(pid, 0) + 1

        per_player: Dict[int, float] = {
            pid: player_jsd_sum.get(pid, 0.0) / max(player_count.get(pid, 1), 1)
            for pid in range(self.num_players)
        }
        global_mean = sum(per_player.values()) / self.num_players if self.num_players > 0 else 0.0
        return {
            "per_player": per_player,
            "global_mean": global_mean,
            "infosets_compared": sum(player_count.values())
        }

    def save_strategy(self, filepath: str):
        """保存策略到文件"""
        import json
        
        data = {
            "num_players": self.num_players,
            "epsilon": self.epsilon,
            "total_iterations": self.total_iterations,
            "infoset_mode": self.infoset_mode,
            "infosets": {}
        }
        
        for key, record in self.infoset_records.items():
            # 确保动作ID转为整数字符串（MERAction枚举需要用int()取值）
            data["infosets"][key] = {
                "player_id": record.player_id,
                "legal_actions": [int(a) for a in record.legal_actions],
                "cumulative_regret": {str(int(k)): v for k, v in record.cumulative_regret.items()},
                "cumulative_strategy": {str(int(k)): v for k, v in record.cumulative_strategy.items()},
                "visit_count": record.visit_count
            }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        
        logger.info(f"策略已保存到: {filepath}")
    
    def load_strategy(self, filepath: str):
        """从文件加载策略"""
        import json
        
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        self.num_players = data["num_players"]
        self.epsilon = data["epsilon"]
        self.total_iterations = data["total_iterations"]
        
        # 从策略文件读取 infoset_mode 并自动切换（兼容旧文件）
        saved_mode = data.get("infoset_mode", "simple")
        if saved_mode != self.infoset_mode:
            logger.warning(f"策略文件模式 '{saved_mode}' 与当前配置 '{self.infoset_mode}' 不一致，自动切换为 '{saved_mode}'")
            self.infoset_mode = saved_mode
            MERObserver.set_mode(saved_mode)
        
        self.infoset_records.clear()
        for key, record_data in data["infosets"].items():
            record = InfoStateRecord(
                infoset_key=key,
                player_id=record_data["player_id"],
                legal_actions=record_data["legal_actions"],
                cumulative_regret={int(k): v for k, v in record_data["cumulative_regret"].items()},
                cumulative_strategy={int(k): v for k, v in record_data["cumulative_strategy"].items()},
                visit_count=record_data["visit_count"]
            )
            self.infoset_records[key] = record
        
        logger.info(f"策略已加载: {len(self.infoset_records)} 个信息集")


# ==========================================
# 8. 便捷工厂函数
# ==========================================

def create_mer_solver(
    warmup_iterations: int = 0,
    perception_mode: str = "deterministic",
    infoset_mode: str = "simple",
    sqrt_max_raw: float = DEFAULT_SQRT_MAX_RAW,  # 开根号前的原始权重上限
    **kwargs
) -> MERCFRSolver:
    """
    创建并初始化MER CFR求解器
    
    【Warmup 机制说明】
    Warmup 现已改为"延迟记录"模式，而非"合成轨迹预热"：
    - 前 warmup_iterations 个真实样本：只更新遗憾值，不更新平均策略
    - 之后的样本：同时更新遗憾值和平均策略
    
    这意味着 Warmup 期间也会调用 LLM 进行真实博弈，但这些早期
    的混乱表现不会被固化到最终策略文件 (cfr_final.json) 中。
    
    :param warmup_iterations: 预热样本数（前 N 个样本不记录平均策略）
    :param perception_mode: 感知模式 ("deterministic", "entropy_based")
    :param infoset_mode: 信息集模式:
                         - "simple": ~120种状态 (P{id}_C{conf}_R{round})
                         - "compact": ~1,080种状态（当前置信度 + 抽象趋势）
                         - "sequence": ~9,720种状态（变化序列编码）
                         - "medium": ~30,000种状态（完整置信度历史）
                         - "full": 万亿级状态（完整历史 + 动作历史）
    :param sqrt_max_raw: 开根号前的原始权重上限 (W = sqrt(min(W_raw, sqrt_max_raw)))
    :param kwargs: 其他参数传递给MERCFRSolver
    :return: 初始化好的求解器实例
    """
    perception_sampler = PerceptionSampler(mode=perception_mode)
    solver = MERCFRSolver(
        perception_sampler=perception_sampler, 
        infoset_mode=infoset_mode,
        sqrt_max_raw=sqrt_max_raw,
        **kwargs
    )
    
    # 设置 Warmup 迭代次数（不再调用旧的 warmup_train）
    if warmup_iterations > 0:
        solver.set_warmup_iterations(warmup_iterations)
    
    return solver


# ==========================================
# 9. 测试代码
# ==========================================

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(name)s - %(levelname)s - %(message)s'
    )
    
    logger.info("="*60)
    logger.info("MER CFR Solver 测试")
    logger.info("="*60)
    
    # 1. 创建求解器
    solver = create_mer_solver(warmup_iterations=500)
    
    # 2. 测试策略查询
    test_cases = [
        {"player": 0, "conf_hist": [3], "act_hist": [], "round": 0},
        {"player": 1, "conf_hist": [2, 3], "act_hist": [(0, MERAction.RAISE)], "round": 1},
        {"player": 2, "conf_hist": [4, 4, 3], "act_hist": [(0, MERAction.FOLD), (1, MERAction.RAISE)], "round": 2}
    ]
    
    for i, case in enumerate(test_cases):
        logger.info(f"\n--- 测试用例 {i+1} ---")
        probs = solver.get_action_probabilities(
            case["player"],
            case["conf_hist"],
            case["act_hist"],
            case["round"]
        )
        advice = solver.format_advice_for_prompt(probs)
        logger.info(f"玩家 {case['player']} (置信度历史={case['conf_hist']}):")
        logger.info(advice)
    
    # 3. 测试手动轨迹更新
    logger.info("\n--- 测试手动轨迹更新 ---")
    test_trajectory = [
        {"player": 0, "infoset": "P0_H[3]_R0_Act:empty", "action": MERAction.RAISE, 
         "legal_actions": list(MERAction), "confidence": 3},
        {"player": 1, "infoset": "P1_H[2]_R0_Act:p0:R", "action": MERAction.CALL,
         "legal_actions": list(MERAction), "confidence": 2},
        {"player": 2, "infoset": "P2_H[1]_R0_Act:p0:R|p1:C", "action": MERAction.FOLD,
         "legal_actions": list(MERAction), "confidence": 1}
    ]
    test_payoffs = [50.0, -20.0, -10.0]
    
    solver.update_from_trajectory(test_trajectory, test_payoffs)
    logger.info(f"轨迹更新完成，当前迭代次数: {solver.total_iterations}")
    
    logger.info("\n" + "="*60)
    logger.info("测试完成")
    logger.info("="*60)
