"""
MERgame.py - MER博弈驱动层
EV = 模型置信度 * CFR策略概率

核心设计理念:
1. MERPlayer: LLM玩家包装器，支持感知/重评估/决策三阶段
2. Referee: 裁判类，支持GT判定和EV判定两种模式
3. MERGameDriver: 博弈驱动器，协调LLM与CFR的交互
4. TrajectoryRecorder: 轨迹记录器，为CFR提供训练数据
"""

import logging
import traceback
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional, Any, Callable, Union, TYPE_CHECKING

# 从 constants.py 导入（单一数据源）
try:
    from .constants import (
        MERAction, ACTION_NAMES, ACTION_SHORT, action_to_name, name_to_action,
        HandStrength, confidence_to_level, level_to_name, level_to_hand_strength,
        DEFAULT_INITIAL_CHIPS, DEFAULT_ANTE, DEFAULT_RAISE_AMOUNT,
        DEFAULT_CONFIDENCE, get_player_short_name
    )
    from .confidence_utils import extract_confidence
    from .prompt import get_discussion_prompt
except ImportError:
    from constants import (
        MERAction, ACTION_NAMES, ACTION_SHORT, action_to_name, name_to_action,
        HandStrength, confidence_to_level, level_to_name, level_to_hand_strength,
        DEFAULT_INITIAL_CHIPS, DEFAULT_ANTE, DEFAULT_RAISE_AMOUNT,
        DEFAULT_CONFIDENCE, get_player_short_name
    )
    from confidence_utils import extract_confidence
    from prompt import get_discussion_prompt

# 类型检查时导入 numpy（避免运行时循环导入）
if TYPE_CHECKING:
    import numpy as np

logger = logging.getLogger("MERGame")


# ==========================================
# 1. 轨迹记录器（常量已从 constants.py 导入）
# ==========================================

@dataclass
class DecisionPoint:
    """
    单个决策点记录
    
    用于CFR的轨迹更新
    
    【OS-MCCFR 关键字段】
    - sigma_a: 策略概率 σ(I, a)，遗憾匹配得到的选该动作概率
    - q_a: 采样概率 q(I, a)，实际采样时选该动作概率（含探索）
    - full_strategy: 完整策略分布（用于遗憾计算）
    """
    player_id: int                    # 玩家ID
    infoset: str                      # 信息集字符串
    action: int                       # 采取的动作
    legal_actions: List[int]          # 合法动作列表
    confidence: int                   # 当时的置信度档位
    perception_prob: float = 1.0      # 感知概率 q_llm
    # === OS-MCCFR 新增字段 ===
    sigma_a: float = 1.0              # 策略概率 σ(I, a)
    q_a: float = 1.0                  # 采样概率 q(I, a)
    full_strategy: Dict[int, float] = field(default_factory=dict)  # 完整策略 σ(I)
    
    def to_dict(self) -> Dict:
        """转换为字典格式（供CFR使用）"""
        return {
            "player": self.player_id,
            "infoset": self.infoset,
            "action": self.action,
            "legal_actions": self.legal_actions,
            "confidence": self.confidence,
            "perception_prob": self.perception_prob,
            # OS-MCCFR 新增
            "sigma_a": self.sigma_a,
            "q_a": self.q_a,
            "full_strategy": self.full_strategy
        }


@dataclass
class GameTrajectory:
    """
    一局完整博弈的轨迹记录
    
    【OS-MCCFR 关键字段】
    - reach_probs: 到达概率 π_i(z) = Π σ_i(I, a)，每个玩家到达终局的策略概率
    - sample_prob: 采样概率 q(z) = Π q(I, a)，采样到该路径的概率
    
    这两个概率用于计算重要性采样权重 W = π_{-i}(z) / q(z)
    
    【零和收益字段】
    - initial_chips: 本局开始时每个玩家的筹码（在_post_ante之前记录）
    - final_payoffs: 最终收益 = final_chips - initial_chips（保证零和）
    """
    sample_name: str                           # 样本名称
    decision_points: List[DecisionPoint] = field(default_factory=list)
    final_payoffs: List[float] = field(default_factory=list)
    winner_id: Optional[int] = None
    winner_prediction: Optional[str] = None
    ground_truth: Optional[str] = None
    # === OS-MCCFR 新增字段 ===
    reach_probs: Dict[int, float] = field(default_factory=dict)  # π_i(z) for each player i
    sample_prob: float = 1.0                   # q(z) = Π q(I, a)
    # === 零和收益新增字段 ===
    initial_chips: List[int] = field(default_factory=list)  # 本局开始时各玩家筹码
    # === 玩家预测保存字段（用于后续 evaluation）===
    player_predictions: List[str] = field(default_factory=list)   # 各玩家的原始预测文本
    player_extracted_labels: List[str] = field(default_factory=list)  # 各玩家的提取标签
    
    def add_decision(self, point: DecisionPoint):
        """添加决策点"""
        self.decision_points.append(point)
    
    def to_cfr_format(self) -> Tuple[List[Dict], List[float]]:
        """转换为CFR更新所需的格式"""
        trajectory = [dp.to_dict() for dp in self.decision_points]
        return trajectory, self.final_payoffs


# ==========================================
# 3. MER玩家类
# ==========================================

class MERPlayer:
    """
    MER博弈玩家：LLM的博弈代理包装器
    
    核心功能:
    1. perceive(): 初始感知，获取置信度（使用熵方法）
    2. re_evaluate(): 重新评估（从文本解析置信度）
    3. decide(): 根据CFR建议做决策
    
    置信度获取方式（关键设计）:
    - 第一回合 (perceive): mode="entropy"，从模型 logits 计算熵转置信度
    - 第二回合+ (re_evaluate): mode="text_parse"，从 LLM 文本回复中解析
    
    设计特点:
    - 不包含任何博弈逻辑（由Driver协调）
    - 置信度历史完整记录（供CFR使用）
    - 支持多模态输入
    """
    
    # LLM callable 返回类型定义
    # 标准返回: (response: str, confidence_or_logits: Union[float, np.ndarray])
    # - 若返回 float，直接作为置信度
    # - 若返回 np.ndarray（logits），由 extract_confidence 计算
    LLM_RETURN_TYPE = Union[Tuple[str, float], Tuple[str, 'np.ndarray'], str]
    
    def __init__(
        self,
        player_id: int,
        name: str,
        llm_callable: Callable[[str, Optional[str], Optional[str], Optional[str]], Any],
        persona: str = "",
        initial_chips: int = None
    ):
        """
        初始化MER玩家
        
        :param player_id: 玩家ID (0, 1, 2)
        :param name: 玩家名称（如 "AffectGPT_A"）
        :param llm_callable: LLM调用接口，签名:
                             (prompt, video_path, audio_path, face_path) -> 
                                 (response, confidence) 或 (response, logits) 或 response
        :param persona: 玩家人设描述
        :param initial_chips: 初始筹码数（None则使用默认值）
        """
        self.player_id = player_id
        self.name = name
        self.llm_callable = llm_callable
        self.persona = persona
        
        # 状态追踪
        self.confidence_history: List[float] = []       # 连续置信度历史
        self.level_history: List[int] = []              # 离散档位历史
        self.prediction: str = ""                       # 当前预测标签
        self.is_folded: bool = False                    # 是否已弃牌
        
        # 【筹码配置】优先使用传入参数，否则使用默认值
        self.initial_chips = initial_chips if initial_chips is not None else DEFAULT_INITIAL_CHIPS
        self.chips: int = self.initial_chips            # 当前筹码数量
        self.sample_start_chips: int = self.initial_chips  # 本局开始时的筹码（用于累积下注计算）
        self.bet_this_round: int = 0                    # 本轮已下注金额
        
        # 当前回合计数（用于判断使用哪种置信度提取方式）
        self._round_counter: int = 0
        
        # 【neutral/空标签处理】强制零置信度标志位
        # 当玩家预测为 neutral 或提取标签为空时，该标志置为 True
        # 此后该玩家在本样本博弈中所有轮次的置信度都强制为 0
        self._force_zero_confidence: bool = False
        
        logger.info(f"MERPlayer {name} (ID={player_id}) 初始化完成, 筹码={self.chips}")
    
    def reset(self):
        """重置玩家状态（新一局开始时调用，不重置筹码）"""
        self.confidence_history = []
        self.level_history = []
        self.prediction = ""
        self.is_folded = False
        self.bet_this_round = 0  # 重置本轮下注
        self._round_counter = 0  # 重置回合计数
        self._force_zero_confidence = False  # 重置强制零置信度标志
        # 【关键修复】记录本局开始时的筹码，用于累积下注计算
        self.sample_start_chips = self.chips
    
    @property
    def current_confidence(self) -> float:
        """获取当前置信度"""
        return self.confidence_history[-1] if self.confidence_history else 0.0
    
    @property
    def current_level(self) -> int:
        """获取当前置信度档位"""
        return self.level_history[-1] if self.level_history else 0
    
    def mark_force_zero_confidence(self):
        """
        【neutral/空标签处理】标记该玩家在本局中需要强制置信度为0
        
        触发条件（由 MERGameDriver 检测）：
        1. 玩家预测包含 "The character's emotional state is neutral"
        2. 提取的标签为空 '[]'
        
        效果：
        - 将已有的置信度历史全部置为 0
        - 后续所有 re_evaluate() 调用也会强制返回 0
        """
        self._force_zero_confidence = True
        # 将已有的置信度历史全部置为 0
        self.confidence_history = [0.0] * len(self.confidence_history)
        self.level_history = [0] * len(self.level_history)
        logger.warning(f"[{self.name}] ⚠️ 强制零置信度已激活（neutral/空标签），历史置信度已清零")
    
    def perceive(
        self,
        prompt: str,
        video_path: str,
        audio_path: Optional[str] = None,
        face_path: Optional[str] = None,
        sample_name: Optional[str] = None,
        subtitle: Optional[str] = None
    ) -> Tuple[str, int]:
        """
        初始感知阶段：观察视频并形成初始判断
        
        置信度获取方式：使用熵方法（mode="entropy"）
        - LLM callable 应返回 (response, logits) 或 (response, confidence)
        - 如果返回 logits，则使用 extract_confidence 计算熵置信度
        
        :param prompt: 感知Prompt（用户问题）
        :param video_path: 视频路径（保留兼容性，实际通过 sample_name 读取）
        :param audio_path: 音频路径（保留兼容性）
        :param face_path: 人脸特征路径（保留兼容性）
        :param sample_name: 样本名称（用于 DataLoader 读取数据）
        :param subtitle: 字幕（用于 prompt 构造）
        :return: (预测结果, 置信度档位)
        """
        try:
            # 调用LLM获取响应（新接口支持 sample_name）
            result = self.llm_callable(
                prompt, video_path, audio_path, face_path,
                sample_name=sample_name, subtitle=subtitle
            )
            
            # 解析返回值
            response, confidence = self._extract_confidence_from_result(
                result, 
                mode="entropy"  # 第一回合使用熵方法
            )
            
            # 记录置信度
            self.confidence_history.append(confidence)
            level = confidence_to_level(confidence)
            self.level_history.append(level)
            
            # 存储预测
            self.prediction = response
            
            # 更新回合计数
            self._round_counter = 1
            
            logger.info(f"[{self.name}] 感知完成 (entropy mode): confidence={confidence:.3f}, level={level_to_name(level)}")
            logger.info(f"  │  💬 预测: {response}")
            
            return response, level
            
        except Exception as e:
            logger.error(f"[{self.name}] 感知失败: {e}")
            traceback.print_exc()
            # 返回默认值（LEVEL_2 对应 0.2，作为 LLM 很少触及的独立档位）
            self.confidence_history.append(DEFAULT_CONFIDENCE)
            self.level_history.append(HandStrength.LEVEL_2.value)
            self._round_counter = 1
            return "", HandStrength.LEVEL_2.value
    
    def re_evaluate(
        self,
        prompt: str,
        video_path: Optional[str] = None,
        audio_path: Optional[str] = None,
        face_path: Optional[str] = None,
        sample_name: Optional[str] = None,
        subtitle: Optional[str] = None
    ) -> int:
        """
        重新评估阶段：看到揭示信息后更新置信度
        
        置信度获取方式：从文本解析（mode="text_parse"）
        - Prompt 应要求 LLM 在回答中明确给出置信度数值
        - 使用 extract_confidence 从文本中解析
        
        :param prompt: 重评估Prompt（包含已揭示的预测信息）
        :param video_path: 视频路径（保留兼容性）
        :param audio_path: 音频路径（保留兼容性）
        :param face_path: 人脸特征路径（保留兼容性）
        :param sample_name: 样本名称
        :param subtitle: 字幕
        :return: 更新后的置信度档位
        """
        try:
            # 调用LLM重新评估（新接口支持 sample_name）
            result = self.llm_callable(
                prompt, video_path, audio_path, face_path,
                sample_name=sample_name, subtitle=subtitle
            )
            
            # 解析返回值，使用文本解析模式
            response, new_confidence = self._extract_confidence_from_result(
                result, 
                mode="text_parse"  # 第二回合及之后使用文本解析
            )
            
            # 【neutral/空标签处理】如果已标记为强制零置信度，覆盖 LLM 返回值
            if self._force_zero_confidence:
                new_confidence = 0.0
                logger.info(f"[{self.name}] 🔒 强制覆盖置信度为 0（neutral/空标签触发）")
            
            # 更新置信度历史
            self.confidence_history.append(new_confidence)
            new_level = confidence_to_level(new_confidence)
            self.level_history.append(new_level)
            
            # 更新回合计数
            self._round_counter += 1
            
            logger.info(f"[{self.name}] 重评估完成 (text_parse mode): confidence={new_confidence:.3f}, level={level_to_name(new_level)}")
            logger.info(f"  │  💬 响应: {response}")
            
            return new_level
            
        except Exception as e:
            logger.error(f"[{self.name}] 重评估失败: {e}")
            # 保持上一个置信度
            if self.confidence_history:
                last_conf = self.confidence_history[-1]
                self.confidence_history.append(last_conf)
                self.level_history.append(self.level_history[-1])
            return self.current_level
    
    def _extract_confidence_from_result(
        self, 
        result: Any, 
        mode: str = "entropy"
    ) -> Tuple[str, float]:
        """
        统一的置信度提取方法
        
        处理 LLM callable 的多种返回格式：
        1. (response, float) - 返回 (文本, 熵置信度)
        2. (response, np.ndarray) - 返回 (文本, logits)
        3. str - 只返回响应
        
        【关键设计】mode 参数决定置信度来源：
        - mode="entropy": 使用 LLM 返回的熵置信度（第一回合感知阶段）
        - mode="text_parse": 从文本中解析置信度（重评估阶段，LLM自己声明的置信度）
        
        :param result: LLM callable 的返回值
        :param mode: 置信度提取模式 ("entropy" 或 "text_parse")
        :return: (response, confidence)
        """
        import numpy as np
        
        # 情况 1: 返回元组 (response, confidence_or_logits)
        if isinstance(result, tuple) and len(result) >= 2:
            response = str(result[0])
            second_element = result[1]
            
            # 【关键修复】当 mode="text_parse" 时，强制从文本解析，忽略返回的熵置信度
            # 这是因为重评估阶段需要使用 LLM 自己声明的置信度，而非熵计算的
            if mode == "text_parse":
                confidence = extract_confidence(response, mode="text_parse")
                return response, confidence
            
            # entropy 模式：使用返回的熵置信度或 logits 计算
            # 1a: 直接是浮点数置信度
            if isinstance(second_element, (int, float)):
                confidence = float(second_element)
                return response, max(0.0, min(1.0, confidence))
            
            # 1b: 是 numpy 数组（logits）
            if isinstance(second_element, np.ndarray):
                confidence = extract_confidence(response, logits=second_element, mode="entropy")
                return response, confidence
        
        # 情况 2: 只返回字符串
        if isinstance(result, str):
            response = result
            # 根据指定的 mode 从文本中提取置信度
            confidence = extract_confidence(response, mode=mode)
            return response, confidence
        
        # 未知格式，使用默认值
        logger.warning(f"Unknown LLM result format: {type(result)}, using default confidence")
        response = str(result) if result else ""
        return response, DEFAULT_CONFIDENCE
    
    def decide(
        self,
        prompt: str,
        cfr_advice: str,
        video_path: Optional[str] = None,
        audio_path: Optional[str] = None,
        face_path: Optional[str] = None,
        sample_name: str = None,
        subtitle: str = None,
        hand_strength: str = None,
        my_chips: int = None,
        pot: int = None,
        current_bet: int = None,
        opponents_chips: List[Tuple[str, int]] = None
    ) -> int:
        """
        决策阶段：根据CFR建议选择动作
        
        :param prompt: Decision prompt
        :param cfr_advice: CFR strategy advice string
        :param video_path: Path to video file
        :param audio_path: Path to audio file
        :param face_path: Path to face image file
        :param sample_name: Sample name for identification
        :param subtitle: Subtitle/transcript text
        :param hand_strength: 手牌强度描述（德州扑克类比）
        :param my_chips: 我的筹码数量
        :param pot: 当前底池大小
        :param current_bet: 当前最高下注额
        :param opponents_chips: 对手筹码列表 [(name, chips), ...]
        :return: Action ID (0=Fold, 1=Check, 2=Call, 3=Raise)
        """
        # 构建筹码信息
        chips_info = ""
        if my_chips is not None and pot is not None:
            chips_info = f"\n\nGame State:"
            chips_info += f"\n- Your chips: {my_chips}"
            chips_info += f"\n- Pot size: {pot}"
            if current_bet is not None:
                call_amount = current_bet - self.bet_this_round
                chips_info += f"\n- To call: {call_amount}"
            if opponents_chips:
                opp_str = ", ".join([f"{name}: {chips}" for name, chips in opponents_chips])
                chips_info += f"\n- Opponents: {opp_str}"
        
        # 手牌强度信息
        hand_info = f"\n\nYour Hand Strength: {hand_strength}" if hand_strength else ""
        
        # 组合prompt
        full_prompt = f"""{prompt}{chips_info}{hand_info}

Strategy Advice:
{cfr_advice}

Available Actions:
- FOLD: Admit defeat and lose your bet
- CHECK: Pass without betting (only if no bet to call)
- CALL: Match the current bet
- RAISE: Increase the bet

**You must reply with exactly "TA: [ACTION]" where ACTION is one of FOLD/CHECK/CALL/RAISE, No any other text**

Your response:"""
        
        # 【调试】打印 prompt（已注释，需要时取消注释）
        # logger.info(f"")
        # logger.info(f"  │  📝 [{self.name}] 决策Prompt:")
        # logger.info(f"  │  {'-'*50}")
        # for line in full_prompt.split('\n'):
        #     logger.info(f"  │  {line}")
        # logger.info(f"  │  {'-'*50}")
        # logger.info(f"")
        
        try:
            result = self.llm_callable(
                full_prompt, video_path, audio_path, face_path,
                sample_name=sample_name, subtitle=subtitle
            )
            
            # 解析返回值
            if isinstance(result, tuple):
                response = result[0]
            else:
                response = result
            
            # 解析动作
            action = self._parse_action(response)
            logger.info(f"[{self.name}] 决策: {self._action_name(action)}")
            logger.info(f"  │  💬 响应: {response}")
            
            if action == 0:  # FOLD
                self.is_folded = True
            
            return action
            
        except Exception as e:
            logger.error(f"[{self.name}] 决策失败: {e}")
            return 1  # 默认CHECK
    
    @staticmethod
    def _parse_action(response: str) -> int:
        """
        解析响应中的动作
        
        【优化】解析优先级：
        1. 优先匹配 "TA: ACTION" 格式（最可靠）
        2. 匹配 "Action: ACTION" 格式
        3. 最后才用关键词匹配（容易误判）
        """
        import re
        text = response.strip()
        text_lower = text.lower()
        
        # 策略1: 匹配 "TA: ACTION" 格式（最可靠）
        ta_match = re.search(r'\bta\s*:\s*(fold|check|call|raise)\b', text_lower)
        if ta_match:
            action_str = ta_match.group(1)
            action_map = {'fold': 0, 'check': 1, 'call': 2, 'raise': 3}
            return action_map.get(action_str, 1)
        
        # 策略2: 匹配 "Action: ACTION" 格式
        action_match = re.search(r'\baction\s*:\s*(fold|check|call|raise)\b', text_lower)
        if action_match:
            action_str = action_match.group(1)
            action_map = {'fold': 0, 'check': 1, 'call': 2, 'raise': 3}
            return action_map.get(action_str, 1)
        
        # 策略3: 匹配 "**ACTION**" 格式（markdown 强调）
        bold_match = re.search(r'\*\*(fold|check|call|raise)\*\*', text_lower)
        if bold_match:
            action_str = bold_match.group(1)
            action_map = {'fold': 0, 'check': 1, 'call': 2, 'raise': 3}
            return action_map.get(action_str, 1)
        
        # 策略4: 查找响应末尾的独立动作词（避免误匹配上下文中的 call 等）
        # 例如 "call your people" 中的 call 不应被匹配
        last_words = text_lower.split()[-5:] if text_lower else []
        for word in reversed(last_words):
            word_clean = re.sub(r'[^a-z]', '', word)
            if word_clean == 'fold':
                return 0
            elif word_clean == 'raise':
                return 3
            elif word_clean == 'call':
                return 2
            elif word_clean == 'check':
                return 1
        
        # 默认 CHECK
        return 1
    
    @staticmethod
    def _action_name(action: int) -> str:
        """动作ID到名称（使用 constants.py 的函数）"""
        return action_to_name(action)


# ==========================================
# 2. 裁判类
# ==========================================

class Referee:
    """
    裁判类：负责判定获胜者
    
    支持两种模式:
    1. GT模式（训练态）：使用Ground Truth判定
    2. EV模式（推理态）：使用期望值方法判定
    """
    
    def __init__(
        self,
        label_extractor: Optional[Callable[[str], str]] = None,
        score_calculator: Optional[Callable[[str, str], float]] = None
    ):
        """
        初始化裁判
        
        :param label_extractor: 标签提取函数 (response -> label)
        :param score_calculator: 评分函数 (pred, gt) -> score
        """
        self.label_extractor = label_extractor or self._default_extractor
        self.score_calculator = score_calculator or self._default_scorer
    
    def judge_by_gt(
        self,
        players: List[MERPlayer],
        ground_truth: str
    ) -> Tuple[List[int], List[float]]:
        """
        GT判定模式：使用Ground Truth评分
        
        【修改】所有玩家都参与得分计算（包括弃牌玩家），
        弃牌玩家只是不参与胜者判定。
        
        :param players: 玩家列表
        :param ground_truth: 真实标签
        :return: (获胜者ID列表, 各玩家得分列表)
        """
        scores = []
        active_players = []  # 未弃牌的玩家（参与胜者判定）
        
        for p in players:
            # 【修改】所有玩家都计算得分
            extracted = self.label_extractor(p.prediction)
            score = self.score_calculator(extracted, ground_truth)
            scores.append(score)
            
            if p.is_folded:
                # 弃牌玩家：计算得分但不参与胜者判定
                logger.info(f"  │  🏷️ [{p.name}] [已弃牌] 提取标签: '{extracted}' vs GT: '{ground_truth}' → 得分: {score:.2f}")
            else:
                # 未弃牌玩家：计算得分并参与胜者判定
                active_players.append(p.player_id)
                logger.info(f"  │  🏷️ [{p.name}] 提取标签: '{extracted}' vs GT: '{ground_truth}' → 得分: {score:.2f}")
        
        if not active_players:
            # 所有玩家都弃牌，返回空获胜者列表
            logger.warning("[Referee] 所有玩家都已弃牌，无获胜者")
            return [], scores
        
        # 在未弃牌玩家中找出最高分
        active_scores = [(pid, scores[pid]) for pid in active_players]
        max_score = max(s for _, s in active_scores)
        winners = [pid for pid, s in active_scores if abs(s - max_score) < 1e-6]
        
        logger.info(f"[Referee] GT判定获胜者: {winners}, 最高分: {max_score:.4f}")
        return winners, scores
    
    def judge_by_ev(
        self,
        players: List[MERPlayer],
        cfr_solver,
        action_history: List[Tuple[int, int]],
        discussion_callback: Optional[Callable] = None,
        extracted_labels: List[str] = None
    ) -> Tuple[List[int], List[float]]:
        """
        累积下注判定模式：使用玩家本局投入的总筹码判定胜者
        
        设计思路（基于博弈论直觉）：
        1. 在 CFR 训练收敛后，下注越多的玩家 = 对自己预测越有信心
        2. 累积下注 = initial_chips - chips（本局总共投入了多少筹码）
        3. 平局处理：
           - 【推理态新增】若提供 discussion_callback，启动讨论回合投票
           - 若无回调或投票失败，回退到初始熵置信度
        
        :param players: 玩家列表
        :param cfr_solver: CFR求解器实例（此方法中不再使用，保留参数兼容性）
        :param action_history: 动作历史（此方法中不再使用，保留参数兼容性）
        :param discussion_callback: 【新增】讨论回合回调函数，签名: (tied_player_ids, extracted_labels) -> winner_id
        :param extracted_labels: 【新增】各玩家的提取标签列表，用于讨论回合展示
        :return: (获胜者ID列表, 各玩家累积下注列表)
        """
        bet_amounts = []  # 累积下注金额
        active_players = []
        
        for p in players:
            if p.is_folded:
                bet_amounts.append(-999.0)  # 弃牌者不参与判定
            else:
                active_players.append(p.player_id)
                # 【关键修复】累积下注 = 本局开始筹码 - 当前筹码
                total_bet = p.sample_start_chips - p.chips
                bet_amounts.append(float(total_bet))
                logger.info(f"[Referee] {p.name}: 累积下注={total_bet}, 初始熵置信度={p.confidence_history[0] if p.confidence_history else 'N/A':.4f}")
        
        if not active_players:
            return [], bet_amounts
        
        # 找出最高累积下注
        active_bets = [(pid, bet_amounts[pid]) for pid in active_players]
        max_bet = max(bet for _, bet in active_bets)
        winners = [pid for pid, bet in active_bets if abs(bet - max_bet) < 1e-6]
        
        # ========== 平局处理 ==========
        if len(winners) > 1:
            logger.info(f"[Referee] 检测到平局: {[players[w].name for w in winners]}")
            
            # 【推理态新增】尝试通过讨论回合投票破局
            if discussion_callback is not None and extracted_labels is not None:
                try:
                    voted_winner = discussion_callback(winners, extracted_labels)
                    if voted_winner is not None and voted_winner in winners:
                        winners = [voted_winner]
                        logger.info(f"[Referee] 讨论回合投票成功，获胜者: {players[voted_winner].name}")
                    else:
                        logger.warning(f"[Referee] 讨论回合投票无效 (返回: {voted_winner})，回退到初始置信度")
                        winners = self._fallback_by_confidence(players, winners)
                except Exception as e:
                    logger.warning(f"[Referee] 讨论回合执行失败 ({e})，回退到初始置信度")
                    winners = self._fallback_by_confidence(players, winners)
            else:
                # 无讨论回调，使用初始置信度打破平局
                winners = self._fallback_by_confidence(players, winners)
        
        logger.info(f"[Referee] 累积下注判定获胜者: {winners}, 最高下注: {max_bet:.0f}")
        return winners, bet_amounts
    
    def _fallback_by_confidence(self, players: List[MERPlayer], winners: List[int]) -> List[int]:
        """
        回退方案：使用初始熵置信度打破平局
        
        :param players: 玩家列表
        :param winners: 平局玩家ID列表
        :return: 单一获胜者列表
        """
        winner_players = [players[pid] for pid in winners]
        best = max(winner_players, key=lambda p: p.confidence_history[0] if p.confidence_history else 0.0)
        logger.info(f"[Referee] 回退方案: 选择初始熵置信度最高者 {best.name} (置信度={best.confidence_history[0]:.4f})")
        return [best.player_id]
        
        logger.info(f"[Referee] 累积下注判定获胜者: {winners}, 最高下注: {max_bet:.0f}")
        return winners, bet_amounts
    
    @staticmethod
    def _default_extractor(response: str) -> str:
        """默认标签提取：取第一行非空内容"""
        lines = response.strip().split('\n')
        for line in lines:
            if line.strip():
                return line.strip()
        return response
    
    @staticmethod
    def _default_scorer(pred: str, gt: str) -> float:
        """默认评分：精确匹配"""
        return 1.0 if pred.lower().strip() == gt.lower().strip() else 0.0


# ==========================================
# 5. 博弈驱动器
# ==========================================

class MERGameDriver:
    """
    MER博弈驱动器：协调LLM与CFR的交互
    
    核心职责:
    1. 管理博弈流程：感知 -> 决策 -> 揭示 -> 重评估 -> 结算
    2. 记录轨迹供CFR学习
    3. 支持训练态和推理态两种模式
    
    设计特点:
    - 不包含数据集加载逻辑（由外部传入）
    - 完全解耦的架构
    """
    
    def __init__(
        self,
        players: List[MERPlayer],
        cfr_solver,
        referee: Referee,
        mode: str = "training",
        blind_mode: bool = False
    ):
        """
        初始化博弈驱动器
        
        :param players: 玩家列表（已初始化的MERPlayer实例）
        :param cfr_solver: CFR求解器实例
        :param referee: 裁判实例
        :param mode: 运行模式 ("training" 或 "inference")
        :param blind_mode: 是否启用盲式流程（仅首轮感知，其后不重评估）
        """
        self.players = players
        self.cfr_solver = cfr_solver
        self.referee = referee
        self.mode = mode
        self.blind_mode = blind_mode
        
        # 当前博弈状态
        self.action_history: List[Tuple[int, int]] = []
        self.revealed_predictions: List[Tuple[int, str]] = []  # (player_id, prediction)
        
        # 【筹码与下注配置】
        self.pot: int = 0                                          # 底池
        self.current_bet: int = 0                                  # 当前轮最高下注额
        self.ante: int = DEFAULT_ANTE                             # 强制底注（所有玩家平等投入）
        self.raise_amount: int = DEFAULT_RAISE_AMOUNT             # 加注金额
        self.dealer_pos: int = 0                                   # 庄家位置（用于决策顺序）
        
        # 系统预测存储（推理态）
        self.system_predictions: Dict[str, str] = {}
        
        # 【推理态配置】是否跳过评分计算（用于处理 MELD 等整数标签数据集）
        self.skip_score_calculation: bool = False
        
        # 【推理态讨论回合】当前样本的多模态路径（供讨论回合使用）
        self._current_video_path: Optional[str] = None
        self._current_audio_path: Optional[str] = None
        self._current_face_path: Optional[str] = None
        self._current_sample_name: Optional[str] = None
        self._current_subtitle: Optional[str] = None
        
        logger.info(f"MERGameDriver初始化: mode={mode}, blind_mode={blind_mode}, num_players={len(players)}, ante={self.ante}")
    
    def configure_betting(self, ante: int = None, 
                          raise_amount: int = None, initial_chips: int = None):
        """
        配置下注参数（从 yaml 读取后调用）
        
        :param ante: 强制底注金额（所有玩家平等投入）
        :param raise_amount: 加注金额
        :param initial_chips: 每位玩家初始筹码
        """
        if ante is not None:
            self.ante = ante
        if raise_amount is not None:
            self.raise_amount = raise_amount
        if initial_chips is not None:
            for p in self.players:
                p.chips = initial_chips
                p.initial_chips = initial_chips
        logger.info(f"博弈配置更新: Ante={self.ante}, "
                    f"Raise={self.raise_amount}, InitialChips={initial_chips}")
    
    def reset_round(self):
        """重置一轮博弈状态"""
        self.action_history = []
        self.revealed_predictions = []
        self.pot = 0
        self.current_bet = 0
        for p in self.players:
            p.reset()
    
    def _check_neutral_predictions(self):
        """
        【neutral/空标签处理】检测玩家是否预测为 neutral 或空标签
        
        在初始感知阶段完成后调用，检测每个玩家的预测结果：
        1. 如果预测包含 "The character's emotional state is neutral"
        2. 或提取的标签为空 '[]'
        
        则标记该玩家在整个样本博弈中强制零置信度。
        
        注意：训练态和推理态都会执行此检测。
        """
        for p in self.players:
            # 条件1: 检测 "neutral" 关键句
            if "The character's emotional state is neutral" in p.prediction:
                p.mark_force_zero_confidence()
                logger.warning(f"  │  ⚠️ [{p.name}] 预测为 neutral，强制置信度为 0")
                continue
            
            # 条件2: 检测空标签（需要 label_extractor）
            if self.referee.label_extractor is not None:
                try:
                    extracted = self.referee.label_extractor(p.prediction)
                    if self._is_empty_label(extracted):
                        p.mark_force_zero_confidence()
                        logger.warning(f"  │  ⚠️ [{p.name}] 提取标签为空 '{extracted}'，强制置信度为 0")
                except Exception as e:
                    logger.error(f"  │  ❌ [{p.name}] 标签提取失败: {e}")
    
    def _is_empty_label(self, extracted: str) -> bool:
        """
        检测提取的标签是否为空
        
        :param extracted: 提取的标签字符串
        :return: 是否为空标签
        """
        if not extracted:
            return True
        clean = extracted.strip()
        return clean in ("[]", "['']", '[""]', "", "[ ]")
    
    # ==========================================
    # 讨论回合（推理态平局时使用）
    # ==========================================
    
    def _run_discussion_round(
        self,
        tied_player_ids: List[int],
        extracted_labels: List[str],
        video_path: str = None,
        audio_path: str = None,
        face_path: str = None,
        sample_name: str = None,
        subtitle: str = None
    ) -> Optional[int]:
        """
        【推理态专用】讨论回合：平局时让所有玩家投票选出最佳预测
        
        设计思路：
        1. 构建平局玩家的预测列表作为选项
        2. 向每位玩家（包括旁观者）展示视频 + 选项列表
        3. 收集所有玩家的投票，采用简单多数决原则
        4. 若投票仍无法决出胜者，返回 None 由调用方回退处理
        
        :param tied_player_ids: 平局玩家的 ID 列表
        :param extracted_labels: 各玩家的提取标签列表（索引对应 player_id）
        :param video_path: 视频路径（用于多模态推理）
        :param audio_path: 音频路径
        :param face_path: 人脸路径
        :param sample_name: 样本名称（用于日志）
        :param subtitle: 字幕
        :return: 投票胜出的玩家 ID，若无法决出则返回 None
        """
        logger.info(f"  ├─ [讨论回合] 启动投票破局 | 平局玩家: {[self.players[pid].name for pid in tied_player_ids]}")
        
        # 【Step 1】构建选项列表：(玩家名, 提取标签)
        tied_predictions = []
        for pid in tied_player_ids:
            player_name = self.players[pid].name
            labels = extracted_labels[pid] if pid < len(extracted_labels) else "[]"
            tied_predictions.append((player_name, labels))
        
        # 【Step 2】生成讨论回合 Prompt
        discussion_prompt = get_discussion_prompt(tied_predictions)
        
        # 【Step 3】收集所有玩家的投票
        votes = {}  # player_id -> voted_option (1-based)
        for p in self.players:
            try:
                # 调用 LLM（复用感知阶段的接口，传入视频等多模态数据）
                result = p.llm_callable(
                    discussion_prompt,
                    video_path,
                    audio_path,
                    face_path,
                    sample_name=sample_name,
                    subtitle=subtitle
                )
                
                # 解析响应：提取数字
                response = result[0] if isinstance(result, tuple) else result
                vote = self._parse_vote(response, len(tied_player_ids))
                votes[p.player_id] = vote
                
                if vote is not None:
                    voted_name = tied_predictions[vote - 1][0] if 1 <= vote <= len(tied_predictions) else "无效"
                    logger.info(f"  │  🗳️ [{p.name}] 投票: Option {vote} ({voted_name})")
                else:
                    logger.warning(f"  │  🗳️ [{p.name}] 投票无效: '{response[:50]}...'")
                    
            except Exception as e:
                logger.warning(f"  │  🗳️ [{p.name}] 投票失败: {e}")
                votes[p.player_id] = None
        
        # 【Step 4】统计投票结果
        vote_counts = {}  # option -> count
        for vote in votes.values():
            if vote is not None:
                vote_counts[vote] = vote_counts.get(vote, 0) + 1
        
        if not vote_counts:
            logger.warning(f"  │  ⚠️ 讨论回合: 无有效投票")
            return None
        
        # 找出最高票选项
        max_votes = max(vote_counts.values())
        top_options = [opt for opt, cnt in vote_counts.items() if cnt == max_votes]
        
        logger.info(f"  │  📊 投票统计: {vote_counts} (最高票: {max_votes})")
        
        # 如果仍有多个选项票数相同，无法决出胜者
        if len(top_options) > 1:
            logger.warning(f"  │  ⚠️ 讨论回合: 投票仍有平局 {top_options}")
            return None
        
        # 转换选项号 (1-based) 为玩家 ID
        winning_option = top_options[0]
        winner_id = tied_player_ids[winning_option - 1]
        
        logger.info(f"  │  ✓ 讨论回合结果: {self.players[winner_id].name} 获得最高票 ({max_votes} 票)")
        return winner_id
    
    def _parse_vote(self, response: str, num_options: int) -> Optional[int]:
        """
        解析投票响应，提取选项编号
        
        :param response: LLM 响应文本
        :param num_options: 有效选项数量
        :return: 选项编号 (1-based)，无效则返回 None
        """
        import re
        # 尝试匹配数字（支持 "1", "Option 1", "选项1" 等格式）
        match = re.search(r'(\d+)', response.strip())
        if match:
            vote = int(match.group(1))
            if 1 <= vote <= num_options:
                return vote
        return None

    def _check_zero_confidence_auto_fold(self) -> List[int]:
        """
        【推理态特有】零置信度玩家自动弃牌
        
        在首轮决策前调用，检测每个玩家的初始置信度等级：
        - 如果置信度等级为 0（EXTREMELY_UNCERTAIN, confidence ≤ 0.10）
        - 则该玩家自动弃牌，后续阶段不再参与决策
        
        设计理由：
        - 置信度等级 0 表示极度不确定，说明模型对该样本完全没有把握
        - 在推理态中，这类预测质量极低，应主动放弃参与博弈
        - 避免低质量预测对最终结果产生干扰
        
        :return: 自动弃牌的玩家ID列表
        """
        auto_folded_players = []
        
        for p in self.players:
            # 检查初始置信度等级是否为 0
            if p.current_level == 0:
                # 标记弃牌
                p.is_folded = True
                auto_folded_players.append(p.player_id)
                logger.info(f"  │  🚫 [{p.name}] 置信度等级为 0 (EXTREMELY_UNCERTAIN)，自动弃牌")
        
        return auto_folded_players
    
    def _post_ante(self):
        """
        执行强制底注(Ante)阶段
        
        【设计原则】
        所有玩家投入相同金额的底注，保证：
        1. 对称性：所有玩家初始筹码状态相同
        2. 零和性：总筹码守恒
        3. CFR正确性：信息集不因位置不同而产生偏差
        
        与盲注的区别：
        - 盲注：只有2人强制下注，金额不同，破坏对称性
        - 底注：所有人强制下注，金额相同，保持对称性
        """
        ante_amounts = []
        for p in self.players:
            # 投入底注（不超过剩余筹码）
            amount = min(self.ante, p.chips)
            p.chips -= amount
            p.bet_this_round = 0  # 底注不算本轮下注，current_bet从0开始
            self.pot += amount
            ante_amounts.append(amount)
        
        # 底注不设置current_bet，所有人从0开始公平竞争
        self.current_bet = 0
        
        ante_info = ", ".join([f"{p.name}:{ante_amounts[i]}" for i, p in enumerate(self.players)])
        logger.info(f"  ├─ [底注] {ante_info} | 底池:{self.pot}")
    
    def _execute_bet(self, player: MERPlayer, action: int) -> int:
        """
        执行下注动作，更新筹码和底池
        
        :param player: 玩家
        :param action: 动作 (0=FOLD, 1=CHECK, 2=CALL, 3=RAISE)
        :return: 实际下注金额
        """
        bet_amount = 0
        
        if action == 0:  # FOLD
            player.is_folded = True
            
        elif action == 1:  # CHECK
            # 过牌不需要下注，但只有在 current_bet == bet_this_round 时才合法
            pass
            
        elif action == 2:  # CALL
            # 跟注：补齐到当前最高注额
            call_amount = self.current_bet - player.bet_this_round
            call_amount = min(call_amount, player.chips)  # 不能超过剩余筹码
            player.chips -= call_amount
            player.bet_this_round += call_amount
            self.pot += call_amount
            bet_amount = call_amount
            
        elif action == 3:  # RAISE
            # 加注：先跟注，再加注
            call_amount = self.current_bet - player.bet_this_round
            raise_total = call_amount + self.raise_amount
            raise_total = min(raise_total, player.chips)  # 不能超过剩余筹码
            
            player.chips -= raise_total
            player.bet_this_round += raise_total
            self.pot += raise_total
            self.current_bet = player.bet_this_round  # 更新最高注额
            bet_amount = raise_total
        
        return bet_amount
    
    def _distribute_pot(self, winners: List[int]):
        """
        分配底池给获胜者
        
        :param winners: 获胜者ID列表
        """
        if not winners:
            # 无获胜者，底池保留（不太可能发生）
            logger.warning(f"  │  ⚠️ 无获胜者，底池 {self.pot} 保留")
            return
        
        # 平分底池
        share = self.pot // len(winners)
        remainder = self.pot % len(winners)  # 余数给第一个获胜者
        
        for i, winner_id in enumerate(winners):
            amount = share + (remainder if i == 0 else 0)
            self.players[winner_id].chips += amount
            logger.info(f"  │  💰 [{self.players[winner_id].name}] 获得 {amount} 筹码, 总筹码: {self.players[winner_id].chips}")
        
        self.pot = 0
    
    def play_round(
        self,
        sample_name: str,
        ground_truth: str,
        video_path: str,
        audio_path: Optional[str] = None,
        face_path: Optional[str] = None,
        subtitle: Optional[str] = None,
        perception_prompt_fn: Callable[[str], str] = lambda x: x,
        reevaluation_prompt_fn: Callable = None,
        decision_prompt_fn: Callable = None
    ) -> GameTrajectory:
        """
        运行一局完整博弈
        
        :param sample_name: 样本名称（用于 DataLoader 读取数据）
        :param ground_truth: 真实标签（训练态使用）
        :param video_path: 视频路径（保留兼容性）
        :param audio_path: 音频路径（保留兼容性）
        :param face_path: 人脸特征路径（保留兼容性）
        :param subtitle: 字幕（用于 prompt 构造）
        :param perception_prompt_fn: 感知Prompt生成函数
        :param reevaluation_prompt_fn: 重评估Prompt生成函数 (revealed, player_id, prediction, confidence, action_history, confidence_history, player_names)
        :param decision_prompt_fn: 决策Prompt生成函数 (context, action_history, player_names, current_round)
        :return: 博弈轨迹记录
        """
        import time
        game_start_time = time.time()
        
        # 设置默认 prompt 函数
        if reevaluation_prompt_fn is None:
            reevaluation_prompt_fn = lambda r, p, pred, conf, ah, ch, pn: ""
        if decision_prompt_fn is None:
            decision_prompt_fn = lambda ctx, ah, pn, cr: ctx
        
        # 创建玩家名称映射（用于统一命名）
        player_names = {p.player_id: p.name for p in self.players}
        
        logger.info(f"\n{'═'*60}")
        logger.info(f"  📍 新一局博弈开始 | 样本: {sample_name}")
        logger.info(f"{'═'*60}")
        self.reset_round()
        
        # 显示各玩家筹码
        chips_info = ", ".join([f"{p.name}:{p.chips}" for p in self.players])
        logger.info(f"  ├─ [Setup] 庄家:{self.players[self.dealer_pos].name} | 筹码: {chips_info}")
        
        trajectory = GameTrajectory(sample_name=sample_name, ground_truth=ground_truth)
        
        # 【推理态讨论回合】保存当前样本的多模态路径
        self._current_video_path = video_path
        self._current_audio_path = audio_path
        self._current_face_path = face_path
        self._current_sample_name = sample_name
        self._current_subtitle = subtitle
        
        # 【零和收益】记录本局开始时的筹码状态（用于计算最终收益）
        trajectory.initial_chips = [p.chips for p in self.players]
        
        # ========== Phase 1: 初始感知 (Preflop) ==========
        logger.info("  ├─ [阶段1] 初始感知")
        perception_prompt = perception_prompt_fn(sample_name)
        
        for p in self.players:
            p.perceive(perception_prompt, video_path, audio_path, face_path,
                      sample_name=sample_name, subtitle=subtitle)
        
        # ========== Phase 1.5: 检测 neutral/空标签 ==========
        # 如果玩家预测为 neutral 或提取标签为空，标记该玩家强制零置信度
        self._check_neutral_predictions()
        
        # ========== Phase 2: 强制底注 ==========
        self._post_ante()
        
        # ========== Phase 2.5: 零置信度玩家自动弃牌（推理态特有）==========
        # 如果玩家初始置信度等级为 0，则自动弃牌，不参与后续博弈
        if self.mode == "inference":
            auto_folded = self._check_zero_confidence_auto_fold()
            if auto_folded:
                # 检查是否只剩一个活跃玩家
                active_players = [p for p in self.players if not p.is_folded]
                if len(active_players) <= 1:
                    logger.info(f"  │  ⚠️ 仅剩 {len(active_players)} 名活跃玩家，直接进入结算")
                    # 直接进入结算阶段
                    logger.info("  ├─ [阶段5] 结算（提前结束）")
                    trajectory = self._settle_round(trajectory, ground_truth)
                    
                    # 庄家轮换
                    old_dealer = self.players[self.dealer_pos].name
                    self.dealer_pos = (self.dealer_pos + 1) % len(self.players)
                    new_dealer = self.players[self.dealer_pos].name
                    logger.info(f"  ├─ [阶段7] 庄家轮换: {old_dealer} → {new_dealer}")
                    
                    # 本局结果汇总
                    game_elapsed = time.time() - game_start_time
                    logger.info("  └─────────────────────────────────────────────────────")
                    winner_name = self.players[trajectory.winner_id].name if trajectory.winner_id is not None else "无"
                    chips_info = ", ".join([f"{p.name}:{p.chips}" for p in self.players])
                    logger.info(f"  🏆 本局获胜: {winner_name} | ⏱ 耗时: {game_elapsed:.2f}s")
                    logger.info(f"  💰 筹码状态: {chips_info}")
                    logger.info(f"{'═'*60}\n")
                    return trajectory
        
        # ========== Phase 3: First Decision Round ==========
        logger.info("  ├─ [阶段3] 首轮决策")
        self._execute_decision_round(
            trajectory, decision_prompt_fn, video_path, audio_path, face_path,
            round_idx=0, sample_name=sample_name, subtitle=subtitle,
            player_names=player_names
        )
        
        # ========== Phase 4: 后续轮 ==========
        num_players = len(self.players)
        if self.blind_mode:
            logger.info("  ├─ [阶段4] 盲式后续轮（无揭示、无重评估，仅 solver 探索动作）")
            for reveal_idx in range(num_players):
                active_count = sum(1 for p in self.players if not p.is_folded)
                if active_count <= 1:
                    logger.info(f"  │  ⚠️ 提前结束: 仅剩 {active_count} 名活跃玩家")
                    break
                self._execute_decision_round(
                    trajectory, decision_prompt_fn, video_path, audio_path, face_path,
                    round_idx=reveal_idx + 1, sample_name=sample_name, subtitle=subtitle,
                    player_names=player_names
                )
        else:
            # 揭示顺序：从庄家开始
            # 决策顺序：被揭示者最后决策
            for reveal_idx in range(num_players):
                # 检查是否提前结束
                active_count = sum(1 for p in self.players if not p.is_folded)
                if active_count <= 1:
                    logger.info(f"  │  ⚠️ 提前结束: 仅剩 {active_count} 名活跃玩家")
                    break
                
                # 计算本轮揭示者位置：从庄家开始轮换
                revealer_pos = (self.dealer_pos + reveal_idx) % num_players
                revealer = self.players[revealer_pos]
                
                logger.info(f"  ├─ [阶段4.{reveal_idx+1}] 揭示轮 ({reveal_idx+1}/{num_players})")
                
                # 揭示该玩家的预测
                self.revealed_predictions.append((revealer.player_id, revealer.prediction))
                logger.info(f"  │     揭示: {revealer.name} → {revealer.prediction}")
                
                # 所有活跃玩家重新评估
                for p in self.players:
                    if not p.is_folded:
                        reeval_prompt = reevaluation_prompt_fn(
                            self.revealed_predictions,
                            p.player_id,
                            p.prediction,
                            p.current_confidence,  # 当前置信度
                            self.action_history,   # 动作历史
                            p.confidence_history,  # 置信度历史
                            player_names           # 玩家名称映射
                        )
                        p.re_evaluate(reeval_prompt, video_path, audio_path, face_path,
                                     sample_name=sample_name, subtitle=subtitle)
                
                # 决策轮：被揭示者最后决策
                self._execute_decision_round_with_revealer_last(
                    trajectory, decision_prompt_fn, video_path, audio_path, face_path,
                    round_idx=reveal_idx+1, revealer_pos=revealer_pos,
                    sample_name=sample_name, subtitle=subtitle,
                    player_names=player_names
                )
        
        # ========== Phase 5: 结算 ==========
        logger.info("  ├─ [阶段5] 结算")
        trajectory = self._settle_round(trajectory, ground_truth)
        
        # ========== Phase 6: CFR更新（仅训练态） ==========
        if self.mode == "training":
            logger.info("  ├─ [阶段6] CFR策略更新 (OS-MCCFR)")
            cfr_trajectory, cfr_payoffs = trajectory.to_cfr_format()
            
            # 【OS-MCCFR】执行标准更新，传入概率追踪
            self.cfr_solver.update_from_trajectory(
                cfr_trajectory, 
                cfr_payoffs,
                reach_probs=trajectory.reach_probs,
                sample_prob=trajectory.sample_prob
            )
            
            # 【调试】显示信息集数量和本次轨迹信息
            logger.debug(f"  │  📊 CFR调试: 总信息集数量={len(self.cfr_solver.infoset_records)}, 本局轨迹长度={len(cfr_trajectory)}")
            
            # 显示每个决策点的信息集 Key 和策略变化
            logger.info(f"  │  📊 信息集更新 (共 {len(self.cfr_solver.infoset_records)} 个):")
            for i, step in enumerate(cfr_trajectory):
                player_id = step["player"]
                infoset = step["infoset"]
                action_taken = action_to_name(step["action"], short=True)
                
                # 直接查询该信息集的当前策略
                if infoset in self.cfr_solver.infoset_records:
                    record = self.cfr_solver.infoset_records[infoset]
                    probs = record.get_current_strategy(self.cfr_solver.epsilon)
                    visits = record.visit_count
                    # 格式化策略（高亮最优动作）
                    best_action = max(probs, key=probs.get)
                    strat_parts = []
                    for a in [0, 1, 2, 3]:  # F K C R
                        p = probs.get(a, 0) * 100
                        name = action_to_name(a, short=True)
                        if a == best_action:
                            strat_parts.append(f"[{name}:{p:.0f}%]")  # 高亮最优
                        else:
                            strat_parts.append(f"{name}:{p:.0f}%")
                    strat_str = " ".join(strat_parts)
                    # 显示完整信息集 key
                    logger.info(f"  │  {i+1}. {infoset}")
                    logger.info(f"  │     → 选:{action_taken} | 策略:{strat_str} | 访问:{visits}次")
                else:
                    logger.info(f"  │  {i+1}. {infoset} (新建)")
        else:
            logger.info("  ├─ [阶段6] 推理态 - CFR冻结")
        
        # ========== Phase 7: 庄家轮换 ==========
        old_dealer = self.players[self.dealer_pos].name
        self.dealer_pos = (self.dealer_pos + 1) % len(self.players)
        new_dealer = self.players[self.dealer_pos].name
        logger.info(f"  ├─ [阶段7] 庄家轮换: {old_dealer} → {new_dealer}")
        
        # ========== 本局结果汇总 ==========
        game_elapsed = time.time() - game_start_time
        logger.info("  └─────────────────────────────────────────────────────")
        winner_name = self.players[trajectory.winner_id].name if trajectory.winner_id is not None else "无"
        chips_info = ", ".join([f"{p.name}:{p.chips}" for p in self.players])
        logger.info(f"  🏆 本局获胜: {winner_name} | ⏱ 耗时: {game_elapsed:.2f}s")
        logger.info(f"  💰 筹码状态: {chips_info}")
        logger.info(f"{'═'*60}\n")
        return trajectory
    
    def _execute_decision_round(
        self,
        trajectory: GameTrajectory,
        prompt_fn: Callable,
        video_path: str,
        audio_path: Optional[str],
        face_path: Optional[str],
        round_idx: int,
        sample_name: str = None,
        subtitle: str = None,
        player_names: Dict[int, str] = None
    ):
        """
        执行一轮决策
        
        决策顺序（德州扑克规则）：
        - 首轮 (round_idx=0, Preflop): 从大盲位下一位(UTG)开始，顺时针到大盲位结束
        - 后续轮: 从小盲位开始，顺时针到庄家结束
        
        【OS-MCCFR 概率追踪】
        对于每个决策点，追踪:
        - σ(a): 策略概率（遗憾匹配）
        - q(a): 采样概率（ε-贪婪）
        - trajectory 累积: π_i(z) *= σ(a), q(z) *= q(a)
        
        【关键设计】动作来源：
        - 训练模式 (training): 使用 CFR 策略采样决策（标准 OS-MCCFR）
        - 推理模式 (inference): 使用 LLM 决策，CFR 只提供建议
        """
        num_players = len(self.players)
        
        # 计算决策起始位置
        # 在 Ante 模式下，所有轮次均从庄家左手边第一位玩家开始
        start_pos = (self.dealer_pos + 1) % num_players
        
        # 按顺序遍历玩家
        for i in range(num_players):
            player_idx = (start_pos + i) % num_players
            p = self.players[player_idx]
            
            if p.is_folded:
                continue
            
            # 【OS-MCCFR】使用 sample_action_with_probs 获取策略和采样概率
            cfr_sampled_action, sigma_a, q_a, full_strategy = self.cfr_solver.sample_action_with_probs(
                player_id=p.player_id,
                confidence_history=p.level_history,
                action_history=self.action_history,
                current_round=round_idx
            )
            
            # 格式化 CFR 建议
            advice = self.cfr_solver.format_advice_for_prompt(full_strategy)
            
            # 显示CFR建议（顺序：K C R F，FOLD放最后避免LLM偏好选择首位）
            probs_str = " ".join([f"{['F','K','C','R'][a]}:{full_strategy.get(a,0)*100:.2f}%" for a in [1,2,3,0]])
            logger.info(f"  │  📈 [{p.name}] CFR建议: {probs_str}")
            
            # 【关键分支】根据模式决定动作来源
            if self.mode == "training":
                # ========== 训练模式：使用 CFR 当前策略采样（标准 OS-MCCFR）==========
                # 动作由 CFR 的 ε-贪婪策略采样决定
                action = cfr_sampled_action
                logger.info(f"  │  🎯 [{p.name}] 训练模式: CFR采样动作 {action_to_name(action)}")
                
            else:
                # ========== 推理模式：使用 CFR 平均策略（收敛后的纳什均衡近似）==========
                # 【重要】CFR 理论：推理时应使用平均策略，而不是当前策略
                avg_strategy = self.cfr_solver.get_average_strategy(
                    player_id=p.player_id,
                    confidence_history=p.level_history,
                    action_history=self.action_history,
                    current_round=round_idx
                )
                # 选择平均策略中概率最大的动作
                action = max(avg_strategy, key=avg_strategy.get)
                # 更新 full_strategy 用于日志显示
                full_strategy = avg_strategy
                probs_str = " ".join([f"{['F','K','C','R'][a]}:{avg_strategy.get(a,0)*100:.2f}%" for a in [1,2,3,0]])
                logger.info(f"  │  📊 [{p.name}] 平均策略: {probs_str}")
                logger.info(f"  │  🎯 [{p.name}] 推理模式: 平均策略最优动作 {action_to_name(action)}")
                
                # # ========== [已禁用] LLM 决策代码 ==========
                # # 如需恢复 LLM 决策，取消以下注释
                # # 生成决策Prompt（传入历史记录）
                # decision_prompt = prompt_fn(
                #     f"Round {round_idx}, Player {p.name}",
                #     self.action_history,
                #     player_names or {},
                #     round_idx
                # )
                # 
                # # 获取手牌强度（德州扑克类比）
                # hand_strength = level_to_hand_strength(p.current_level)
                # 
                # # 构建对手筹码列表（使用简称）
                # opponents_chips = [
                #     (get_player_short_name(player_name=opp.name), opp.chips) 
                #     for opp in self.players 
                #     if opp.player_id != p.player_id and not opp.is_folded
                # ]
                # 
                # # LLM 决策
                # action = p.decide(
                #     decision_prompt, advice, video_path, audio_path, face_path,
                #     sample_name=sample_name, subtitle=subtitle, hand_strength=hand_strength,
                #     my_chips=p.chips, pot=self.pot, current_bet=self.current_bet,
                #     opponents_chips=opponents_chips
                # )
                
                # 推理模式下，重新计算 CFR 选择动作的概率
                n_actions = len(full_strategy)
                sigma_a = full_strategy.get(action, 1.0 / n_actions)
                epsilon = self.cfr_solver.epsilon
                q_a = epsilon * (1.0 / n_actions) + (1.0 - epsilon) * sigma_a
            
            # 执行下注动作
            bet_amount = self._execute_bet(p, action)
            logger.info(f"  │  🎲 [{p.name}] 选择 {action_to_name(action)}" + 
                       (f" 下注{bet_amount}" if bet_amount > 0 else "") +
                       f" | 剩余筹码:{p.chips} 底池:{self.pot}")
            
            # 记录动作
            self.action_history.append((p.player_id, action))
            
            # 生成信息集字符串（与solver.py对齐）
            # 使用延迟导入避免循环依赖（兼容包内和独立运行）
            try:
                from .solver import MERObserver, MERGameState, MERAction
            except ImportError:
                from solver import MERObserver, MERGameState, MERAction
            state = MERGameState(current_round=round_idx, current_player=p.player_id)
            state.confidence_history[p.player_id] = p.level_history
            state.action_history = self.action_history[:-1]  # 不包含当前动作
            infoset = MERObserver.get_infoset_string(state, p.player_id)
            
            # 【OS-MCCFR】记录完整的决策点（包含概率信息）
            trajectory.add_decision(DecisionPoint(
                player_id=p.player_id,
                infoset=infoset,
                action=action,
                legal_actions=list(MERAction),
                confidence=p.current_level,
                perception_prob=1.0,  # 确定性模式
                sigma_a=sigma_a,      # 策略概率
                q_a=q_a,              # 采样概率
                full_strategy=full_strategy  # 完整策略
            ))
            
            # 【OS-MCCFR】累积概率到 trajectory
            # 初始化概率（如果尚未初始化）
            if not trajectory.reach_probs:
                trajectory.reach_probs = {pid: 1.0 for pid in range(num_players)}
            
            # π_i(z) *= σ(a) for current player
            trajectory.reach_probs[p.player_id] *= sigma_a
            # q(z) *= q(a)
            trajectory.sample_prob *= q_a
    
    def _execute_decision_round_with_revealer_last(
        self,
        trajectory: GameTrajectory,
        prompt_fn: Callable,
        video_path: str,
        audio_path: Optional[str],
        face_path: Optional[str],
        round_idx: int,
        revealer_pos: int,
        sample_name: str = None,
        subtitle: str = None,
        player_names: Dict[int, str] = None
    ):
        """
        执行揭示轮的决策（被揭示者最后决策）
        
        决策顺序：
        - 从被揭示者的下一位开始，顺时针绕一圈
        - 被揭示者最后决策
        
        例如：玩家顺序为 [0:Qwen, 1:Llama, 2:Gemma]
        - 若揭示者是 1:Llama，则决策顺序为：2:Gemma → 0:Qwen → 1:Llama
        
        【OS-MCCFR 概率追踪】同 _execute_decision_round
        【关键设计】动作来源：训练模式用 CFR 采样，推理模式用 LLM
        """
        num_players = len(self.players)
        revealer = self.players[revealer_pos]
        
        # 构建决策顺序：从揭示者下一位开始，揭示者最后
        decision_order = []
        for i in range(1, num_players):
            pos = (revealer_pos + i) % num_players
            decision_order.append(pos)
        decision_order.append(revealer_pos)  # 揭示者最后
        
        logger.debug(f"  │  📋 决策顺序: {[self.players[pos].name for pos in decision_order]} (揭示者:{revealer.name}最后)")
        
        # 按顺序遍历玩家
        for player_idx in decision_order:
            p = self.players[player_idx]
            
            if p.is_folded:
                continue
            
            # 【OS-MCCFR】使用 sample_action_with_probs 获取策略和采样概率
            cfr_sampled_action, sigma_a, q_a, full_strategy = self.cfr_solver.sample_action_with_probs(
                player_id=p.player_id,
                confidence_history=p.level_history,
                action_history=self.action_history,
                current_round=round_idx
            )
            
            # 格式化 CFR 建议
            advice = self.cfr_solver.format_advice_for_prompt(full_strategy)
            
            # 显示CFR建议（顺序：K C R F，FOLD放最后避免LLM偏好选择首位）
            probs_str = " ".join([f"{['F','K','C','R'][a]}:{full_strategy.get(a,0)*100:.2f}%" for a in [1,2,3,0]])
            logger.info(f"  │  📈 [{p.name}] CFR建议: {probs_str}")
            
            # 【关键分支】根据模式决定动作来源
            if self.mode == "training":
                # ========== 训练模式：使用 CFR 当前策略采样（标准 OS-MCCFR）==========
                action = cfr_sampled_action
                logger.info(f"  │  🎯 [{p.name}] 训练模式: CFR采样动作 {action_to_name(action)}")
                
            else:
                # ========== 推理模式：使用 CFR 平均策略（收敛后的纳什均衡近似）==========
                # 【重要】CFR 理论：推理时应使用平均策略，而不是当前策略
                avg_strategy = self.cfr_solver.get_average_strategy(
                    player_id=p.player_id,
                    confidence_history=p.level_history,
                    action_history=self.action_history,
                    current_round=round_idx
                )
                # 选择平均策略中概率最大的动作
                action = max(avg_strategy, key=avg_strategy.get)
                # 更新 full_strategy 用于日志显示
                full_strategy = avg_strategy
                probs_str = " ".join([f"{['F','K','C','R'][a]}:{avg_strategy.get(a,0)*100:.2f}%" for a in [1,2,3,0]])
                logger.info(f"  │  📊 [{p.name}] 平均策略: {probs_str}")
                logger.info(f"  │  🎯 [{p.name}] 推理模式: 平均策略最优动作 {action_to_name(action)}")
                
                # # ========== [已禁用] LLM 决策代码 ==========
                # # 如需恢复 LLM 决策，取消以下注释
                # decision_prompt = prompt_fn(
                #     f"Round {round_idx}, Player {p.name}",
                #     self.action_history,
                #     player_names or {},
                #     round_idx
                # )
                # 
                # hand_strength = level_to_hand_strength(p.current_level)
                # opponents_chips = [
                #     (get_player_short_name(player_name=opp.name), opp.chips) 
                #     for opp in self.players 
                #     if opp.player_id != p.player_id and not opp.is_folded
                # ]
                # 
                # action = p.decide(
                #     decision_prompt, advice, video_path, audio_path, face_path,
                #     sample_name=sample_name, subtitle=subtitle, hand_strength=hand_strength,
                #     my_chips=p.chips, pot=self.pot, current_bet=self.current_bet,
                #     opponents_chips=opponents_chips
                # )
                
                # 推理模式下，重新计算概率
                n_actions = len(full_strategy)
                sigma_a = full_strategy.get(action, 1.0 / n_actions)
                epsilon = self.cfr_solver.epsilon
                q_a = epsilon * (1.0 / n_actions) + (1.0 - epsilon) * sigma_a
            
            # 执行下注动作
            bet_amount = self._execute_bet(p, action)
            logger.info(f"  │  🎲 [{p.name}] 选择 {action_to_name(action)}" + 
                       (f" 下注{bet_amount}" if bet_amount > 0 else "") +
                       f" | 剩余筹码:{p.chips} 底池:{self.pot}")
            
            # 记录动作
            self.action_history.append((p.player_id, action))
            
            # 生成信息集字符串（与solver.py对齐）
            try:
                from .solver import MERObserver, MERGameState, MERAction
            except ImportError:
                from solver import MERObserver, MERGameState, MERAction
            state = MERGameState(current_round=round_idx, current_player=p.player_id)
            state.confidence_history[p.player_id] = p.level_history
            state.action_history = self.action_history[:-1]  # 不包含当前动作
            infoset = MERObserver.get_infoset_string(state, p.player_id)
            
            # 【OS-MCCFR】记录完整的决策点（包含概率信息）
            trajectory.add_decision(DecisionPoint(
                player_id=p.player_id,
                infoset=infoset,
                action=action,
                legal_actions=list(MERAction),
                confidence=p.current_level,
                perception_prob=1.0,  # 确定性模式
                sigma_a=sigma_a,
                q_a=q_a,
                full_strategy=full_strategy
            ))
            
            # 【OS-MCCFR】累积概率到 trajectory
            if not trajectory.reach_probs:
                trajectory.reach_probs = {pid: 1.0 for pid in range(num_players)}
            trajectory.reach_probs[p.player_id] *= sigma_a
            trajectory.sample_prob *= q_a

    def _settle_round(
        self,
        trajectory: GameTrajectory,
        ground_truth: str
    ) -> GameTrajectory:
        """
        结算一局博弈（零和筹码版）
        
        【核心公式】
        payoff[i] = final_chips[i] - initial_chips[i]
        
        【场景覆盖】
        A. 唯一获胜者：获胜者得底池，其他人损失投入
        B. 多人获胜（平局）：获胜者平分底池
        C. 无人获胜（所有未弃牌者预测错误）：未弃牌者平分底池
        D. 独存者获胜（其他人全弃牌）：独存者得底池
        E. 全员弃牌：所有人平分底池
        
        【零和保证】
        由于筹码总量守恒，Σ payoff = Σ(final - initial) = 0 恒成立
        """
        active_players = [p for p in self.players if not p.is_folded]
        num_players = len(self.players)
        
        # 【Step 1】计算所有玩家的预测得分（用于判定获胜者）
        all_scores = []
        all_extracted = []  # 【新增】保存所有玩家的提取标签，用于后续评估
        for p in self.players:
            # 无论是否跳过评分，都先提取标签（确保预测文本被处理）
            extracted = self.referee.label_extractor(p.prediction)
            all_extracted.append(extracted)
            
            # 【配置可选】推理态可跳过评分计算
            if self.skip_score_calculation and self.mode == "inference":
                # 跳过评分：仅记录提取结果，不计算与 GT 的匹配分数
                score = 0.0
                fold_mark = "[已弃牌] " if p.is_folded else ""
                logger.info(f"  │  🏷️ [{p.name}] {fold_mark}提取标签: '{extracted}' (评分已跳过)")
            else:
                # 正常评分逻辑
                try:
                    score = self.referee.score_calculator(extracted, ground_truth)
                except Exception as e:
                    logger.warning(f"  │  ⚠️ [{p.name}] 评分失败 ({e})，默认给 0 分")
                    score = 0.0
                fold_mark = "[已弃牌] " if p.is_folded else ""
                logger.info(f"  │  🏷️ [{p.name}] {fold_mark}提取标签: '{extracted}' vs GT: '{ground_truth}' → 得分: {score:.2f}")
            
            all_scores.append(score)
        
        # 【新增】保存所有玩家的预测和提取标签到 trajectory（用于后续 evaluation）
        trajectory.player_predictions = [p.prediction for p in self.players]
        trajectory.player_extracted_labels = all_extracted
        
        # 【Step 2】确定获胜者并分配底池
        winners = []
        
        # 场景 E：全员弃牌 → 底池平分给所有人
        if not active_players:
            logger.info("  │  ⚠️ 场景E: 全员弃牌，底池平分给所有人")
            winners = list(range(num_players))  # 所有人都是"获胜者"
            self._distribute_pot(winners)
        
        # 场景 D：独存者获胜（其他人全弃牌）
        elif len(active_players) == 1:
            winner = active_players[0]
            winners = [winner.player_id]
            trajectory.winner_id = winner.player_id
            trajectory.winner_prediction = winner.prediction
            logger.info(f"  │  ✓ 场景D: 独存者获胜 → {winner.name}")
            self._distribute_pot(winners)
        
        # 场景 A/B/C：多名玩家存活，需要判定
        else:
            if self.mode == "training":
                # 训练态：基于 GT 得分判定
                active_ids = [p.player_id for p in active_players]
                active_scores = [(pid, all_scores[pid]) for pid in active_ids]
                max_score = max(s for _, s in active_scores)
                
                # 找出所有最高分玩家（可能平局）
                winners = [pid for pid, s in active_scores if abs(s - max_score) < 1e-6]
                
                # 判断是否有人预测正确（得分 > 0 表示正确）
                if max_score > 0:
                    # 场景 A/B：有人预测正确
                    if len(winners) == 1:
                        logger.info(f"  │  ✓ 场景A: 唯一获胜者 → {self.players[winners[0]].name} (得分: {max_score:.4f})")
                    else:
                        logger.info(f"  │  ✓ 场景B: 平局获胜 → {[self.players[w].name for w in winners]} (得分: {max_score:.4f})")
                else:
                    # 场景 C：所有未弃牌者预测错误 → 底池平分给所有未弃牌者
                    winners = active_ids
                    logger.info(f"  │  ✓ 场景C: 无人预测正确，底池平分给未弃牌者 → {[self.players[w].name for w in winners]}")
            else:
                # 推理态：置信度加权 EV 判定 + 讨论回合平局打破
                def discussion_callback(tied_ids, labels):
                    """平局时触发讨论回合，让所有玩家投票选出最佳预测"""
                    return self._run_discussion_round(
                        tied_ids, labels,
                        self._current_video_path,
                        self._current_audio_path,
                        self._current_face_path,
                        self._current_sample_name,
                        self._current_subtitle
                    )
                
                winners, _ = self.referee.judge_by_ev(
                    self.players, self.cfr_solver, self.action_history,
                    discussion_callback=discussion_callback,
                    extracted_labels=all_extracted
                )
                logger.info(f"  │  ✓ 推理态判定获胜: {[self.players[w].name for w in winners]}")
            
            # 分配底池
            if winners:
                self._distribute_pot(winners)
                trajectory.winner_id = winners[0]
                trajectory.winner_prediction = self.players[winners[0]].prediction
        
        # 【Step 3】计算零和收益：payoff = final_chips - initial_chips
        trajectory.final_payoffs = []
        for i, p in enumerate(self.players):
            initial = trajectory.initial_chips[i] if trajectory.initial_chips else p.initial_chips
            payoff = float(p.chips - initial)
            trajectory.final_payoffs.append(payoff)
        
        # 【Step 4】零和验证与日志
        total_payoff = sum(trajectory.final_payoffs)
        if abs(total_payoff) > 1e-6:
            logger.warning(f"  │  ⚠️ 零和验证失败! Σ payoff = {total_payoff:.2f}")
        else:
            logger.info(f"  │  ✅ 零和验证通过: Σ payoff = {total_payoff:.2f}")
        
        # 显示详细收益
        payoff_strs = [f"{self.players[i].name}: {trajectory.final_payoffs[i]:+.0f}" for i in range(num_players)]
        logger.info(f"  │  💰 本局收益: {', '.join(payoff_strs)}")
        
        # 推理态记录预测
        if self.mode == "inference" and trajectory.winner_prediction:
            self.system_predictions[trajectory.sample_name] = trajectory.winner_prediction
        
        return trajectory
    
    def get_predictions(self) -> Dict[str, str]:
        """获取所有系统预测（推理态）"""
        return self.system_predictions.copy()


# ==========================================
# 6. 便捷工厂函数
# ==========================================

def create_game_driver(
    llm_callables: List[Callable],
    player_names: List[str],
    cfr_solver,
    mode: str = "training",
    label_extractor: Optional[Callable] = None,
    score_calculator: Optional[Callable] = None
) -> MERGameDriver:
    """
    创建博弈驱动器
    
    :param llm_callables: LLM调用接口列表
    :param player_names: 玩家名称列表
    :param cfr_solver: CFR求解器
    :param mode: 运行模式
    :param label_extractor: 标签提取器
    :param score_calculator: 评分计算器
    :return: MERGameDriver实例
    """
    # 创建玩家
    players = []
    for i, (callable_fn, name) in enumerate(zip(llm_callables, player_names)):
        player = MERPlayer(
            player_id=i,
            name=name,
            llm_callable=callable_fn
        )
        players.append(player)
    
    # 创建裁判
    referee = Referee(
        label_extractor=label_extractor,
        score_calculator=score_calculator
    )
    
    # 创建驱动器
    driver = MERGameDriver(
        players=players,
        cfr_solver=cfr_solver,
        referee=referee,
        mode=mode
    )
    
    return driver


# ==========================================
# 7. 测试代码
# ==========================================

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(name)s - %(levelname)s - %(message)s'
    )
    
    logger.info("="*60)
    logger.info("MERGame Driver 测试")
    logger.info("="*60)
    
    # 模拟LLM调用（返回固定响应和随机置信度）
    import random
    
    def mock_llm(prompt, video_path, audio_path, face_path):
        """模拟LLM调用"""
        confidence = random.uniform(0.3, 0.9)
        responses = ["happy", "sad", "angry", "neutral", "surprised"]
        response = random.choice(responses)
        return response, confidence
    
    # 创建CFR求解器
    from OpenSpielGame.solver import create_mer_solver
    cfr_solver = create_mer_solver(warmup_iterations=100)
    
    # 创建驱动器
    driver = create_game_driver(
        llm_callables=[mock_llm, mock_llm, mock_llm],
        player_names=["Player_A", "Player_B", "Player_C"],
        cfr_solver=cfr_solver,
        mode="training"
    )
    
    # 运行几局测试
    for i in range(3):
        logger.info(f"\n>>> 测试局 {i+1} <<<")
        trajectory = driver.play_round(
            sample_name=f"test_sample_{i}",
            ground_truth="happy",
            video_path="/fake/video.mp4"
        )
        logger.info(f"Winner: {trajectory.winner_id}, Payoffs: {trajectory.final_payoffs}")
    
    logger.info("\n" + "="*60)
    logger.info("测试完成")
    logger.info("="*60)
