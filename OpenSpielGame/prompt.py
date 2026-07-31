'''
prompt.py - CFR 博弈专用 Prompt 生成器

核心功能:
1. 感知阶段 (Perception): 与 inference_hybird.py 对齐的情感识别问题
2. 重评估阶段 (Re-evaluation): 看到其他玩家预测后重新评估置信度
3. 决策阶段 (Decision): 基于手牌强度和策略建议选择动作

设计原则:
- 轻量级，不依赖 dataset_cls
- 与 MERgame.py 解耦，便于独立测试和修改
- 常量从 constants.py 导入（单一数据源）
'''

import logging
from typing import List, Tuple, Optional, Dict

# 从 constants.py 导入（单一数据源）
try:
    from .constants import (
        ACTION_NAMES, ACTION_SHORT, action_to_name,
        PLAYER_NAME_MAP, get_player_short_name,
        DEFAULT_NUM_PLAYERS
    )
except ImportError:
    from constants import (
        ACTION_NAMES, ACTION_SHORT, action_to_name,
        PLAYER_NAME_MAP, get_player_short_name,
        DEFAULT_NUM_PLAYERS
    )

logger = logging.getLogger("PromptGenerator")


# ==========================================
# 历史格式化工具函数（prompt 专用）
# ==========================================

def format_action_history(
    action_history: List[Tuple[int, int]],
    player_names: Dict[int, str] = None,
    max_rounds: int = 5
) -> str:
    """
    格式化动作历史为可读字符串
    
    输出格式：
    Round 1: Player-Q RAISE, Player-L CALL, Player-G FOLD
    Round 2: Player-Q CHECK, Player-L RAISE
    
    :param action_history: 动作历史 [(player_id, action), ...]
    :param player_names: 玩家ID到名称的映射 {0: "AffectGPT-Qwen", ...}
    :param max_rounds: 最多显示多少轮（避免 prompt 过长）
    :return: 格式化的历史字符串
    """
    if not action_history:
        return "(No actions yet)"
    
    # 按轮次分组（使用 constants 中的默认玩家数）
    num_players = DEFAULT_NUM_PLAYERS
    rounds = []
    current_round = []
    
    for i, (pid, action) in enumerate(action_history):
        short_name = get_player_short_name(pid, player_names.get(pid) if player_names else None)
        action_name = action_to_name(action, short=False)
        current_round.append(f"{short_name} {action_name}")
        
        # 每 num_players 个动作为一轮（简化逻辑）
        if len(current_round) >= num_players:
            rounds.append(current_round)
            current_round = []
    
    # 剩余的动作
    if current_round:
        rounds.append(current_round)
    
    # 只取最近 max_rounds 轮
    if len(rounds) > max_rounds:
        rounds = rounds[-max_rounds:]
        start_round = len(action_history) // num_players - max_rounds + 1
    else:
        start_round = 1
    
    # 格式化输出
    lines = []
    for i, round_actions in enumerate(rounds):
        round_num = start_round + i
        actions_str = ", ".join(round_actions)
        lines.append(f"Round {round_num}: {actions_str}")
    
    return "\n".join(lines)


def format_confidence_history(
    confidence_history: List[float],
    max_entries: int = 5
) -> str:
    """
    格式化置信度历史为可读字符串
    
    输出格式：
    0.70 → 0.60 → 0.55 (decreasing trend)
    
    :param confidence_history: 置信度历史 [0.7, 0.6, 0.55, ...]
    :param max_entries: 最多显示多少个条目
    :return: 格式化的历史字符串
    """
    if not confidence_history:
        return "(No history yet)"
    
    # 只取最近的条目
    if len(confidence_history) > max_entries:
        history = confidence_history[-max_entries:]
        prefix = "... → "
    else:
        history = confidence_history
        prefix = ""
    
    # 格式化数值
    values_str = " → ".join([f"{c:.2f}" for c in history])
    
    # 分析趋势
    if len(history) >= 2:
        first, last = history[0], history[-1]
        if last > first + 0.05:
            trend = "(increasing trend ↑)"
        elif last < first - 0.05:
            trend = "(decreasing trend ↓)"
        else:
            trend = "(stable)"
    else:
        trend = ""
    
    return f"{prefix}{values_str} {trend}".strip()


# ==========================================
# CFR 训练专用 Prompt 工厂（轻量级，不依赖 dataset_cls）
# ==========================================

class CFRGamePromptFactory:
    """
    CFR 博弈训练专用 Prompt 工厂
    
    设计理念：
    - 不依赖 dataset_cls，可独立使用
    - 专注于博弈流程中的三个阶段：感知、重评估、决策
    - 与 AffectGamePromptGenerator 互补，后者专注于多模态 Prompt
    
    使用场景：
    - train_cfr.py 的 prompt 函数创建
    - 需要快速生成博弈 prompt 的场景
    """
    
    # ==========================================
    # 感知阶段 Prompt
    # ==========================================
    
    @staticmethod
    def perception_prompt(sample_name: str = None) -> str:
        """
        Initial perception phase prompt.
        
        CRITICAL: This returns ONLY the user_message portion.
        The full multimodal prompt is constructed by DataLoader.get_prompt()
        which calls dataset_cls.get_prompt_for_multimodal().
        
        This ensures EXACT alignment with inference_hybird.py behavior.
        
        :param sample_name: Sample name (unused, kept for interface compatibility)
        :return: User message for emotion recognition
        """
        # Return the standard zeroshot question used in inference_hybird.py
        # This matches: dataset_cls.func_get_qa_ovlabel(sample=None, question_only=True)
        return "Please recognize all possible emotional states of the character."

    # ==========================================
    # 重评估阶段 Prompt
    # ==========================================
    
    @staticmethod
    def reevaluation_prompt(
        revealed_predictions: List[Tuple[int, str]],
        player_id: int,
        my_prediction: str,
        current_confidence: float = 0.5,
        action_history: List[Tuple[int, int]] = None,
        confidence_history: List[float] = None,
        player_names: Dict[int, str] = None
    ) -> str:
        """
        重评估阶段 Prompt（看到其他玩家预测后）
        
        :param revealed_predictions: 已揭示的预测 [(player_id, prediction), ...]
        :param player_id: 当前玩家 ID
        :param my_prediction: 当前玩家的预测
        :param current_confidence: 当前置信度 (0.0 - 1.0)
        :param action_history: 动作历史 [(player_id, action), ...]
        :param confidence_history: 当前玩家的置信度历史 [0.7, 0.6, ...]
        :param player_names: 玩家ID到名称的映射
        :return: 重评估 Prompt
        """
        # 当前玩家简称
        my_short_name = get_player_short_name(player_id, player_names.get(player_id) if player_names else None)
        
        # 格式化其他玩家的预测（使用统一命名）
        revealed_lines = []
        for pid, pred in revealed_predictions:
            short_name = get_player_short_name(pid, player_names.get(pid) if player_names else None)
            pred_short = pred[:80] + "..." if len(pred) > 80 else pred
            revealed_lines.append(f"{short_name}: {pred_short}")
        revealed_str = "\n".join(revealed_lines) if revealed_lines else "(None revealed yet)"
        
        my_pred_short = my_prediction[:80] + "..." if len(my_prediction) > 80 else my_prediction
        
        # 格式化历史记录
        action_history_str = format_action_history(action_history or [], player_names or {})
        confidence_history_str = format_confidence_history(confidence_history or [])
        
        return f"""RE-EVALUATION PHASE

You are {my_short_name}.

=== Game History ===
Action History:
{action_history_str}

Your Confidence History: {confidence_history_str}

=== Current Round ===
Other players' predictions:
{revealed_str}

Your prediction: {my_pred_short}

=== Confidence Scale ===
- 0.00 - 0.20: Very Uncertain
- 0.20 - 0.40: Tentative
- 0.40 - 0.60: Moderate
- 0.60 - 0.80: Confident
- 0.80 - 1.00: Very Confident

Your current confidence in your prediction: {current_confidence:.2f}

Based on this new information, reassess your confidence level. 
Consider: Do other predictions agree with yours? Should you adjust your confidence?
=== Re-evaluation Guidelines ===
1. No one knows the correct answer. Others may bluff.
2. Consider both their predictions AND betting behavior to judge their true confidence.
3. If predictions strongly agree with yours → increase confidence (0.7+)
   If predictions strongly disagree → decrease confidence (0.3-)
   DO NOT default to 0.5. Moderate confidence (0.4-0.6) should be rare.

**IMPORTANT**: Respond with ONLY a number between 0.0 and 1.0. No text, no explanation.

Your updated confidence: """

    # ==========================================
    # 决策阶段 Prompt
    # ==========================================
    
    @staticmethod
    def decision_prompt(
        game_context: str,
        solver_advice: Optional[str] = None,
        player_confidence: Optional[float] = None,
        action_history: List[Tuple[int, int]] = None,
        player_names: Dict[int, str] = None,
        current_round: int = 0
    ) -> str:
        """
        博弈决策阶段 Prompt
        
        【优化】使用简洁格式，强制 TA: 前缀输出
        注意: 此函数返回的prompt不包含"Your response:"，由MERgame.py组合时添加
        
        :param game_context: 博弈上下文信息 (如 "Round 0, Player AffectGPT-Qwen")
        :param solver_advice: CFR Solver 的策略建议（可选）
        :param player_confidence: 玩家置信度（可选）
        :param action_history: 动作历史 [(player_id, action), ...]
        :param player_names: 玩家ID到名称的映射 {0: "AffectGPT-Qwen", ...}
        :param current_round: 当前轮次
        :return: 决策 Prompt (不含"Your response:")
        """
        # 解析玩家身份：根据名称确定 Player-Q/L/G
        player_short_name = get_player_short_name(player_name=game_context)
        
        # 格式化动作历史
        action_history_str = format_action_history(action_history or [], player_names or {})
        
        # 构建prompt
        prompt = f"""You are {player_short_name} playing Texas Hold'em. Round {current_round}

=== Game History ===
Action History:
{action_history_str}

=== Your Decision ===
Choose your action based on your hand strength, game history, and Strategy Advice.

Rules: Texas Hold'em"""
        
        return prompt

    # ==========================================
    # 讨论回合 Prompt（推理态平局时使用）
    # ==========================================
    
    @staticmethod
    def discussion_prompt(
        tied_predictions: List[Tuple[str, str]],
        player_names: Dict[int, str] = None
    ) -> str:
        """
        讨论回合 Prompt：平局时让所有玩家投票选出最佳预测
        
        设计思路：
        - 向 LLM 展示平局玩家的预测标签列表
        - 让 LLM 根据「精确度 + 覆盖度」原则选择最佳答案
        - 仅需返回选项编号，便于解析
        
        :param tied_predictions: 平局玩家的预测列表 [(player_name, extracted_labels), ...]
        :param player_names: 玩家ID到名称的映射（未使用，保留接口兼容性）
        :return: 讨论回合 Prompt
        """
        # 构建选项列表
        options_lines = []
        for i, (player_name, labels) in enumerate(tied_predictions):
            short_name = get_player_short_name(player_name=player_name)
            options_lines.append(f"  Option {i+1} ({short_name}): {labels}")
        options_str = "\n".join(options_lines)
        
        return f"""DISCUSSION ROUND: Select the Best Emotion Prediction

You have just watched a video clip showing a person's emotional state.

Your task: Select the BEST prediction based on these criteria:
1. **Accuracy**: The emotional labels must be CORRECT (no wrong emotions) and PRECISE (matching observed emotions exactly)
2. **Coverage**: Among accurate predictions, prefer the one that covers more relevant emotion types

=== Candidate Predictions ===
{options_str}

=== Your Vote ===
Reply with ONLY the option number (1, 2, 3, etc.) that you believe is the best.

Your choice: """

    # ==========================================
    # 便捷工厂方法
    # ==========================================
    
    @classmethod
    def create_prompt_functions(cls):
        """
        创建三个阶段的 Prompt 生成函数
        
        用于 train_cfr.py 等需要批量创建 Prompt 函数的场景
        
        :return: (perception_fn, reevaluation_fn, decision_fn) 三元组
        """
        def perception_fn(sample_name: str) -> str:
            return cls.perception_prompt(sample_name)
        
        def reevaluation_fn(
            revealed: List[Tuple[int, str]], 
            player_id: int, 
            my_prediction: str,
            current_confidence: float = 0.5,
            action_history: List[Tuple[int, int]] = None,
            confidence_history: List[float] = None,
            player_names: Dict[int, str] = None
        ) -> str:
            return cls.reevaluation_prompt(
                revealed, player_id, my_prediction, current_confidence,
                action_history, confidence_history, player_names
            )
        
        def decision_fn(
            context: str,
            action_history: List[Tuple[int, int]] = None,
            player_names: Dict[int, str] = None,
            current_round: int = 0
        ) -> str:
            return cls.decision_prompt(
                context, 
                action_history=action_history,
                player_names=player_names,
                current_round=current_round
            )
        
        return perception_fn, reevaluation_fn, decision_fn


# ==========================================
# 便捷函数：直接导出供外部使用
# ==========================================

def get_perception_prompt(sample_name: str) -> str:
    """快捷方式：获取感知 Prompt"""
    return CFRGamePromptFactory.perception_prompt(sample_name)

def get_reevaluation_prompt(
    revealed_predictions: List[Tuple[int, str]],
    player_id: int,
    my_prediction: str,
    current_confidence: float = 0.5,
    action_history: List[Tuple[int, int]] = None,
    confidence_history: List[float] = None,
    player_names: Dict[int, str] = None
) -> str:
    """快捷方式：获取重评估 Prompt"""
    return CFRGamePromptFactory.reevaluation_prompt(
        revealed_predictions, player_id, my_prediction, current_confidence,
        action_history, confidence_history, player_names
    )

def get_decision_prompt(
    game_context: str,
    solver_advice: Optional[str] = None,
    player_confidence: Optional[float] = None,
    action_history: List[Tuple[int, int]] = None,
    player_names: Dict[int, str] = None,
    current_round: int = 0
) -> str:
    """快捷方式：获取决策 Prompt"""
    return CFRGamePromptFactory.decision_prompt(
        game_context, solver_advice, player_confidence,
        action_history, player_names, current_round
    )

def create_cfr_prompt_functions():
    """快捷方式：创建 CFR 训练所需的三个 Prompt 函数"""
    return CFRGamePromptFactory.create_prompt_functions()


def get_discussion_prompt(
    tied_predictions: List[Tuple[str, str]],
    player_names: Dict[int, str] = None
) -> str:
    """快捷方式：获取讨论回合 Prompt"""
    return CFRGamePromptFactory.discussion_prompt(tied_predictions, player_names)


# ==========================================
# 导出接口
# ==========================================

__all__ = [
    'CFRGamePromptFactory',
    'get_player_short_name',
    'format_action_history',
    'format_confidence_history',
    'get_perception_prompt',
    'get_reevaluation_prompt', 
    'get_decision_prompt',
    'get_discussion_prompt',
    'create_cfr_prompt_functions',
    'PLAYER_NAME_MAP',
    'ACTION_NAMES',
]

