"""
constants.py - 博弈系统常量定义（单一数据源）

核心原则:
- 所有常量、枚举、映射在此统一定义
- 其他模块从此导入，禁止重复定义
- 确保全系统一致性

内容:
1. MERAction: 动作枚举
2. ACTION_NAMES/ACTION_SHORT: 动作名称映射
3. HandStrength: 置信度档位枚举
4. PLAYER_NAME_MAP: 玩家命名映射
5. 相关工具函数
"""

from enum import IntEnum
from typing import Dict, Optional


# ==========================================
# 1. 动作定义（唯一来源）
# ==========================================

class MERAction(IntEnum):
    """
    MER博弈动作空间
    
    使用 IntEnum 确保可以用整数索引，同时保持类型安全
    """
    FOLD = 0   # 弃牌：放弃当前回合
    CHECK = 1  # 过牌：不加注，观望
    CALL = 2   # 跟注：匹配当前最高注额
    RAISE = 3  # 加注：增加注额


# 动作完整名称（用于日志和 prompt）
ACTION_NAMES: Dict[int, str] = {
    MERAction.FOLD: "FOLD",
    MERAction.CHECK: "CHECK",
    MERAction.CALL: "CALL",
    MERAction.RAISE: "RAISE",
}

# 动作缩写（用于信息集字符串和紧凑显示）
ACTION_SHORT: Dict[int, str] = {
    MERAction.FOLD: "F",
    MERAction.CHECK: "K",  # K for checK，避免与 Call 的 C 混淆
    MERAction.CALL: "C",
    MERAction.RAISE: "R",
}

# 动作列表（便于遍历）
ALL_ACTIONS = [MERAction.FOLD, MERAction.CHECK, MERAction.CALL, MERAction.RAISE]
NUM_ACTIONS = len(ALL_ACTIONS)


def action_to_name(action: int, short: bool = False) -> str:
    """
    动作ID转名称
    
    :param action: 动作ID (0-3)
    :param short: 是否使用缩写
    :return: 动作名称
    """
    mapping = ACTION_SHORT if short else ACTION_NAMES
    return mapping.get(action, f"ACTION-{action}")


def name_to_action(name: str) -> Optional[int]:
    """
    动作名称转ID
    
    :param name: 动作名称（支持全名和缩写，大小写不敏感）
    :return: 动作ID，无效名称返回 None
    """
    name_upper = name.upper().strip()
    
    # 全名映射
    name_map = {"FOLD": 0, "CHECK": 1, "CALL": 2, "RAISE": 3}
    if name_upper in name_map:
        return name_map[name_upper]
    
    # 缩写映射
    short_map = {"F": 0, "K": 1, "C": 2, "R": 3}
    if name_upper in short_map:
        return short_map[name_upper]
    
    return None


# ==========================================
# 2. 置信度档位定义（唯一来源）
# ==========================================

class HandStrength(IntEnum):
    """
    手牌强度（置信度档位）- 10档精细划分
    
    映射规则（每 0.1 一档，捕获 LLM 的细微置信度变化）:
    - 0.00 - 0.10: LEVEL_0 (极度困惑)
    - 0.10 - 0.20: LEVEL_1 (非常不确定)
    - 0.20 - 0.30: LEVEL_2 (不太确定)
    - 0.30 - 0.40: LEVEL_3 (略有怀疑)
    - 0.40 - 0.50: LEVEL_4 (中等偏低)
    - 0.50 - 0.60: LEVEL_5 (中等偏高)
    - 0.60 - 0.70: LEVEL_6 (较有把握)
    - 0.70 - 0.80: LEVEL_7 (相当确信)
    - 0.80 - 0.90: LEVEL_8 (非常确信)
    - 0.90 - 1.00: LEVEL_9 (绝对笃定)
    """
    LEVEL_0 = 0
    LEVEL_1 = 1
    LEVEL_2 = 2
    LEVEL_3 = 3
    LEVEL_4 = 4
    LEVEL_5 = 5
    LEVEL_6 = 6
    LEVEL_7 = 7
    LEVEL_8 = 8
    LEVEL_9 = 9


# 置信度档位数量
NUM_CONFIDENCE_LEVELS = 10

# 档位名称映射
LEVEL_NAMES: Dict[int, str] = {
    0: "EXTREMELY_UNCERTAIN",
    1: "VERY_UNCERTAIN",
    2: "UNCERTAIN",
    3: "SLIGHTLY_DOUBTFUL",
    4: "BELOW_MODERATE",
    5: "ABOVE_MODERATE",
    6: "FAIRLY_CONFIDENT",
    7: "CONFIDENT",
    8: "VERY_CONFIDENT",
    9: "ABSOLUTE",
}

# 档位到德州扑克手牌的映射（用于 prompt）- 10级牌力体系
LEVEL_TO_HAND: Dict[int, str] = {
    0: "High Card (weakest - scattered cards, no combinations)",
    1: "Low Pair (very weak - small pair like 2-2)",
    2: "One Pair (weak - single pair, easily beaten)",
    3: "Two Pair (below average - vulnerable to higher hands)",
    4: "Three of a Kind (medium-low - decent but not strong)",
    5: "Straight (medium - connected cards, moderate strength)",
    6: "Flush (medium-high - same suit, good winning chance)",
    7: "Full House (strong - pair + three of a kind)",
    8: "Four of a Kind (very strong - rare and powerful)",
    9: "Royal Flush (invincible - the absolute nuts)",
}

# 档位边界（9个阈值划分10个区间）
CONFIDENCE_THRESHOLDS = [0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90]


def confidence_to_level(confidence: float) -> int:
    """
    将连续置信度映射到离散档位（10档）
    
    :param confidence: 置信度值 [0.0, 1.0]
    :return: 档位 [0, 9]
    """
    # 使用阈值列表进行映射
    for i, threshold in enumerate(CONFIDENCE_THRESHOLDS):
        if confidence <= threshold:
            return i
    return HandStrength.LEVEL_9  # confidence > 0.90


def level_to_name(level: int) -> str:
    """档位到名称的映射"""
    return LEVEL_NAMES.get(level, "UNKNOWN")


def level_to_hand_strength(level: int) -> str:
    """
    将置信度档位映射为德州扑克手牌强度描述
    
    用于在 Prompt 中告知 LLM 当前的"手牌"强度
    
    :param level: 置信度档位 [0, 9]
    :return: 德州扑克手牌描述
    """
    return LEVEL_TO_HAND.get(level, "Unknown Hand")


# ==========================================
# 3. 玩家命名系统（唯一来源）
# ==========================================

# 玩家名称映射：从完整名称/player_id 到统一简称
PLAYER_NAME_MAP: Dict = {
    # 基于名称关键词（小写）
    "qwen": "Player-Q",
    "llama": "Player-L",
    "gemma": "Player-G",
    # 基于 player_id（默认映射）
    0: "Player-Q",
    1: "Player-L",
    2: "Player-G",
}

# LLM 类型映射
LLM_TYPE_MAP: Dict[str, str] = {
    "qwen": "Qwen25",
    "llama": "Llama31",
    "gemma": "Gemma3",
}

# 默认玩家数量
DEFAULT_NUM_PLAYERS = 3


def get_player_short_name(player_id: int = None, player_name: str = None) -> str:
    """
    获取玩家的统一简称
    
    优先级：
    1. 根据 player_name 中的关键词匹配 (qwen/llama/gemma)
    2. 根据 player_id 使用默认映射
    3. 返回 "Player-X" (X = player_id)
    
    :param player_id: 玩家ID (0, 1, 2)
    :param player_name: 玩家完整名称 (如 "AffectGPT-Qwen")
    :return: 统一简称 (如 "Player-Q")
    """
    # 优先根据名称匹配
    if player_name:
        name_lower = player_name.lower()
        for key, short_name in PLAYER_NAME_MAP.items():
            if isinstance(key, str) and key in name_lower:
                return short_name
    
    # 根据 player_id 映射
    if player_id is not None and player_id in PLAYER_NAME_MAP:
        return PLAYER_NAME_MAP[player_id]
    
    # 默认返回
    return f"Player-{player_id}" if player_id is not None else "Player"


def get_llm_type(player_name: str) -> str:
    """
    根据玩家名称获取 LLM 类型
    
    :param player_name: 玩家名称
    :return: LLM 类型 (Qwen25, Llama31, Gemma3)
    """
    if player_name:
        name_lower = player_name.lower()
        for key, llm_type in LLM_TYPE_MAP.items():
            if key in name_lower:
                return llm_type
    return "Qwen25"  # 默认


# ==========================================
# 4. 博弈配置常量
# ==========================================

# 默认轮次配置
DEFAULT_NUM_ROUNDS = 4  # 1轮感知(Preflop) + N轮揭示(N=玩家数)

# CFR 默认参数
DEFAULT_EPSILON = 0.05      # 最小动作概率（探索保证）
DEFAULT_WEIGHT_MAX = 1e4    # 最终权重硬上限
DEFAULT_SQRT_MAX_RAW = 1e6  # 开根号前的原始权重上限（sqrt(1e4)=100）

# 筹码配置（可被 yaml 配置覆盖）
DEFAULT_INITIAL_CHIPS = 100000000  # 每位玩家初始筹码
DEFAULT_ANTE = 10              # 强制底注（所有玩家平等投入，保证对称性）
DEFAULT_RAISE_AMOUNT = 20

# 【已废弃】盲注配置 - 改用强制底注以保证CFR对称性
# DEFAULT_SMALL_BLIND = 10
# DEFAULT_BIG_BLIND = 20

# 默认置信度（0.2 对应 LEVEL_2，作为 LLM 很少触及的独立档位）
DEFAULT_CONFIDENCE = 0.15


# ==========================================
# 5. 导出接口
# ==========================================

__all__ = [
    # 动作相关
    'MERAction',
    'ACTION_NAMES',
    'ACTION_SHORT',
    'ALL_ACTIONS',
    'NUM_ACTIONS',
    'action_to_name',
    'name_to_action',
    
    # 置信度相关
    'HandStrength',
    'NUM_CONFIDENCE_LEVELS',
    'LEVEL_NAMES',
    'LEVEL_TO_HAND',
    'CONFIDENCE_THRESHOLDS',
    'confidence_to_level',
    'level_to_name',
    'level_to_hand_strength',
    
    # 玩家相关
    'PLAYER_NAME_MAP',
    'LLM_TYPE_MAP',
    'DEFAULT_NUM_PLAYERS',
    'get_player_short_name',
    'get_llm_type',
    
    # 博弈配置
    'DEFAULT_NUM_ROUNDS',
    'DEFAULT_EPSILON',
    'DEFAULT_WEIGHT_MAX',
    'DEFAULT_SQRT_MAX_RAW',
    'DEFAULT_INITIAL_CHIPS',
    'DEFAULT_ANTE',
    'DEFAULT_RAISE_AMOUNT',
    'DEFAULT_CONFIDENCE',
]
