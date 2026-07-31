"""
OpenSpielGame - MER博弈系统

核心模块:
- constants.py: 统一常量定义（单一数据源）
- solver.py: CFR求解器，信息集管理，策略学习
- MERgame.py: 博弈驱动器，LLM玩家，裁判系统
- prompt.py: Prompt生成器
- confidence_utils.py: 统一的置信度提取模块（熵/文本解析）

"""

# 从 constants.py 导入（单一数据源）
from .constants import (
    MERAction,
    ACTION_NAMES,
    ACTION_SHORT,
    ALL_ACTIONS,
    action_to_name,
    name_to_action,
    HandStrength,
    NUM_CONFIDENCE_LEVELS,
    LEVEL_NAMES,
    confidence_to_level,
    level_to_name,
    level_to_hand_strength,
    PLAYER_NAME_MAP,
    get_player_short_name,
    DEFAULT_NUM_PLAYERS,
    DEFAULT_NUM_ROUNDS,
    DEFAULT_CONFIDENCE,
)

from .solver import (
    PerceptionSampler,
    InfoStateRecord,
    MERGameState,
    MERObserver,
    PathProbabilityCalculator,
    MERCFRSolver,
    create_mer_solver,
)

from .MERgame import (
    DecisionPoint,
    GameTrajectory,
    MERPlayer,
    Referee,
    MERGameDriver,
    create_game_driver,
)

from .confidence_utils import (
    extract_confidence,
    batch_extract_confidence,
    format_confidence_for_prompt,
    describe_confidence_level,
    MIN_CONFIDENCE,
    MAX_CONFIDENCE,
)

from .prompt import (
    CFRGamePromptFactory,
    format_action_history,
    format_confidence_history,
    get_perception_prompt,
    get_reevaluation_prompt,
    get_decision_prompt,
    create_cfr_prompt_functions,
)

__all__ = [
    # constants.py (单一数据源)
    "MERAction",
    "ACTION_NAMES",
    "ACTION_SHORT",
    "ALL_ACTIONS",
    "action_to_name",
    "name_to_action",
    "HandStrength",
    "NUM_CONFIDENCE_LEVELS",
    "LEVEL_NAMES",
    "confidence_to_level",
    "level_to_name",
    "level_to_hand_strength",
    "PLAYER_NAME_MAP",
    "get_player_short_name",
    "DEFAULT_NUM_PLAYERS",
    "DEFAULT_NUM_ROUNDS",
    "DEFAULT_CONFIDENCE",
    # solver.py
    "PerceptionSampler",
    "InfoStateRecord",
    "MERGameState",
    "MERObserver",
    "PathProbabilityCalculator",
    "MERCFRSolver",
    "create_mer_solver",
    # MERgame.py
    "DecisionPoint",
    "GameTrajectory",
    "MERPlayer",
    "Referee",
    "MERGameDriver",
    "create_game_driver",
    # confidence_utils.py
    "extract_confidence",
    "batch_extract_confidence",
    "format_confidence_for_prompt",
    "describe_confidence_level",
    "MIN_CONFIDENCE",
    "MAX_CONFIDENCE",
    # prompt.py
    "CFRGamePromptFactory",
    "format_action_history",
    "format_confidence_history",
    "get_perception_prompt",
    "get_reevaluation_prompt",
    "get_decision_prompt",
    "create_cfr_prompt_functions",
]
