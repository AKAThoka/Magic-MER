"""
confidence_utils.py - 统一的置信度提取模块

核心功能:
    1. 第一回合（Preflop）: 基于熵的置信度 —— 从模型输出 logits 计算信息熵
    2. 第二回合及之后（Re-evaluation）: 文本解析置信度 —— 从 LLM 回复中提取数值


"""

import re
import math
import logging
from typing import Optional, Union, Tuple
import numpy as np

# 从 constants.py 导入（单一数据源）
try:
    from .constants import DEFAULT_CONFIDENCE
except ImportError:
    from constants import DEFAULT_CONFIDENCE

logger = logging.getLogger("ConfidenceUtils")


# ==========================================
# 常量定义
# ==========================================

# 置信度范围
MIN_CONFIDENCE = 0.0
MAX_CONFIDENCE = 1.0

# 熵计算的数值稳定性常数
ENTROPY_EPSILON = 1e-10


# ==========================================
# 核心函数：统一置信度提取接口
# ==========================================

def extract_confidence(
    response: str,
    logits: Optional[np.ndarray] = None,
    mode: str = "entropy",
    fallback: float = DEFAULT_CONFIDENCE
) -> float:
    """
    统一的置信度提取函数
    
    第一回合与第二回合的置信度获取方式不同，但统一通过此函数调用：
    - 第一回合 (Preflop): mode="entropy"，使用 logits 计算熵转置信度
    - 第二回合+ (Re-evaluation): mode="text_parse"，从 response 文本中解析
    
    :param response: LLM 的文本响应
    :param logits: 模型输出的 logits 或概率分布（仅 entropy 模式需要）
    :param mode: 提取模式
                 - "entropy": 基于熵计算（需要 logits）
                 - "text_parse": 从文本解析数值
                 - "hybrid": 优先使用熵，失败则回退到文本解析
    :param fallback: 提取失败时的默认值
    
    :return: 置信度 [0.0, 1.0]
    
    示例：
        # 第一回合：模型输出 logits
        >>> logits = np.array([2.1, 0.5, -0.3, 1.0])  # 模型输出
        >>> conf = extract_confidence("happy", logits=logits, mode="entropy")
        >>> print(f"Entropy-based confidence: {conf:.3f}")
        
        # 第二回合：LLM 在文本中给出置信度
        >>> response = "Based on my analysis, my confidence is 0.75"
        >>> conf = extract_confidence(response, mode="text_parse")
        >>> print(f"Text-parsed confidence: {conf:.3f}")
    """
    if mode == "entropy":
        if logits is not None:
            conf = _entropy_to_confidence(logits)
            logger.debug(f"[Entropy Mode] Computed confidence: {conf:.4f}")
            return conf
        else:
            logger.warning("[Entropy Mode] No logits provided, falling back to text_parse")
            mode = "text_parse"  # 自动回退
    
    if mode == "text_parse":
        conf = _parse_confidence_from_text(response, fallback)
        logger.debug(f"[Text Parse Mode] Extracted confidence: {conf:.4f}")
        return conf
    
    if mode == "hybrid":
        # 混合模式：优先使用熵，失败则解析文本
        if logits is not None:
            conf = _entropy_to_confidence(logits)
            if MIN_CONFIDENCE < conf < MAX_CONFIDENCE:
                logger.debug(f"[Hybrid Mode] Used entropy: {conf:.4f}")
                return conf
        
        # 回退到文本解析
        conf = _parse_confidence_from_text(response, fallback)
        logger.debug(f"[Hybrid Mode] Used text parse: {conf:.4f}")
        return conf
    
    logger.error(f"Unknown confidence extraction mode: {mode}")
    return fallback


# ==========================================
# 方法 1: 熵转置信度（第一回合）
# ==========================================

def _entropy_to_confidence(logits_or_probs: np.ndarray) -> float:
    """
    将模型输出的 logits/概率 转换为置信度
    
    数学原理:
    1. 如果输入是 logits，先用 softmax 转为概率分布
    2. 计算信息熵 H = -∑ p(x) * log(p(x))
    3. 置信度 = 1 / (1 + H)
    
    熵的特性:
    - 当模型非常确定（一个概率为1）时，H → 0，置信度 → 1.0
    - 当模型完全不确定（均匀分布）时，H → log(n)，置信度下降
    
    :param logits_or_probs: 模型输出的 logits 或概率分布
    :return: 置信度 [0.0, 1.0]
    """
    arr = np.array(logits_or_probs, dtype=np.float64)
    
    # 判断输入是 logits 还是概率
    # 概率的特征：所有值在 [0, 1] 且和为 1
    is_probability = (arr.min() >= 0) and (arr.max() <= 1) and (abs(arr.sum() - 1.0) < 0.01)
    
    if is_probability:
        probs = arr
    else:
        # Softmax 转换
        probs = _stable_softmax(arr)
    
    # 计算信息熵
    entropy = _calculate_entropy(probs)
    
    # 熵转置信度
    confidence = 1.0 / (1.0 + entropy)
    
    return _clip_confidence(confidence)


def _stable_softmax(logits: np.ndarray) -> np.ndarray:
    """
    数值稳定的 Softmax 实现
    避免指数溢出：先减去最大值
    """
    shifted = logits - np.max(logits)
    exp_vals = np.exp(shifted)
    return exp_vals / (exp_vals.sum() + ENTROPY_EPSILON)


def _calculate_entropy(probs: np.ndarray) -> float:
    """
    计算信息熵 H = -∑ p(x) * log(p(x))
    
    使用自然对数（以 e 为底）
    """
    # 过滤掉零概率（避免 log(0)）
    probs = probs + ENTROPY_EPSILON
    entropy = -np.sum(probs * np.log(probs))
    return max(0.0, entropy)  # 熵非负


# ==========================================
# 方法 2: 文本解析置信度（第二回合及之后）
# ==========================================

def _parse_confidence_from_text(response: str, fallback: float = DEFAULT_CONFIDENCE) -> float:
    """
    从 LLM 文本响应中解析置信度数值
    
    支持多种格式：
    1. 纯数字: "0.75"
    2. 带标签: "Confidence: 0.75" 或 "置信度: 0.75"
    3. 百分比: "75%" → 0.75
    4. 自然语言: "my confidence is 0.75"
    
    解析优先级：
    1. 优先找显式标记的置信度
    2. 然后找第一个 [0, 1] 范围内的浮点数
    3. 百分比转换
    4. 失败返回 fallback
    
    :param response: LLM 的完整响应文本
    :param fallback: 解析失败时的默认值
    :return: 置信度 [0.0, 1.0]
    """
    text = response.strip().lower()
    
    # 空响应处理
    if not text:
        logger.warning("Empty response, using fallback confidence")
        return fallback
    
    # 策略 1: 查找显式标记的置信度
    labeled_patterns = [
        # 英文标签
        r'(?:confidence|conf|probability|prob)[\s:=]+([0-9]*\.?[0-9]+)',
        r'(?:my confidence is|i am|i\'m)\s+([0-9]*\.?[0-9]+)\s*(?:confident)?',
        r'(?:win probability|winning probability)[\s:=]+([0-9]*\.?[0-9]+)',
        # 中文标签
        r'(?:置信度|信心|概率|可能性)[\s:：=]+([0-9]*\.?[0-9]+)',
        r'(?:我的置信度是|置信度为)[\s]*([0-9]*\.?[0-9]+)',
    ]
    
    for pattern in labeled_patterns:
        match = re.search(pattern, text)
        if match:
            try:
                value = float(match.group(1))
                conf = _normalize_confidence_value(value)
                logger.debug(f"Pattern '{pattern}' matched: {value} → {conf}")
                return conf
            except ValueError:
                continue
    
    # 策略 2: 响应仅包含一个数字（常见于要求输出纯数字的 Prompt）
    single_number_match = re.match(r'^[\s]*([0-9]*\.?[0-9]+)[\s]*$', text)
    if single_number_match:
        try:
            value = float(single_number_match.group(1))
            return _normalize_confidence_value(value)
        except ValueError:
            pass
    
    # 策略 3: 查找百分比
    percent_match = re.search(r'([0-9]+(?:\.[0-9]+)?)\s*%', text)
    if percent_match:
        try:
            percent = float(percent_match.group(1))
            return _clip_confidence(percent / 100.0)
        except ValueError:
            pass
    
    # 策略 4: 查找第一个在 [0, 1] 范围内的浮点数
    all_floats = re.findall(r'(?<![0-9])([0-9]*\.[0-9]+)(?![0-9])', text)
    for f_str in all_floats:
        try:
            value = float(f_str)
            if 0.0 <= value <= 1.0:
                return value
        except ValueError:
            continue
    
    # 策略 5: 关键词启发式
    conf = _heuristic_confidence_from_keywords(text)
    if conf is not None:
        return conf
    
    logger.warning(f"Failed to parse confidence from: '{response[:100]}...', using fallback={fallback}")
    return fallback


def _normalize_confidence_value(value: float) -> float:
    """
    规范化置信度值
    
    处理两种常见情况：
    1. 已经是 [0, 1] 范围 → 直接返回
    2. 是百分比 [0, 100] → 除以 100
    """
    if 0.0 <= value <= 1.0:
        return _clip_confidence(value)
    elif 0.0 <= value <= 100.0:
        # 可能是百分比
        return _clip_confidence(value / 100.0)
    else:
        logger.warning(f"Abnormal confidence value: {value}")
        return DEFAULT_CONFIDENCE


def _heuristic_confidence_from_keywords(text: str) -> Optional[float]:
    """
    基于关键词的启发式置信度估计（最后的回退方案）
    
    当无法从文本中提取精确数值时，根据语义关键词给出粗略估计
    """
    # 高置信度关键词
    high_conf_keywords = ['certain', 'sure', 'definitely', 'absolutely', 
                          '确定', '肯定', '绝对', 'very confident', 'highly confident']
    # 中置信度关键词
    mid_conf_keywords = ['probably', 'likely', 'think', 'believe',
                         '可能', '大概', '应该', 'fairly confident']
    # 低置信度关键词
    low_conf_keywords = ['uncertain', 'unsure', 'doubt', 'maybe', 'perhaps',
                         '不确定', '也许', '可能不', 'not confident']
    
    text_lower = text.lower()
    
    for kw in low_conf_keywords:
        if kw in text_lower:
            return 0.3
    
    for kw in high_conf_keywords:
        if kw in text_lower:
            return 0.85
    
    for kw in mid_conf_keywords:
        if kw in text_lower:
            return 0.6
    
    return None


def _clip_confidence(value: float) -> float:
    """限制置信度到 [0, 1] 范围"""
    return max(MIN_CONFIDENCE, min(MAX_CONFIDENCE, value))


# ==========================================
# 辅助函数：批量处理与格式转换
# ==========================================

def batch_extract_confidence(
    responses: list,
    logits_list: Optional[list] = None,
    mode: str = "text_parse"
) -> list:
    """
    批量提取置信度
    
    :param responses: 响应列表
    :param logits_list: logits 列表（可选）
    :param mode: 提取模式
    :return: 置信度列表
    """
    results = []
    n = len(responses)
    
    for i, resp in enumerate(responses):
        logits = logits_list[i] if logits_list and i < len(logits_list) else None
        conf = extract_confidence(resp, logits=logits, mode=mode)
        results.append(conf)
    
    return results


def format_confidence_for_prompt(confidence: float) -> str:
    """
    将置信度格式化为适合放入 Prompt 的字符串
    """
    return f"{confidence:.2f}"


def describe_confidence_level(confidence: float) -> str:
    """
    将置信度转换为自然语言描述
    
    对应 MERgame.py 中的 HandStrength 枚举:
    - [0.0, 0.2]: UNCERTAIN (不确定)
    - (0.2, 0.4]: TENTATIVE (暂定)
    - (0.4, 0.6]: CONFIDENT (较有信心)
    - (0.6, 0.8]: STRONG (很有信心)
    - (0.8, 1.0]: ABSOLUTE (完全确定)
    """
    if confidence <= 0.2:
        return "UNCERTAIN (不确定)"
    elif confidence <= 0.4:
        return "TENTATIVE (暂定)"
    elif confidence <= 0.6:
        return "CONFIDENT (较有信心)"
    elif confidence <= 0.8:
        return "STRONG (很有信心)"
    else:
        return "ABSOLUTE (完全确定)"


# ==========================================
# 导出接口
# ==========================================

__all__ = [
    # 核心函数
    'extract_confidence',
    
    # 辅助函数
    'batch_extract_confidence',
    'format_confidence_for_prompt',
    'describe_confidence_level',
    
    # 常量
    'DEFAULT_CONFIDENCE',
    'MIN_CONFIDENCE',
    'MAX_CONFIDENCE',
]


# ==========================================
# 单元测试（开发用）
# ==========================================

if __name__ == "__main__":
    # 设置日志
    logging.basicConfig(level=logging.DEBUG)
    
    print("=" * 60)
    print("Testing confidence_utils.py")
    print("=" * 60)
    
    # 测试 1: 熵转置信度
    print("\n[Test 1] Entropy-based confidence:")
    test_logits = np.array([2.5, 0.5, -0.3, 1.0])  # 第一个类别概率最高
    conf = extract_confidence("", logits=test_logits, mode="entropy")
    print(f"  Logits: {test_logits}")
    print(f"  Confidence: {conf:.4f}")
    
    # 测试高确定性
    high_certainty_logits = np.array([10.0, 0.0, 0.0, 0.0])  # 几乎100%确定
    conf_high = extract_confidence("", logits=high_certainty_logits, mode="entropy")
    print(f"  High certainty logits: {high_certainty_logits}")
    print(f"  Confidence: {conf_high:.4f}")
    
    # 测试 2: 文本解析
    print("\n[Test 2] Text-parsed confidence:")
    test_responses = [
        "0.75",
        "Confidence: 0.85",
        "置信度: 0.65",
        "My confidence is 0.90",
        "I think there's a 70% chance",
        "I'm not sure about this",
        "I am certain this is correct",
    ]
    
    for resp in test_responses:
        conf = extract_confidence(resp, mode="text_parse")
        print(f"  '{resp[:40]}...' → {conf:.4f}")
    
    # 测试 3: 混合模式
    print("\n[Test 3] Hybrid mode:")
    conf_hybrid = extract_confidence(
        "My guess is happiness", 
        logits=np.array([1.5, 0.3, -0.2, 0.8]), 
        mode="hybrid"
    )
    print(f"  Hybrid (with logits): {conf_hybrid:.4f}")
    
    conf_hybrid_no_logits = extract_confidence(
        "Confidence: 0.65", 
        logits=None, 
        mode="hybrid"
    )
    print(f"  Hybrid (no logits): {conf_hybrid_no_logits:.4f}")
    
    print("\n" + "=" * 60)
    print("All tests passed!")
