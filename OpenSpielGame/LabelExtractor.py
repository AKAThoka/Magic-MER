"""
LabelExtractor.py - 情感标签提取器

功能：
    使用独立的 LLM (Qwen2.5) 从 AffectGPT 的自然语言输出中提取情感标签
    
核心设计：
    1. 复用 my_affectgpt/evaluation/ew_metric.py 的 reason_to_openset 逻辑
    2. 复用 my_affectgpt/evaluation/wheel.py 的 wheel_metric_calculation 评分逻辑
    3. 支持单样本评分（适用于博弈论的实时结算）

使用方式:
    extractor = OVLabelExtractor(device="cuda:2")
    labels = extractor.extract_openset_labels("The character appears happy and excited...")
    score = extractor.compute_wheel_score(labels, "happy, excited")
"""

import os
import re
import sys
import logging
from pathlib import Path
from typing import Optional, List, Dict, Tuple, Union
import numpy as np

# 添加项目根目录到 path
SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))

logger = logging.getLogger("LabelExtractor")


class OVLabelExtractor:
    """
    开放词汇情感标签提取器
    
    完全复用 MER2025 的评估流程：
    1. reason -> openset labels (使用 reason_to_openset_qwen 的 prompt)
    2. 使用 wheel_metric_calculation 的评分逻辑
    """
    
    def __init__(
        self,
        model_name: str = "Qwen25",
        device: str = "cuda:2",
        use_vllm: bool = False,
        temperature: float = 0.0,
        max_tokens: int = 512
    ):
        """
        初始化标签提取器
        
        :param model_name: 模型名称（对应 config.py 中的 PATH_TO_LLM）
        :param device: GPU 设备（如 "cuda:2"）
        :param use_vllm: 是否使用 vLLM
        :param temperature: 生成温度
        :param max_tokens: 最大生成 token 数
        """
        self.model_name = model_name
        self.device = device
        self.use_vllm = use_vllm
        self.temperature = temperature
        self.max_tokens = max_tokens
        
        # 模型状态
        self.llm = None
        self.tokenizer = None
        self.sampling_params = None
        self._initialized = False
        
        # 加载 wheel 评估所需的映射
        self.format_mapping = None
        self.raw_mapping = None
        self._wheel_maps = {}  # 缓存不同 wheel/level 的映射
        
        logger.info(f"OVLabelExtractor 初始化: model={model_name}, device={device}, use_vllm={use_vllm}")
    
    def _load_wheel_mappings(self):
        """加载 wheel 评估所需的映射表"""
        if self.format_mapping is not None:
            return
        
        try:
            from my_affectgpt.evaluation.wheel import read_format2raws, read_candidate_synonym_merge
            self.format_mapping = read_format2raws()          # level3 -> level2
            self.raw_mapping = read_candidate_synonym_merge() # level2 -> level1
            logger.info(f"✓ 加载 wheel 映射表: format_mapping={len(self.format_mapping)}, raw_mapping={len(self.raw_mapping)}")
        except Exception as e:
            logger.warning(f"无法加载 wheel 映射表: {e}")
            self.format_mapping = {}
            self.raw_mapping = {}
    
    def _get_wheel_map(self, wheel: str = 'wheel1', level: str = 'level1') -> Dict[str, str]:
        """获取 wheel 聚类映射（带缓存）"""
        key = f"{wheel}_{level}"
        if key not in self._wheel_maps:
            try:
                from my_affectgpt.evaluation.wheel import func_get_wheel_cluster
                self._wheel_maps[key] = func_get_wheel_cluster(wheel, level)
            except Exception as e:
                logger.warning(f"无法加载 wheel_map ({key}): {e}")
                self._wheel_maps[key] = {}
        return self._wheel_maps[key]
    
    def initialize(self):
        """延迟加载模型"""
        if self._initialized:
            return
        
        logger.info(f"正在加载标签提取模型 {self.model_name}...")
        
        # 加载 wheel 映射
        self._load_wheel_mappings()
        
        # 获取模型路径
        import config
        model_path = config.PATH_TO_LLM.get(self.model_name)
        if model_path is None:
            raise ValueError(f"未找到模型 {self.model_name}，请检查 config.py 中的 PATH_TO_LLM")
        
        # 处理相对路径
        if not os.path.isabs(model_path):
            model_path = os.path.join(PROJECT_ROOT, model_path)
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"模型路径不存在: {model_path}")
        
        if self.use_vllm:
            self._init_vllm(model_path)
        else:
            self._init_transformers(model_path)
        
        self._initialized = True
        logger.info(f"✓ 标签提取模型加载完成")
    
    def _init_vllm(self, model_path: str):
        """使用 vLLM 初始化"""
        try:
            from vllm import LLM, SamplingParams
            from transformers import AutoTokenizer
            
            # 解析 GPU ID
            gpu_id = 0
            if "cuda:" in self.device:
                gpu_id = int(self.device.split(":")[1])
            
            os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
            
            self.llm = LLM(model=model_path, tensor_parallel_size=1, gpu_memory_utilization=0.5, seed=42)
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
            
            # 确保可复现性：temperature=0 时使用贪婪解码
            if self.temperature == 0:
                self.sampling_params = SamplingParams(
                    temperature=0,        # 贪婪解码
                    top_p=1.0,
                    top_k=-1,             # 禁用 top_k 采样
                    repetition_penalty=1.05,
                    max_tokens=self.max_tokens,
                    seed=42               # vLLM 随机种子
                )
            else:
                self.sampling_params = SamplingParams(
                    temperature=self.temperature,
                    top_p=1.0,
                    repetition_penalty=1.05,
                    max_tokens=self.max_tokens,
                    seed=42
                )
        except Exception as e:
            logger.warning(f"vLLM 初始化失败: {e}，回退到 transformers 模式")
            self.use_vllm = False
            self._init_transformers(model_path)
    
    def _init_transformers(self, model_path: str):
        """使用 transformers 初始化"""
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        
        logger.info(f"📦 使用 transformers 加载模型到 {self.device}")
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.llm = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True
        )
        self.llm = self.llm.to(self.device)
        self.llm.eval()
        
        logger.info(f"✅ transformers 模型加载完成，设备: {self.device}")
    
    def _build_openset_prompt(self, reason: str) -> str:
        """
        构建开放词汇标签提取 Prompt
        
        完全复用 toolkit/utils/qwen.py 中的 reason_to_openset_qwen
        """
        prompt = f"""Please assume the role of an expert in the field of emotions. \
We provide clues that may be related to the emotions of the characters. Based on the provided clues, please identify the emotional states of the main character. \
The main character is the one with the most detailed clues. \
Please separate different emotional categories with commas and output only the clearly identifiable emotional categories in a list format. \
If none are identified, please output an empty list. \
Input: We cannot recognize his emotional state; Output: [] \
Input: His emotional state is happy, sad, and angry; Output: [happy, sad, angry] \
Input: {reason}; Output: """
        return prompt
    
    def _postprocess_qwen(self, response: str) -> str:
        """
        后处理 Qwen 输出
        
        复用 toolkit/utils/qwen.py 中的 func_postprocess_qwen
        """
        response = response.strip()
        prefixes = ["输入", "输出", "翻译", "让我们来翻译一下：", "output", "Output", "input", "Input"]
        for prefix in prefixes:
            if response.startswith(prefix):
                response = response[len(prefix):]
        response = response.strip()
        if response.startswith(":") or response.startswith("："):
            response = response[1:]
        response = response.strip().replace('\n', '')
        return response
    
    def _generate_response(self, prompt: str) -> str:
        """生成模型响应（确保可复现性）"""
        if self.use_vllm:
            outputs = self.llm.generate([prompt], self.sampling_params)
            return outputs[0].outputs[0].text
        else:
            import torch
            messages = [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ]
            text = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            inputs = self.tokenizer([text], return_tensors="pt").to(self.device)
            
            with torch.no_grad():
                # 确保可复现性：temperature=0 时强制使用贪婪解码
                if self.temperature == 0:
                    outputs = self.llm.generate(
                        inputs.input_ids,
                        attention_mask=inputs.attention_mask,
                        max_new_tokens=self.max_tokens,
                        do_sample=False,               # 禁用采样，使用贪婪解码
                        num_beams=1,                   # 单束搜索
                        pad_token_id=self.tokenizer.eos_token_id
                    )
                else:
                    outputs = self.llm.generate(
                        inputs.input_ids,
                        attention_mask=inputs.attention_mask,
                        max_new_tokens=self.max_tokens,
                        temperature=self.temperature,
                        do_sample=True,
                        pad_token_id=self.tokenizer.eos_token_id
                    )
            return self.tokenizer.decode(
                outputs[0][len(inputs.input_ids[0]):],
                skip_special_tokens=True
            )
    
    def extract_openset_labels(self, reason: str) -> str:
        """
        从 LLM 响应中提取开放词汇标签
        
        :param reason: AffectGPT 的输出文本
        :return: 逗号分隔的标签字符串，如 "happy, sad, angry"
        """
        self.initialize()
        
        prompt = self._build_openset_prompt(reason)
        raw_output = self._generate_response(prompt)
        response = self._postprocess_qwen(raw_output)
        
        logger.debug(f"提取结果: '{reason[:50]}...' -> '{response}'")
        return response
    
    def extract_openset_labels_batch(self, reasons: List[str]) -> List[str]:
        """批量提取标签"""
        self.initialize()
        
        prompts = [self._build_openset_prompt(r) for r in reasons]
        
        if self.use_vllm:
            outputs = self.llm.generate(prompts, self.sampling_params)
            raw_outputs = [out.outputs[0].text for out in outputs]
        else:
            raw_outputs = [self._generate_response(p) for p in prompts]
        
        return [self._postprocess_qwen(out) for out in raw_outputs]
    
    def _string_to_list(self, s: str) -> List[str]:
        """
        将字符串转换为标签列表
        
        支持格式：
        - "happy, sad, angry"
        - "[happy, sad, angry]"
        - "['happy', 'sad']"
        """
        if not s or s == '[]':
            return []
        
        # 去除方括号
        s = s.strip()
        if s.startswith('[') and s.endswith(']'):
            s = s[1:-1]
        
        # 分割并清理
        labels = []
        for item in s.split(','):
            label = item.strip().strip('"').strip("'").lower()
            if label and label != '[]':
                labels.append(label)
        
        return labels
    
    def _map_label_to_synonym(self, label: str, metric: str = 'case3_wheel1_level1') -> str:
        """
        将标签映射到标准化形式
        
        复用 wheel.py 中的 func_map_label_to_synonym 逻辑
        """
        self._load_wheel_mappings()
        
        label = label.lower().strip()
        
        # 根据 metric 类型选择映射层级
        if metric.startswith('case3'):
            _, wheelname, levelname = metric.split('_')
            wheel_map = self._get_wheel_map(wheelname, levelname)
            
            # level3 -> level2 -> level1 -> wheel cluster
            if label not in self.format_mapping:
                return ""
            
            level1_whole = []
            for format_label in self.format_mapping.get(label, []):
                for raw in self.raw_mapping.get(format_label, []):
                    level1_whole.append(raw)
            
            for level1 in sorted(level1_whole):
                if level1 in wheel_map:
                    return wheel_map[level1]
            return ""
        
        elif metric.startswith('case2'):
            if label not in self.format_mapping:
                return ""
            stage1 = sorted(self.format_mapping.get(label, []))[0] if self.format_mapping.get(label) else ""
            if stage1 and stage1 in self.raw_mapping:
                return sorted(self.raw_mapping[stage1])[0]
            return ""
        
        else:  # case1
            if label not in self.format_mapping:
                return ""
            return sorted(self.format_mapping.get(label, []))[0] if self.format_mapping.get(label) else ""
    
    def compute_wheel_score(
        self,
        pred_labels: str,
        gt_labels: str,
        level: str = 'level1'
    ) -> Tuple[float, float, float]:
        """
        计算 wheel-based 评分
        
        复用 wheel.py 中的 calculate_openset_overlap_rate 逻辑
        
        :param pred_labels: 预测标签（逗号分隔字符串或列表格式字符串）
        :param gt_labels: GT 标签（逗号分隔字符串）
        :param level: 评估层级 ('level1' 或 'level2')
        :return: (fscore, precision, recall)
        """
        self._load_wheel_mappings()
        
        # 使用多个 wheel 计算平均分数（与 wheel_metric_calculation 一致）
        if level == 'level1':
            metrics = [
                'case3_wheel1_level1',
                'case3_wheel2_level1',
                'case3_wheel3_level1',
                'case3_wheel4_level1',
                'case3_wheel5_level1',
            ]
        else:
            metrics = [
                'case3_wheel1_level2',
                'case3_wheel2_level2',
                'case3_wheel3_level2',
                'case3_wheel4_level2',
                'case3_wheel5_level2',
            ]
        
        whole_scores = []
        for metric in metrics:
            precision, recall = self._calculate_single_overlap(pred_labels, gt_labels, metric)
            if precision + recall > 0:
                fscore = 2 * (precision * recall) / (precision + recall)
            else:
                fscore = 0.0
            whole_scores.append([fscore, precision, recall])
        
        # 计算平均分数
        avg_scores = np.mean(whole_scores, axis=0).tolist() if whole_scores else [0.0, 0.0, 0.0]
        return tuple(avg_scores)
    
    def _calculate_single_overlap(self, pred_labels: str, gt_labels: str, metric: str) -> Tuple[float, float]:
        """
        计算单个 metric 下的 precision 和 recall
        """
        # 解析标签
        gt_list = self._string_to_list(gt_labels)
        pred_list = self._string_to_list(pred_labels)
        
        # 映射到标准化形式
        gt_mapped = set()
        for label in gt_list:
            mapped = self._map_label_to_synonym(label, metric)
            if mapped:
                gt_mapped.add(mapped)
        
        pred_mapped = set()
        for label in pred_list:
            mapped = self._map_label_to_synonym(label, metric)
            if mapped:
                pred_mapped.add(mapped)
        
        # 计算 precision 和 recall
        if len(gt_mapped) == 0:
            return 0.0, 0.0
        
        if len(pred_mapped) == 0:
            return 0.0, 0.0
        
        intersection = len(gt_mapped & pred_mapped)
        precision = intersection / len(pred_mapped)
        recall = intersection / len(gt_mapped)
        
        return precision, recall
    
    def extract_and_score(self, reason: str, gt_labels: str) -> Tuple[str, float]:
        """
        一站式：提取标签并计算分数
        
        :param reason: AffectGPT 的输出文本
        :param gt_labels: GT 标签
        :return: (提取的标签, fscore)
        """
        pred_labels = self.extract_openset_labels(reason)
        fscore, _, _ = self.compute_wheel_score(pred_labels, gt_labels)
        return pred_labels, fscore
    
    # ==========================================
    # Referee 兼容接口
    # ==========================================
    
    def create_label_extractor_fn(self):
        """创建与 Referee 兼容的 label_extractor 函数"""
        def extractor(response: str) -> str:
            return self.extract_openset_labels(response)
        return extractor
    
    def create_score_calculator_fn(self):
        """创建与 Referee 兼容的 score_calculator 函数"""
        def calculator(pred: str, gt: str) -> float:
            fscore, _, _ = self.compute_wheel_score(pred, gt)
            return fscore
        return calculator


class RuleBasedLabelExtractor(OVLabelExtractor):
    """
    轻量级规则标签提取器（无额外 LLM 显存开销）

    设计目标：
    1. 复用 OVLabelExtractor 的 wheel 评分逻辑（compute_wheel_score）
    2. 仅通过关键词匹配提取标签，不加载任何大模型
    3. 作为推理消融场景的低显存替代方案
    """

    BASE_KEYWORDS = {
        "happy", "sad", "angry", "fear", "fearful", "anxious", "anxiety",
        "surprised", "surprise", "neutral", "excited", "disappointed",
        "frustrated", "stress", "stressed", "calm", "joy", "joyful",
        "depressed", "nervous", "confident", "worried", "questioning",
        "critical", "dissatisfied", "embarrassed", "ashamed", "grateful",
        "relieved", "confused", "curious"
    }

    def __init__(self):
        super().__init__(model_name="RULE", device="cpu", use_vllm=False, temperature=0.0, max_tokens=0)
        self._keyword_vocab: List[str] = []

    def initialize(self):
        """规则模式初始化：仅加载映射，不加载 LLM。"""
        if self._initialized:
            return

        self._load_wheel_mappings()
        self._keyword_vocab = self._build_keyword_vocab()
        self._initialized = True
        logger.info(f"✓ 规则标签提取器初始化完成（关键词数量={len(self._keyword_vocab)}）")

    def _build_keyword_vocab(self) -> List[str]:
        vocab = set(self.BASE_KEYWORDS)

        for k, v in (self.format_mapping or {}).items():
            if isinstance(k, str):
                vocab.add(k.lower().strip())
            if isinstance(v, (list, tuple, set)):
                for item in v:
                    if isinstance(item, str):
                        vocab.add(item.lower().strip())

        for k, v in (self.raw_mapping or {}).items():
            if isinstance(k, str):
                vocab.add(k.lower().strip())
            if isinstance(v, (list, tuple, set)):
                for item in v:
                    if isinstance(item, str):
                        vocab.add(item.lower().strip())

        cleaned = []
        for term in vocab:
            term = term.strip()
            if not term:
                continue
            if len(term) <= 1:
                continue
            if len(term) > 48:
                continue
            cleaned.append(term)

        # 长词优先，减少 "sad" 命中 "dissatisfied" 之类子串误匹配
        cleaned = sorted(set(cleaned), key=lambda x: (-len(x), x))
        return cleaned

    def _extract_bracket_labels(self, text: str) -> List[str]:
        matches = re.findall(r"\[([^\]]+)\]", text)
        labels = []
        for content in matches:
            for part in re.split(r"[,;/，；、\n]+", content):
                token = part.strip().strip('"').strip("'").lower()
                if token:
                    labels.append(token)
        return labels

    def _extract_keyword_labels(self, text: str) -> List[str]:
        hits = []
        for term in self._keyword_vocab:
            # 仅对英文字母做边界约束，避免误匹配子串
            pattern = rf"(?<![a-z]){re.escape(term)}(?![a-z])"
            m = re.search(pattern, text)
            if m:
                hits.append((m.start(), term))
        hits.sort(key=lambda x: x[0])
        return [term for _, term in hits]

    def extract_openset_labels(self, reason: str) -> str:
        self.initialize()

        if not reason:
            return "[]"

        text = str(reason).lower()

        ordered = []
        seen = set()

        for token in self._extract_bracket_labels(text):
            if token not in seen:
                ordered.append(token)
                seen.add(token)

        for token in self._extract_keyword_labels(text):
            if token not in seen:
                ordered.append(token)
                seen.add(token)

        if not ordered:
            return "[]"

        # 控制输出上限，避免长文本误召回过多噪声标签
        return ", ".join(ordered[:12])


# ==========================================
# 保持向后兼容的别名
# ==========================================

LabelExtractor = OVLabelExtractor


# ==========================================
# 便捷工厂函数
# ==========================================

def create_label_extractor(
    use_llm: bool = True,
    model_name: str = "Qwen25",
    device: str = "cuda:2",
    **kwargs
) -> LabelExtractor:
    """
    创建标签提取器
    
    :param use_llm: True 使用 LLM，False 使用规则提取
    :param model_name: LLM 模型名称
    :param device: GPU 设备
    :return: 标签提取器实例
    """
    if use_llm:
        return OVLabelExtractor(model_name=model_name, device=device, **kwargs)
    return RuleBasedLabelExtractor()


# ==========================================
# 测试代码
# ==========================================

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    print("=" * 60)
    print("OVLabelExtractor 测试")
    print("=" * 60)
    
    # 测试 OVLabelExtractor（需要 LLM）
    print("\n--- 测试 OVLabelExtractor ---")
    print("注意：需要可用的 Qwen2.5 LLM")
    
    try:
        extractor = OVLabelExtractor(device="cuda:2")
        extractor.initialize()
        
        test_cases = [
            # (response, gt_labels)
            ("The character's emotional state is dissatisfaction, questioning, criticism, challenge, anger, frustration, anxiety, stress.",
             "dissatisfied, disappointed, questioning, under pressure"),
            ("The character appears to be very happy and excited about the news.",
             "happy, excited"),
            ("Based on the facial expressions and body language, the person seems sad and disappointed.",
             "sad, disappointed"),
        ]
        
        for response, gt in test_cases:
            pred_labels = extractor.extract_openset_labels(response)
            fscore, precision, recall = extractor.compute_wheel_score(pred_labels, gt)
            print(f"输入: {response[:60]}...")
            print(f"GT: {gt}")
            print(f"提取: {pred_labels}")
            print(f"F-score: {fscore:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}")
            print()
    except Exception as e:
        print(f"错误: {e}")
        print("请确保有可用的 Qwen2.5 LLM")
    
    print("\n" + "=" * 60)
    print("测试完成")
    print("=" * 60)

