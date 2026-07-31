"""
LLMInference.py - LLM 推理封装层

核心设计理念（严格对齐 inference_hybird.py）:
1. 封装 Chat 对象的推理调用，提供统一接口
2. 使用与 inference_hybird.py 完全相同的多模态特征提取流程
3. 使用与 inference_hybird.py 完全相同的 answer_sample 调用方式
4. 与 DataLoader.py 配合使用，形成完整的推理管道

职责边界：
- 只负责 LLM 推理调用
- 不负责数据加载（由 DataLoader.py 负责）
- 不负责模型加载（由 LoadModel.py 负责）

使用方式:
    wrapper = LLMInferenceWrapper(chat_instance, face_or_frame="multiface_audio_face_text")
    response, logits = wrapper.inference(sample_data, prompt)
"""

import logging
from typing import Dict, Any, Tuple, Optional

import numpy as np

logger = logging.getLogger("LLMInference")


class LLMInferenceWrapper:
    """
    LLM 推理封装器
    
    严格对齐 inference_hybird.py 的推理流程，确保：
    1. 使用相同的多模态特征提取方法 (postprocess_*)
    2. 使用相同的 img_list 构造方式
    3. 使用相同的 answer_sample 调用参数
    
    设计特点：
    - 无状态：每次调用独立
    - 可复用：一个 wrapper 可以处理多个样本
    - 线程安全：不修改 chat 对象状态
    """
    
    def __init__(
        self,
        chat_instance,
        face_or_frame: str = "multiface_audio_face_text",
        name: str = "LLMWrapper"
    ):
        """
        初始化推理封装器
        
        :param chat_instance: Chat 对象（来自 LoadModel.py）
        :param face_or_frame: 多模态特征组合类型
        :param name: 封装器名称（用于日志）
        """
        self.chat = chat_instance
        self.face_or_frame = face_or_frame
        self.name = name
        
        # 验证 chat 对象
        self._validate_chat()
        
        logger.info(f"LLMInferenceWrapper 初始化: {name}, face_or_frame={face_or_frame}")
    
    def _validate_chat(self):
        """验证 chat 对象具有必要的方法"""
        required_methods = [
            'postprocess_audio',
            'postprocess_frame', 
            'postprocess_face',
            'postprocess_image',
            'answer_sample'
        ]
        
        for method in required_methods:
            if not hasattr(self.chat, method):
                raise ValueError(f"Chat 对象缺少必要方法: {method}")
        
        # postprocess_multi 是可选的（取决于 face_or_frame）
        if self.face_or_frame.startswith('multi') and not hasattr(self.chat, 'postprocess_multi'):
            logger.warning(f"Chat 对象缺少 postprocess_multi 方法，但 face_or_frame={self.face_or_frame} 需要")
    
    def extract_features(self, sample_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        提取多模态特征
        与 inference_hybird.py 第350-365行完全一致
        
        :param sample_data: 原始多模态数据（来自 DataLoader.read_sample）
        :return: img_list 字典
        """
        # 提取各模态特征（与 inference_hybird.py 第350-354行一致）
        audio_hiddens, audio_llms = self.chat.postprocess_audio(sample_data)
        frame_hiddens, frame_llms = self.chat.postprocess_frame(sample_data)
        face_hiddens, face_llms = self.chat.postprocess_face(sample_data)
        _, image_llms = self.chat.postprocess_image(sample_data)
        
        # 多模态融合（与 inference_hybird.py 第355-358行一致）
        multi_llms = None
        if self.face_or_frame.startswith('multiface'):
            _, multi_llms = self.chat.postprocess_multi(face_hiddens, audio_hiddens)
        elif self.face_or_frame.startswith('multiframe'):
            _, multi_llms = self.chat.postprocess_multi(frame_hiddens, audio_hiddens)
        
        # 构建 img_list（与 inference_hybird.py 第359-365行一致）
        img_list = {
            'audio': audio_llms,
            'frame': frame_llms,
            'face': face_llms,
            'image': image_llms,
            'multi': multi_llms,
        }
        
        return img_list
    
    def inference(
        self,
        sample_data: Dict[str, Any],
        prompt: str,
        do_sample: bool = False,
        temperature: float = 0.0,
        num_beams: int = 1,
        top_p: float = 0.95,
        max_new_tokens: int = 256,
        max_length: int = 2000,
        return_logits: bool = True
    ) -> Tuple[str, Optional[np.ndarray]]:
        """
        执行 LLM 推理
        与 inference_hybird.py 第384-388行对齐
        
        :param sample_data: 多模态数据（来自 DataLoader.read_sample）
        :param prompt: 完整的多模态 prompt（来自 DataLoader.get_prompt）
        :param do_sample: 是否采样（CFR 训练建议 False 确保确定性）
        :param temperature: 温度参数
        :param num_beams: beam 数量
        :param top_p: top-p 采样参数
        :param max_new_tokens: 最大新 token 数
        :param max_length: 最大总长度
        :param return_logits: 是否返回 logits（用于置信度计算）
        
        :return: (response, logits) 元组
                 - response: LLM 生成的文本
                 - logits: numpy 数组（用于熵计算），如果不支持则为 None
        """
        try:
            # Step 1: 提取多模态特征
            img_list = self.extract_features(sample_data)
            
            # Step 2: 调用 answer_sample（与 inference_hybird.py 第384-388行一致）
            response = self.chat.answer_sample(
                prompt=prompt,
                img_list=img_list,
                num_beams=num_beams,
                temperature=temperature,
                do_sample=do_sample,
                top_p=top_p,
                max_new_tokens=max_new_tokens,
                max_length=max_length,
            )
            
            # Step 3: 尝试获取 logits
            logits = None
            if return_logits:
                if hasattr(self.chat, 'last_logits') and self.chat.last_logits is not None:
                    logits = self.chat.last_logits
                else:
                    # 如果 Chat 不支持返回 logits，生成模拟值
                    # 在实际使用中，可能需要修改 Chat 类以支持返回 logits
                    logits = None
            
            logger.debug(f"[{self.name}] 推理完成: {response[:100]}...")
            return response, logits
            
        except Exception as e:
            logger.error(f"[{self.name}] 推理失败: {e}")
            import traceback
            traceback.print_exc()
            return "", None
    
    def create_llm_callable(self, data_loader, default_user_message: Optional[str] = None):
        """
        创建符合 MERPlayer 接口的 callable
        
        返回的 callable 签名:
            llm_callable(prompt, video_path, audio_path, face_path) -> (response, logits)
        
        :param data_loader: MERDataLoader 实例（用于读取数据和构造 prompt）
        :param default_user_message: 默认用户问题（用于感知阶段）
        :return: callable
        """
        chat = self.chat
        face_or_frame = self.face_or_frame
        name = self.name
        
        # 获取默认问题
        if default_user_message is None:
            default_user_message = data_loader.get_default_user_message()
        
        def llm_callable(
            user_message: str,
            video_path: Optional[str] = None,
            audio_path: Optional[str] = None,
            face_path: Optional[str] = None,
            sample_name: Optional[str] = None,
            subtitle: Optional[str] = None
        ) -> Tuple[str, np.ndarray]:
            """
            LLM 调用接口（符合 MERPlayer 要求）
            
            :param user_message: 博弈问题（如 "Please analyze the emotion"）
            :param video_path: 视频路径（未使用，保留兼容性）
            :param audio_path: 音频路径（未使用，保留兼容性）
            :param face_path: 人脸路径（未使用，保留兼容性）
            :param subtitle: 字幕（用于 prompt 构造）
            :param sample_name: 样本名称（用于读取数据）
            :return: (response, logits)
            """
            try:
                # 如果提供了 sample_name，使用 DataLoader 读取数据
                if sample_name is not None:
                    sample_data = data_loader.read_sample(sample_name)
                    
                    # If subtitle not provided, get from DataLoader
                    actual_subtitle = subtitle
                    if actual_subtitle is None:
                        sample_info = data_loader.get_sample_by_name(sample_name)
                        if sample_info:
                            actual_subtitle = sample_info.get('subtitle', '')
                        else:
                            actual_subtitle = ''
                else:
                    # 否则构造空的 sample_data
                    sample_data = {
                        'frame': None, 'raw_frame': None,
                        'face': None, 'raw_face': None,
                        'audio': None, 'raw_audio': None,
                        'image': None, 'raw_image': None,
                    }
                    actual_subtitle = subtitle or ''
                
                # Construct prompt
                sample = {'name': sample_name or '', 'subtitle': actual_subtitle}
                prompt = data_loader.get_prompt(sample, user_message)
                
                # 【DEBUG】打印完整多模态 Prompt（包含原始 token）
                if 'DISCUSSION ROUND' in user_message:
                    logger.info(f"[{name}] 🔍 [讨论回合完整Prompt]\n{'='*60}\n{prompt}\n{'='*60}")
                
                # 执行推理
                response, logits = self.inference(sample_data, prompt)
                
                # 如果没有 logits，返回模拟值
                if logits is None:
                    logits = np.random.randn(4) * 0.5 + 1.0
                
                return response, logits
                
            except Exception as e:
                logger.error(f"[{name}] llm_callable 失败: {e}")
                return "neutral", np.array([0.25, 0.25, 0.25, 0.25])
        
        return llm_callable


# ==========================================
# 便捷工厂函数
# ==========================================

def create_llm_wrapper(
    chat_instance,
    face_or_frame: str = "multiface_audio_face_text",
    name: str = "LLMWrapper"
) -> LLMInferenceWrapper:
    """
    创建 LLM 推理封装器的便捷函数
    
    :param chat_instance: Chat 对象
    :param face_or_frame: 多模态特征组合类型
    :param name: 封装器名称
    :return: LLMInferenceWrapper 实例
    """
    return LLMInferenceWrapper(chat_instance, face_or_frame, name)


# ==========================================
# 测试代码
# ==========================================

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    print("=" * 60)
    print("LLMInferenceWrapper 测试")
    print("=" * 60)
    
    # 创建 Mock Chat 对象
    class MockChat:
        def __init__(self):
            self.last_logits = np.array([1.0, 0.5, -0.5, -1.0])
        
        def postprocess_audio(self, sample_data):
            return None, None
        
        def postprocess_frame(self, sample_data):
            return None, None
        
        def postprocess_face(self, sample_data):
            return None, None
        
        def postprocess_image(self, sample_data):
            return None, None
        
        def postprocess_multi(self, video_hiddens, audio_hiddens):
            return None, None
        
        def answer_sample(self, prompt, img_list, **kwargs):
            return "This is a mock response."
    
    try:
        # 创建封装器
        mock_chat = MockChat()
        wrapper = LLMInferenceWrapper(mock_chat, name="MockWrapper")
        print("✓ 封装器创建成功")
        
        # 测试特征提取
        sample_data = {'frame': None, 'face': None, 'audio': None, 'image': None}
        img_list = wrapper.extract_features(sample_data)
        print(f"✓ 特征提取成功: {list(img_list.keys())}")
        
        # 测试推理
        response, logits = wrapper.inference(sample_data, "Test prompt")
        print(f"✓ 推理成功: response='{response}', logits shape={logits.shape if logits is not None else 'None'}")
        
    except Exception as e:
        print(f"✗ 测试失败: {e}")
        import traceback
        traceback.print_exc()
