import re
import copy
import dataclasses
from enum import auto, Enum
from typing import List, Tuple, Any
from PIL import Image
import numpy as np

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, LlamaTokenizer
from transformers import StoppingCriteria, StoppingCriteriaList
from my_affectgpt.common.registry import registry
from my_affectgpt.processors import Blip2ImageEvalProcessor
from my_affectgpt.processors.video_processor import ToTHWC, ToUint8, load_video, load_face
from my_affectgpt.models.ImageBind.data import load_audio, transform_audio
from my_affectgpt.datasets.builders.image_text_pair_builder import *
import config

class SeparatorStyle(Enum):
    """Different separator style."""
    SINGLE = auto()
    TWO = auto()

@dataclasses.dataclass
class Conversation:
    """A class that keeps all conversation history."""
    system: str
    roles: List[str]
    messages: List[List[str]]
    offset: int
    sep_style: SeparatorStyle = SeparatorStyle.SINGLE
    sep: str = "###"
    sep2: str = None

    skip_next: bool = False
    conv_id: Any = None

    def get_prompt(self):
        if self.sep_style == SeparatorStyle.SINGLE:
            ret = self.system + self.sep
            for role, message in self.messages:
                if message:
                    ret += role + ": " + message + self.sep
                else:
                    ret += role + ":"
            return ret
        elif self.sep_style == SeparatorStyle.TWO:
            seps = [self.sep, self.sep2]
            ret = self.system + seps[0]
            for i, (role, message) in enumerate(self.messages):
                if message:
                    ret += role + ": " + message + seps[i % 2]
                else:
                    ret += role + ":"
            return ret
        else:
            raise ValueError(f"Invalid style: {self.sep_style}")

    def append_message(self, role, message):
        self.messages.append([role, message])

    def copy(self):
        return Conversation(
            system=self.system,
            roles=self.roles,
            messages=[[x, y] for x, y in self.messages],
            offset=self.offset,
            sep_style=self.sep_style,
            sep=self.sep,
            sep2=self.sep2,
            conv_id=self.conv_id)

    def dict(self):
        return {
            "system": self.system,
            "roles": self.roles,
            "messages": self.messages,
            "offset": self.offset,
            "sep": self.sep,
            "sep2": self.sep2,
            "conv_id": self.conv_id,
        }


class StoppingCriteriaSub(StoppingCriteria):
    def __init__(self, stops=[], encounters=1):
        super().__init__()
        self.stops = stops

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor):
        for stop in self.stops:
            if torch.all((stop == input_ids[0][-len(stop):])).item():
                return True
        return False


default_conversation = Conversation(
    system="",
    roles=("Human", "Assistant"),
    messages=[],
    offset=0,
    sep_style=SeparatorStyle.SINGLE,
    sep="###",
)

# Gemma 3 专用对话模板（遵循官方格式）
gemma_conversation = Conversation(
    system="",
    roles=("user", "model"),  # Gemma 3 使用 user/model 作为角色名
    messages=[],
    offset=0,
    sep_style=SeparatorStyle.TWO,
    sep="<end_of_turn>\n",     # 第一个分隔符
    sep2="<end_of_turn>\n",    # 第二个分隔符
)


######################################################
# ============== (only for inference) ============== #
######################################################
class Chat:
    def __init__(self, model, model_cfg, device='cuda:0'):
        self.device = device
        self.model = model
        self.tokenizer = model.llama_tokenizer
        self.llama_model_name = model.llama_model_name  # 获取模型名称用于判断
       
        # 根据模型类型设置不同的停止符
        if self.llama_model_name in ['Gemma3']:
            # Gemma 3 使用 <end_of_turn> 作为停止符
            end_of_turn_id = self.tokenizer.encode('<end_of_turn>', add_special_tokens=False)[0]
            stop_words_ids = [
                torch.tensor([self.tokenizer.eos_token_id]).to(self.device),
                torch.tensor([end_of_turn_id]).to(self.device),
            ]
        elif self.llama_model_name.startswith('Qwen'):
            # 【修复】Qwen 3/Qwen 2.5 专用停止符配置
            # Qwen 使用 ChatML 格式，原生停止符是 <|im_end|>
            # 在 Vicuna 格式下，还需要识别 ### 作为模板停止符
            hash_token_ids = self.tokenizer.encode('###', add_special_tokens=False)
            stop_words_ids = [
                torch.tensor([self.tokenizer.eos_token_id]).to(self.device),  # 原生结束符 <|im_end|>
                torch.tensor(hash_token_ids).to(self.device),                  # ### 模板结束符
            ]
        else:
            # Llama 等其他模型使用 ### 作为停止符
            id_tre_jin = self.tokenizer('###', add_special_tokens=False)['input_ids'][0]
            id_two_jin = self.tokenizer('a##', add_special_tokens=False)['input_ids'][1]
            id_one_jin = self.tokenizer('a#', add_special_tokens=False)['input_ids'][1]
            stop_words_ids = [torch.tensor([self.tokenizer.eos_token_id]).to(self.device),
                            torch.tensor([id_tre_jin]).to(self.device),
                            torch.tensor([id_two_jin, id_one_jin]).to(self.device),
                            torch.tensor([id_one_jin, id_two_jin]).to(self.device)]
            
        self.stopping_criteria = StoppingCriteriaList([StoppingCriteriaSub(stops=stop_words_ids)])
        # 设置其他参数
        self.num_video_query_token = model_cfg.num_video_query_token
        self.num_audio_query_token = model_cfg.num_audio_query_token
        self.num_multi_query_token = model_cfg.num_multi_query_token
        self.num_image_query_token = model_cfg.num_image_query_token


    def to_token_ids(self, text, max_length):
        # 【Gemma 3 优化】官方建议显式添加 BOS token 以提升生成质量和对话状态管理
        add_special_tokens = True if self.llama_model_name in ['Gemma3'] else False
        input_ids = self.tokenizer(text,
                                   return_tensors="pt",
                                   padding="longest",
                                   max_length=max_length,
                                   truncation=True,
                                   add_special_tokens=add_special_tokens).input_ids[0]
        return input_ids

    def replace_token_for_multimodal(self, prompt):
        replace_token = config.DEFAULT_FRAME_PATCH_TOKEN * self.num_video_query_token
        prompt = prompt.replace(config.DEFAULT_FRAME_PATCH_TOKEN, replace_token)
        replace_token = config.DEFAULT_FACE_PATCH_TOKEN * self.num_video_query_token
        prompt = prompt.replace(config.DEFAULT_FACE_PATCH_TOKEN, replace_token)
        replace_token = config.DEFAULT_AUDIO_PATCH_TOKEN * self.num_audio_query_token
        prompt = prompt.replace(config.DEFAULT_AUDIO_PATCH_TOKEN, replace_token)
        replace_token = config.DEFAULT_MULTI_PATCH_TOKEN * self.num_multi_query_token
        prompt = prompt.replace(config.DEFAULT_MULTI_PATCH_TOKEN, replace_token)
        replace_token = config.DEFAULT_IMAGE_PATCH_TOKEN * self.num_image_query_token
        prompt = prompt.replace(config.DEFAULT_IMAGE_PATCH_TOKEN, replace_token)
        return prompt
   
    def postprocess_audio(self, sample_data):
        if sample_data['audio'] is None:
            return None, None
        
        audio = sample_data['audio'].unsqueeze(0).to(self.device)
        raw_audio = sample_data['raw_audio'].unsqueeze(0).to(self.device)
        audio_hiddens, audio_llms = self.model.encode_audio_merge(audio, raw_audio)
        return audio_hiddens, audio_llms

    def postprocess_face(self, sample_data):
        if sample_data['face'] is None:
            return None, None
        
        face = sample_data['face'].unsqueeze(0).to(self.device) # [1, 3, 8, 224, 224]
        raw_face = sample_data['raw_face'].unsqueeze(0).to(self.device) # [1, 3, 8, 224, 224]
        face_hiddens, face_llms = self.model.encode_video_merge(face, raw_face)
        return face_hiddens, face_llms
    
    def postprocess_frame(self, sample_data):
        if sample_data['frame'] is None:
            return None, None
        
        video = sample_data['frame'].unsqueeze(0).to(self.device) # [1, 3, 8, 224, 224]
        raw_video = sample_data['raw_frame'].unsqueeze(0).to(self.device) # [1, 3, 8, 224, 224]
        frame_hiddens, frame_llms = self.model.encode_video_merge(video, raw_video)
        return frame_hiddens, frame_llms

    def postprocess_image(self, sample_data):
        if sample_data['image'] is None:
            return None, None
        
        image = sample_data['image'].unsqueeze(0).to(self.device) # [1, 3, 8, 224, 224]
        raw_image = sample_data['raw_image'].unsqueeze(0).to(self.device) # [1, 3, 8, 224, 224]
        image_hiddens, image_llms = self.model.encode_image_merge(image, raw_image)
        return image_hiddens, image_llms


    def postprocess_multi(self, video_hiddens, audio_hiddens):
        if video_hiddens is None or audio_hiddens is None:
            return None, None

        multi_hiddens, multi_llms = self.model.encode_multi_merge(video_hiddens, audio_hiddens)
        return multi_hiddens, multi_llms

    
    # 整体过程就是在模拟inference过程 => 尝试完全按照 training 的方式进行读写
    # 【可复现性】do_sample=False 使用贪婪解码，确保输出确定
    def answer_sample(self, prompt, img_list, num_beams=1, temperature=0.0, do_sample=False, top_p=0.9,
                    max_new_tokens=1000, min_length=1, max_length=2000, repetition_penalty=1.0, length_penalty=1.0,
                    return_entropy=False):
        
        # 【修复】对于 Qwen/Llama (Vicuna 格式)，确保 prompt 末尾有正确的对话格式
        # 否则模型会认为对话已结束，直接输出 ### 停止符
        if not self.llama_model_name.startswith('Gemma'):
            prompt_stripped = prompt.rstrip()
            needs_suffix = not (prompt_stripped.endswith('Assistant:') or prompt_stripped.endswith('###Assistant:'))
            
            if needs_suffix:
                if '###' in prompt:
                    prompt = prompt_stripped + "\n###Assistant:"
                else:
                    prompt = prompt_stripped + "\n###Assistant:"
        
        IMAGE_PATCH_TOKEN_ID = self.tokenizer.get_vocab()[config.DEFAULT_IMAGE_PATCH_TOKEN]
        AUDIO_PATCH_TOKEN_ID = self.tokenizer.get_vocab()[config.DEFAULT_AUDIO_PATCH_TOKEN]
        FRAME_PATCH_TOKEN_ID = self.tokenizer.get_vocab()[config.DEFAULT_FRAME_PATCH_TOKEN]
        FACE_PATCH_TOKEN_ID  = self.tokenizer.get_vocab()[config.DEFAULT_FACE_PATCH_TOKEN]
        MULTI_PATCH_TOKEN_ID = self.tokenizer.get_vocab()[config.DEFAULT_MULTI_PATCH_TOKEN]

        ###### step1: => (input_id, attention_mask) 
        ## replace and add
        prompt = self.replace_token_for_multimodal(prompt)
        input_id = self.to_token_ids(prompt, max_length)
        # print (prompt)  # 注释掉避免控制台输出过多
        
        ## length limits
        current_max_len = len(input_id) + max_new_tokens
        if current_max_len - max_length > 0:
            print('Warning: The number of tokens in current conversation exceeds the max length. '
                'The model will not see the contexts outside the range.')
        begin_idx = max(0, current_max_len - max_length)
        input_id = input_id[begin_idx:]
        attention_mask=input_id.ne(self.tokenizer.pad_token_id).to(self.device)

        ###### step2: (input_ids) => (inputs_embeds)
        temp_input_id = copy.deepcopy(input_id).to(self.device)
        temp_input_id[temp_input_id == FRAME_PATCH_TOKEN_ID] = 0
        temp_input_id[temp_input_id == FACE_PATCH_TOKEN_ID]  = 0
        temp_input_id[temp_input_id == AUDIO_PATCH_TOKEN_ID] = 0
        temp_input_id[temp_input_id == MULTI_PATCH_TOKEN_ID] = 0
        temp_input_id[temp_input_id == IMAGE_PATCH_TOKEN_ID] = 0
        # 【架构兼容】使用 Transformers 标准 API 而非硬编码路径，支持 Gemma3/Llama/Qwen 等不同架构
        cur_input_embeds = self.model.llama_model.get_input_embeddings()(temp_input_id)
        cur_input_ids = input_id
        
        # replace <ImageHere>, <AudioHere>, <FrameHere>, <FaceHere> with features
        cur_idx = 0
        replaced_count = 0
        
        for (patch_token_id, query_token_number, embeds) in [(FRAME_PATCH_TOKEN_ID, self.num_video_query_token, img_list['frame']),
                                                            (FACE_PATCH_TOKEN_ID,  self.num_video_query_token, img_list['face']),
                                                            (AUDIO_PATCH_TOKEN_ID, self.num_audio_query_token, img_list['audio']),
                                                            (MULTI_PATCH_TOKEN_ID, self.num_multi_query_token, img_list['multi']),
                                                            (IMAGE_PATCH_TOKEN_ID, self.num_image_query_token, img_list['image']),
                                                            ]:
            if (cur_input_ids == patch_token_id).sum() != 0:
                assert embeds is not None, f'Some input info is missing.'
                cur_features = embeds[cur_idx]
                # print(f'  - 替换 token_id={patch_token_id}, 特征形状={cur_features.shape}')  # 调试信息
                replaced_count += 1
                if (cur_input_ids == patch_token_id).sum() != query_token_number:
                    raise ValueError("The number of audio patch tokens should be the same as the number of audio patches.")
                masked_indices = torch.where(cur_input_ids == patch_token_id)[0]
                mask_index_start = masked_indices[0]
                if (masked_indices != torch.arange(mask_index_start, mask_index_start+query_token_number, device=masked_indices.device, dtype=masked_indices.dtype)).any():
                    raise ValueError("The image patch tokens should be consecutive.")
                cur_input_embeds = torch.cat((cur_input_embeds[:mask_index_start], 
                                            cur_features, 
                                            cur_input_embeds[mask_index_start+query_token_number:]), dim=0)
        
        # print(f'  共替换 {replaced_count} 个模态特征')  # 调试信息
                    
        cur_input_embeds = cur_input_embeds.unsqueeze(0) 
        attention_mask = attention_mask.unsqueeze(0) 
        
        # 确保输入嵌入与模型权重类型一致（对 bfloat16 模型至关重要）
        cur_input_embeds = cur_input_embeds.to(self.model.llama_model.dtype)
        
        ###### step3: (inputs_embeds, attention_masks) => response
        # 【新增】固定采样种子，确保推理可复现
        if do_sample:
            torch.manual_seed(42)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(42)
        
        outputs = self.model.llama_model.generate(
            inputs_embeds=cur_input_embeds,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            stopping_criteria=self.stopping_criteria,
            num_beams=num_beams,
            do_sample=do_sample,
            min_length=min_length,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            length_penalty=length_penalty,
            temperature=temperature,
        )

        ###### step4: convert to batch samples 
        # maybe <bos> aaa <stop token> bbb <eos>
        response = self.tokenizer.decode(outputs[0], add_special_tokens=False)
        
        # 【修复】Qwen 的 bos_token/eos_token 可能为 None，需要检查
        if self.tokenizer.bos_token is not None and self.tokenizer.bos_token in response:
            response = response.split(self.tokenizer.bos_token)[1]
        if self.tokenizer.eos_token is not None and self.tokenizer.eos_token in response:
            response = response.split(self.tokenizer.eos_token)[0]
        
        # 根据模型类型进行不同的后处理
        if self.llama_model_name in ['Gemma3']:
            # Gemma 3: 移除 <start_of_turn>model 前缀和 <end_of_turn> 后缀
            if '<start_of_turn>model' in response:
                response = response.split('<start_of_turn>model')[-1]
            if '<end_of_turn>' in response:
                response = response.split('<end_of_turn>')[0]
            response = response.strip()
        else:
            # 其他模型（Llama/Qwen）：移除 ### 和 Assistant: 标记
            response = response.rsplit('###', 1)[0]  # split from stop tokens '###'
            response = response.split('Assistant:')[-1].strip()
        
        return response

    
    # 【批量推理】支持多样本并行生成，显著提升吞吐量
    def answer_batch(self, prompts, img_lists, num_beams=1, temperature=1.0, do_sample=True, top_p=0.9,
                     max_new_tokens=1000, min_length=1, max_length=2000, repetition_penalty=1.0, length_penalty=1.0):
        """
        批量处理多个样本的推理请求
        
        Args:
            prompts: List[str] - 多个 prompt
            img_lists: List[dict] - 每个样本的多模态特征字典列表
            其他参数同 answer_sample()
        
        Returns:
            List[str] - 每个样本的生成结果
        """
        batch_size = len(prompts)
        assert len(img_lists) == batch_size, "prompts 和 img_lists 数量必须一致"
        
        IMAGE_PATCH_TOKEN_ID = self.tokenizer.get_vocab()[config.DEFAULT_IMAGE_PATCH_TOKEN]
        AUDIO_PATCH_TOKEN_ID = self.tokenizer.get_vocab()[config.DEFAULT_AUDIO_PATCH_TOKEN]
        FRAME_PATCH_TOKEN_ID = self.tokenizer.get_vocab()[config.DEFAULT_FRAME_PATCH_TOKEN]
        FACE_PATCH_TOKEN_ID  = self.tokenizer.get_vocab()[config.DEFAULT_FACE_PATCH_TOKEN]
        MULTI_PATCH_TOKEN_ID = self.tokenizer.get_vocab()[config.DEFAULT_MULTI_PATCH_TOKEN]
        
        ###### Step1: 为每个样本构建 input_embeds 和 attention_mask
        all_input_embeds = []
        all_attention_masks = []
        
        for idx in range(batch_size):
            prompt = prompts[idx]
            img_list = img_lists[idx]
            
            # 替换多模态 token
            prompt = self.replace_token_for_multimodal(prompt)
            input_id = self.to_token_ids(prompt, max_length)
            
            # 【调试】打印第一个样本的 token 化后的信息（已注释）
            # if idx == 0:
            #     print(f'\n[DEBUG answer_batch] 第一个样本:')
            #     print(f'  input_id 长度: {len(input_id)}')
            #     print(f'  input_id 前20个: {input_id[:20].tolist()}')
            #     print(f'  prompt 结尾100字符: ...{prompt[-100:]}')
            
            # 长度限制
            current_max_len = len(input_id) + max_new_tokens
            if current_max_len - max_length > 0:
                begin_idx = max(0, current_max_len - max_length)
                input_id = input_id[begin_idx:]
            
            attention_mask = input_id.ne(self.tokenizer.pad_token_id).to(self.device)
            
            # 将 input_ids 转换为 embeddings
            temp_input_id = copy.deepcopy(input_id).to(self.device)
            temp_input_id[temp_input_id == FRAME_PATCH_TOKEN_ID] = 0
            temp_input_id[temp_input_id == FACE_PATCH_TOKEN_ID]  = 0
            temp_input_id[temp_input_id == AUDIO_PATCH_TOKEN_ID] = 0
            temp_input_id[temp_input_id == MULTI_PATCH_TOKEN_ID] = 0
            temp_input_id[temp_input_id == IMAGE_PATCH_TOKEN_ID] = 0
            
            cur_input_embeds = self.model.llama_model.get_input_embeddings()(temp_input_id)
            cur_input_ids = input_id
            
            # 替换多模态 token 为对应特征
            for (patch_token_id, query_token_number, embeds) in [
                (FRAME_PATCH_TOKEN_ID, self.num_video_query_token, img_list['frame']),
                (FACE_PATCH_TOKEN_ID,  self.num_video_query_token, img_list['face']),
                (AUDIO_PATCH_TOKEN_ID, self.num_audio_query_token, img_list['audio']),
                (MULTI_PATCH_TOKEN_ID, self.num_multi_query_token, img_list['multi']),
                (IMAGE_PATCH_TOKEN_ID, self.num_image_query_token, img_list['image']),
            ]:
                if (cur_input_ids == patch_token_id).sum() != 0:
                    assert embeds is not None, f'某些输入模态数据缺失'
                    cur_features = embeds[0]  # [query_tokens, hidden_dim]
                    
                    if (cur_input_ids == patch_token_id).sum() != query_token_number:
                        raise ValueError("多模态 token 数量不匹配")
                    
                    masked_indices = torch.where(cur_input_ids == patch_token_id)[0]
                    mask_index_start = masked_indices[0]
                    
                    if (masked_indices != torch.arange(mask_index_start, mask_index_start+query_token_number, 
                                                       device=masked_indices.device, dtype=masked_indices.dtype)).any():
                        raise ValueError("多模态 token 必须连续")
                    
                    cur_input_embeds = torch.cat((
                        cur_input_embeds[:mask_index_start], 
                        cur_features, 
                        cur_input_embeds[mask_index_start+query_token_number:]
                    ), dim=0)
            
            all_input_embeds.append(cur_input_embeds)
            all_attention_masks.append(attention_mask)
        
        ###### Step2: Batch padding（使用 left padding，因为是 decoder-only 模型）
        max_seq_len = max([embeds.shape[0] for embeds in all_input_embeds])
        embed_dim = all_input_embeds[0].shape[-1]
        
        # 初始化批量张量
        batch_input_embeds = torch.zeros(batch_size, max_seq_len, embed_dim, 
                                          dtype=self.model.llama_model.dtype, device=self.device)
        batch_attention_mask = torch.zeros(batch_size, max_seq_len, dtype=torch.long, device=self.device)
        
        # Left padding（对于 decoder-only 模型，padding 应该在左侧）
        for idx in range(batch_size):
            cur_len = all_input_embeds[idx].shape[0]
            padding_len = max_seq_len - cur_len
            
            # 将实际内容放在右侧
            batch_input_embeds[idx, padding_len:] = all_input_embeds[idx]
            batch_attention_mask[idx, padding_len:] = all_attention_masks[idx]
        
        ###### Step3: 批量生成
        # 【新增】固定采样种子，确保推理可复现
        if do_sample:
            torch.manual_seed(42)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(42)
        
        outputs = self.model.llama_model.generate(
            inputs_embeds=batch_input_embeds,
            attention_mask=batch_attention_mask,
            max_new_tokens=max_new_tokens,
            stopping_criteria=self.stopping_criteria,
            num_beams=num_beams,
            do_sample=do_sample,
            min_length=min_length,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            length_penalty=length_penalty,
            temperature=temperature,
        )
        
        ###### Step4: 批量解码
        responses = []
        for idx in range(batch_size):
            response = self.tokenizer.decode(outputs[idx], add_special_tokens=False)
            
            # 【修复】Qwen 的 bos_token/eos_token 可能为 None，需要检查
            if self.tokenizer.bos_token is not None and self.tokenizer.bos_token in response:
                response = response.split(self.tokenizer.bos_token)[1]
            if self.tokenizer.eos_token is not None and self.tokenizer.eos_token in response:
                response = response.split(self.tokenizer.eos_token)[0]
            
            # 根据模型类型进行不同的后处理
            if self.llama_model_name in ['Gemma3']:
                # Gemma 3: 移除 <start_of_turn>model 前缀和 <end_of_turn> 后缀
                if '<start_of_turn>model' in response:
                    response = response.split('<start_of_turn>model')[-1]
                if '<end_of_turn>' in response:
                    response = response.split('<end_of_turn>')[0]
                response = response.strip()
            else:
                # 其他模型（Llama/Qwen）：移除 ### 和 Assistant: 标记
                response = response.rsplit('###', 1)[0]
                response = response.split('Assistant:')[-1].strip()
            
            responses.append(response)
        
        return responses
