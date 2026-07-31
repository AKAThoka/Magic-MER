"""
AffectGPT Gemma 3 适配模块
专门针对 Gemma 3 模型的架构适配，主要处理：
1. LoRA 目标模块的差异
2. 嵌入层访问方式的兼容性
3. 新增特殊 Token 的初始化
4. 前向传播中的模型结构差异
"""
import copy
import torch
import torch.nn as nn
from my_affectgpt.common.registry import registry
from my_affectgpt.models.affectgpt import AffectGPT
import config


@registry.register_model("affectgpt_gemma")
class AffectGPTGemma(AffectGPT):
    """
    AffectGPT 的 Gemma 3 特化版本
    继承自 AffectGPT 基类，重写关键方法以适配 Gemma 3 的架构特性
    
    关键差异：
    - 使用 get_input_embeddings() 而非硬编码的 .model.embed_tokens
    - LoRA 配置适配 Gemma 3 的层结构
    - 词表扩展后的嵌入初始化
    - Gemma 3 的 hidden_size 需要从 text_config 中获取
    """
    
    PRETRAINED_MODEL_CONFIG_DICT = {
        "pretrain_vicuna": "configs/models/affectgpt.yaml",
    }
    
    def _get_llm_hidden_size(self):
        """
        获取 LLM 的隐藏层大小
        Gemma 3 是多模态模型，hidden_size 在 text_config 中
        而不是直接在 config 中
        """
        config = self.llama_model.config
        # Gemma 3 多模态模型：hidden_size 在 text_config 中
        if hasattr(config, 'text_config') and hasattr(config.text_config, 'hidden_size'):
            return config.text_config.hidden_size
        # 普通 LLM（如 Llama）：hidden_size 直接在 config 中
        elif hasattr(config, 'hidden_size'):
            return config.hidden_size
        # PEFT 包装后的模型：检查 base_model_config
        elif hasattr(self.llama_model, 'base_model') and hasattr(self.llama_model.base_model, 'config'):
            base_config = self.llama_model.base_model.config
            if hasattr(base_config, 'text_config') and hasattr(base_config.text_config, 'hidden_size'):
                return base_config.text_config.hidden_size
            elif hasattr(base_config, 'hidden_size'):
                return base_config.hidden_size
        raise AttributeError(f"无法从配置中获取 hidden_size。Config 类型: {type(config)}")
    
    def __init__(
        self,
        visual_encoder_name,
        acoustic_encoder_name,
        llama_model_name,
        frozen_video_proj,
        frozen_video_Qformer,
        frozen_audio_Qformer,
        frozen_audio_proj,
        frozen_llm,
        lora_r,
        num_video_query_token,
        num_audio_query_token,
        num_multi_query_token,
        num_image_query_token,
        frozen_multi_Qformer,
        frozen_multi_llama_proj,
        multi_fusion_type,
        video_fusion_type,
        audio_fusion_type,
        image_fusion_type,
    ):
        # 注意：不能直接调用 super().__init__()，因为需要在初始化时处理 Gemma 特有逻辑
        # 先调用 Blip2Base 的初始化
        nn.Module.__init__(self)
        
        print('====== Loading LLM (Gemma 3) ======')
        from transformers import AutoTokenizer, AutoModelForCausalLM
        from my_affectgpt.models.tokenizer import load_tokenizer_from_LLM
        
        self.llama_model_name = llama_model_name    
        self.llama_tokenizer = load_tokenizer_from_LLM(llama_model_name)
        
        # 获取多模态占位符的 Token ID
        DEFAULT_IMAGE_PATCH_TOKEN = config.DEFAULT_IMAGE_PATCH_TOKEN
        DEFAULT_AUDIO_PATCH_TOKEN = config.DEFAULT_AUDIO_PATCH_TOKEN
        DEFAULT_FRAME_PATCH_TOKEN = config.DEFAULT_FRAME_PATCH_TOKEN
        DEFAULT_FACE_PATCH_TOKEN  = config.DEFAULT_FACE_PATCH_TOKEN
        DEFAULT_MULTI_PATCH_TOKEN = config.DEFAULT_MULTI_PATCH_TOKEN
        self.IMAGE_PATCH_TOKEN_ID = self.llama_tokenizer.get_vocab()[DEFAULT_IMAGE_PATCH_TOKEN]
        self.AUDIO_PATCH_TOKEN_ID = self.llama_tokenizer.get_vocab()[DEFAULT_AUDIO_PATCH_TOKEN]
        self.FRAME_PATCH_TOKEN_ID = self.llama_tokenizer.get_vocab()[DEFAULT_FRAME_PATCH_TOKEN]
        self.FACE_PATCH_TOKEN_ID  = self.llama_tokenizer.get_vocab()[DEFAULT_FACE_PATCH_TOKEN]
        self.MULTI_PATCH_TOKEN_ID = self.llama_tokenizer.get_vocab()[DEFAULT_MULTI_PATCH_TOKEN]
        
        # 加载 Gemma 3 模型（强制使用 eager 注意力实现，官方推荐配置）
        from transformers import AutoConfig
        model_path = config.PATH_TO_LLM[llama_model_name]
        model_config = AutoConfig.from_pretrained(model_path)
        model_config._attn_implementation = "eager" # 显式强制 eager
        
        self.llama_model = AutoModelForCausalLM.from_pretrained(
            model_path,
            config=model_config,
            torch_dtype=torch.bfloat16,  # Gemma 3 官方推荐使用 bfloat16 而非 float16
            attn_implementation='eager' # 双重保险
        )
        
        print(f"====== LLM 注意力实现确认: {self.llama_model.config._attn_implementation} ======")
        
        # 扩展词表以包含新增的多模态特殊 Token
        original_vocab_size = len(self.llama_tokenizer) - 5  # 减去新增的 5 个特殊 Token
        self.llama_model.resize_token_embeddings(len(self.llama_tokenizer))
        
        # 初始化新增 Token 的嵌入（使用现有词表的均值）
        with torch.no_grad():
            input_embeddings = self.llama_model.get_input_embeddings()
            output_embeddings = self.llama_model.get_output_embeddings()
            
            # 计算原始词表的均值
            input_embeddings_avg = input_embeddings.weight[:original_vocab_size].mean(dim=0)
            output_embeddings_avg = output_embeddings.weight[:original_vocab_size].mean(dim=0)
            
            # 用均值初始化新增的 Token
            input_embeddings.weight[original_vocab_size:] = input_embeddings_avg
            output_embeddings.weight[original_vocab_size:] = output_embeddings_avg
        
        print(f'扩展词表: {original_vocab_size} -> {len(self.llama_tokenizer)}')
        
        # 冻结基础模型参数
        for name, param in self.llama_model.named_parameters():
            param.requires_grad = False
        
        print('====== Using LoRA on LLM (Gemma 3 适配) ======')
        from peft import get_peft_model, LoraConfig, TaskType
        
        # Gemma 3 的 LoRA 目标模块
        try:
            # Gemma 3 多模态模型：语言模型在 model.language_model.model.layers
            if hasattr(self.llama_model, 'language_model') and hasattr(self.llama_model.language_model, 'model'):
                layer_num = len(self.llama_model.language_model.model.layers)
                layer_prefix = 'language_model.model.layers.'
                print(f'检测到 Gemma 3 多模态结构，共 {layer_num} 层')
            # 普通 LLM 结构：model.model.layers
            elif hasattr(self.llama_model, 'model') and hasattr(self.llama_model.model, 'layers'):
                layer_num = len(self.llama_model.model.layers)
                layer_prefix = 'model.layers.'
                print(f'检测到标准 LLM 结构，共 {layer_num} 层')
            else:
                raise AttributeError("无法自动检测模型层结构")
            
            # Gemma 3 的注意力和 MLP 模块名称
            target_modules = [
                layer_prefix + str(i) + '.' + k 
                for i in range(layer_num) 
                for k in ["self_attn.q_proj", "self_attn.k_proj", 
                         "self_attn.v_proj", "self_attn.o_proj", 
                         "mlp.gate_proj", "mlp.down_proj", "mlp.up_proj"]
            ]
            
        except Exception as e:
            print(f'警告：无法自动检测 LoRA 目标模块，使用默认配置。错误: {e}')
            # 回退到简单配置（PEFT 会自动匹配所有包含这些名称的模块）
            target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", 
                            "gate_proj", "down_proj", "up_proj"]
        
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM, 
            inference_mode=False, 
            r=lora_r, 
            lora_alpha=32, 
            lora_dropout=0.05, 
            target_modules=target_modules
        )
        self.llama_model = get_peft_model(self.llama_model, peft_config)
        
        if frozen_llm:
            for param in self.llama_model.parameters():
                param.requires_grad = False
            print('freeze: Gemma Model')
        else:
            print('trainable: Gemma Model (LoRA 可训练)')
        self.llama_model.print_trainable_parameters()
        
        # ========== 以下部分完全复用父类的初始化逻辑 ==========
        # 直接调用父类的方法来初始化视觉/音频编码器和投影层
        self._init_vision_audio_modules(
            visual_encoder_name, acoustic_encoder_name,
            frozen_video_proj, frozen_video_Qformer,
            frozen_audio_Qformer, frozen_audio_proj,
            frozen_multi_Qformer, frozen_multi_llama_proj,
            num_video_query_token, num_audio_query_token,
            num_multi_query_token, num_image_query_token,
            multi_fusion_type, video_fusion_type,
            audio_fusion_type, image_fusion_type
        )
        
        print('====== AffectGPTGemma 初始化完成 ======')
    
    def _init_vision_audio_modules(
        self,
        visual_encoder_name,
        acoustic_encoder_name,
        frozen_video_proj,
        frozen_video_Qformer,
        frozen_audio_Qformer,
        frozen_audio_proj,
        frozen_multi_Qformer,
        frozen_multi_llama_proj,
        num_video_query_token,
        num_audio_query_token,
        num_multi_query_token,
        num_image_query_token,
        multi_fusion_type,
        video_fusion_type,
        audio_fusion_type,
        image_fusion_type
    ):
        """
        初始化视觉和音频相关模块（与父类逻辑完全一致）
        单独提取出来是为了避免在 __init__ 中重复代码
        """
        print('====== Loading Image Encoder ======')
        self.image_fusion_type = image_fusion_type
        self.num_image_query_token = num_image_query_token
        self.visual_encoder = registry.get_visual_encoder_class(visual_encoder_name)()
        self.image_llama_proj = nn.Linear(
            self.visual_encoder.hidden_size, 
            self._get_llm_hidden_size()
        )
        
        print('====== Loading Video Q-Former ======')
        self.video_fusion_type = video_fusion_type
        self.num_video_query_token = num_video_query_token
        
        if self.video_fusion_type == 'qformer':
            from transformers import BertConfig
            from my_affectgpt.models.Qformer import BertLMHeadModel
            
            self.video_frame_position_embedding = nn.Embedding(32, self.visual_encoder.hidden_size)
            self.video_Qformer, self.video_query_tokens = self.init_video_Qformer(
                num_query_token=num_video_query_token,
                vision_width=self.visual_encoder.hidden_size, 
                num_hidden_layers=2
            )
            self.video_Qformer.cls = None
            self.video_Qformer.bert.embeddings.word_embeddings = None
            self.video_Qformer.bert.embeddings.position_embeddings = None
            for layer in self.video_Qformer.bert.encoder.layer:
                layer.output = None
                layer.intermediate = None
            
            if frozen_video_Qformer:
                for name, param in self.video_Qformer.named_parameters():
                    param.requires_grad = False
                for name, param in self.video_frame_position_embedding.named_parameters():
                    param.requires_grad = False
                self.video_query_tokens.requires_grad = False
                print('freeze: video_Qformer')
            else:
                for name, param in self.video_Qformer.named_parameters():
                    param.requires_grad = True
                for name, param in self.video_frame_position_embedding.named_parameters():
                    param.requires_grad = True
                self.video_query_tokens.requires_grad = True
                print('trainable: video_Qformer')
            video_hidden_size = self.video_Qformer.config.hidden_size
        elif self.video_fusion_type == 'mean':
            video_hidden_size = self.visual_encoder.hidden_size
        elif self.video_fusion_type == 'attention':
            self.video_attention_mlp = nn.Linear(self.visual_encoder.hidden_size, 1)
            video_hidden_size = self.visual_encoder.hidden_size
        
        print(f'====== Loading Video LLAMA proj ======')
        self.affectgpt_proj = nn.Linear(video_hidden_size, self._get_llm_hidden_size())
        if frozen_video_proj:
            for name, param in self.affectgpt_proj.named_parameters():
                param.requires_grad = False
            print('freeze: Video Q-Former LLaMA proj')
        else:
            for name, param in self.affectgpt_proj.named_parameters():
                param.requires_grad = True
            print('trainable: Video Q-Former LLaMA proj')
        
        print(f'====== Loading Audio Encoder ======')
        self.acoustic_encoder = registry.get_acoustic_encoder_class(acoustic_encoder_name)()
        
        print('====== Loading Audio Q-Former ======')
        self.audio_fusion_type = audio_fusion_type
        self.num_audio_query_token = num_audio_query_token
        
        if self.audio_fusion_type == 'qformer':
            from transformers import BertConfig
            from my_affectgpt.models.Qformer import BertLMHeadModel
            
            self.audio_position_embedding = nn.Embedding(8, self.acoustic_encoder.hidden_size)
            self.audio_Qformer, self.audio_query_tokens = self.init_video_Qformer(
                num_query_token=self.num_audio_query_token,
                vision_width=self.acoustic_encoder.hidden_size, 
                num_hidden_layers=2
            )
            self.audio_Qformer.cls = None
            self.audio_Qformer.bert.embeddings.word_embeddings = None
            self.audio_Qformer.bert.embeddings.position_embeddings = None
            for layer in self.audio_Qformer.bert.encoder.layer:
                layer.output = None
                layer.intermediate = None
            
            if frozen_audio_Qformer:
                for name, param in self.audio_Qformer.named_parameters():
                    param.requires_grad = False
                self.audio_query_tokens.requires_grad = False
                for name, param in self.audio_position_embedding.named_parameters():
                    param.requires_grad = False
                print('freeze: audio_Qformer')
            else:
                for name, param in self.audio_Qformer.named_parameters():
                    param.requires_grad = True
                self.audio_query_tokens.requires_grad = True
                for name, param in self.audio_position_embedding.named_parameters():
                    param.requires_grad = True
                print('trainable: audio_Qformer')
            audio_hidden_size = self.audio_Qformer.config.hidden_size
        elif self.audio_fusion_type == 'mean':
            audio_hidden_size = self.acoustic_encoder.hidden_size
        elif self.audio_fusion_type == 'attention':
            self.audio_attention_mlp = nn.Linear(self.acoustic_encoder.hidden_size, 1)
            audio_hidden_size = self.acoustic_encoder.hidden_size
        
        print('====== Loading audio_llama_proj ======')
        self.audio_llama_proj = nn.Linear(
            audio_hidden_size, 
            self._get_llm_hidden_size()
        )
        if frozen_audio_proj:
            for name, param in self.audio_llama_proj.named_parameters():
                param.requires_grad = False
            print('freeze: Audio Q-Former LLaMA proj')
        else:
            for name, param in self.audio_llama_proj.named_parameters():
                param.requires_grad = True
            print('trainable: Audio Q-Former LLaMA proj')
        
        print('====== Loading Multi Q-Former (pre-fusion) ======')
        self.num_multi_query_token = num_multi_query_token
        self.multi_fusion_type = multi_fusion_type
        self.max_hidden_size = max(self.acoustic_encoder.hidden_size, self.visual_encoder.hidden_size)
        self.multi_audio_embs = nn.Linear(self.acoustic_encoder.hidden_size, self.max_hidden_size)
        self.multi_video_embs = nn.Linear(self.visual_encoder.hidden_size, self.max_hidden_size)
        
        if self.multi_fusion_type == 'qformer':
            from transformers import BertConfig
            from my_affectgpt.models.Qformer import BertLMHeadModel
            
            self.multi_position_embedding = nn.Embedding(264, self.max_hidden_size)
            self.multi_Qformer, self.multi_query_tokens = self.init_video_Qformer(
                num_query_token=self.num_multi_query_token,
                vision_width=self.max_hidden_size, 
                num_hidden_layers=2
            )
            self.multi_Qformer.cls = None
            self.multi_Qformer.bert.embeddings.word_embeddings = None
            self.multi_Qformer.bert.embeddings.position_embeddings = None
            for layer in self.multi_Qformer.bert.encoder.layer:
                layer.output = None
                layer.intermediate = None
            
            if frozen_multi_Qformer:
                for name, param in self.multi_Qformer.named_parameters():
                    param.requires_grad = False
                self.multi_query_tokens.requires_grad = False
                for name, param in self.multi_position_embedding.named_parameters():
                    param.requires_grad = False
                print('freeze: multi_Qformer')
            else:
                for name, param in self.multi_Qformer.named_parameters():
                    param.requires_grad = True
                self.multi_query_tokens.requires_grad = True
                for name, param in self.multi_position_embedding.named_parameters():
                    param.requires_grad = True
                print('trainable: multi_Qformer')
            multi_hidden_size = self.multi_Qformer.config.hidden_size
        elif self.multi_fusion_type == 'attention':
            self.attention_mlp = nn.Linear(self.max_hidden_size * 2, self.max_hidden_size)
            self.fc_att = nn.Linear(self.max_hidden_size, 2)
            multi_hidden_size = self.max_hidden_size
        
        print('====== Loading multi_llama_proj ======')
        self.multi_llama_proj = nn.Linear(multi_hidden_size, self._get_llm_hidden_size())
        if frozen_multi_llama_proj:
            for name, param in self.multi_llama_proj.named_parameters():
                param.requires_grad = False
            print('freeze: Multi Q-Former LLaMA proj')
        else:
            for name, param in self.multi_llama_proj.named_parameters():
                param.requires_grad = True
            print('trainable: Multi Q-Former LLaMA proj')
    
    def forward(self, samples):
        """
        重写 forward 方法以适配 Gemma 3 的嵌入层访问方式
        主要差异：使用 get_input_embeddings() 替代硬编码的 .model.model.embed_tokens
        """
        self.face_or_frame = samples['face_or_frame']
        frame_llms, face_llms, audio_llms, image_llms, multi_llms = None, None, None, None, None
        
        # 编码各模态输入（与父类逻辑相同）
        if 'frames' in samples: 
            frame_hiddens, frame_llms = self.encode_video_merge(samples['frames'], samples['raw_frames'])
        if 'faces' in samples: 
            face_hiddens, face_llms = self.encode_video_merge(samples['faces'], samples['raw_faces'])
        if 'audios' in samples: 
            audio_hiddens, audio_llms = self.encode_audio_merge(samples['audios'], samples['raw_audios'])
        if 'images' in samples: 
            image_hiddens, image_llms = self.encode_image_merge(samples['images'], samples['raw_images'])
        if (samples['input_ids'][0] == self.MULTI_PATCH_TOKEN_ID).sum() != 0:
            if self.face_or_frame.startswith('multiface'):
                multi_hiddens, multi_llms = self.encode_multi_merge(face_hiddens, audio_hiddens)
            if self.face_or_frame.startswith('multiframe'):
                multi_hiddens, multi_llms = self.encode_multi_merge(frame_hiddens, audio_hiddens)

        # 替换多模态占位符为 0
        input_ids = samples['input_ids']
        temp_input_ids = copy.deepcopy(input_ids)
        temp_input_ids[temp_input_ids == self.FRAME_PATCH_TOKEN_ID] = 0
        temp_input_ids[temp_input_ids == self.FACE_PATCH_TOKEN_ID] = 0
        temp_input_ids[temp_input_ids == self.AUDIO_PATCH_TOKEN_ID] = 0
        temp_input_ids[temp_input_ids == self.MULTI_PATCH_TOKEN_ID] = 0
        temp_input_ids[temp_input_ids == self.IMAGE_PATCH_TOKEN_ID] = 0
        
        # ========== 关键修改：使用通用方法访问嵌入层 ==========
        embed_tokens = self.llama_model.get_input_embeddings()
        temp_input_embedding = embed_tokens(temp_input_ids)

        # 将多模态特征插入到对应位置（逻辑与父类完全相同）
        cur_idx = 0
        new_input_embeds = []
        for cur_input_ids, cur_input_embeds in zip(input_ids, temp_input_embedding):
            for (patch_token_id, query_token_number, embeds) in [
                (self.FRAME_PATCH_TOKEN_ID, self.num_video_query_token, frame_llms),
                (self.FACE_PATCH_TOKEN_ID, self.num_video_query_token, face_llms),
                (self.AUDIO_PATCH_TOKEN_ID, self.num_audio_query_token, audio_llms),
                (self.MULTI_PATCH_TOKEN_ID, self.num_multi_query_token, multi_llms),
                (self.IMAGE_PATCH_TOKEN_ID, self.num_image_query_token, image_llms),
            ]:
                if (cur_input_ids == patch_token_id).sum() != 0:
                    assert embeds is not None, f'Some input info is missing.'
                    cur_features = embeds[cur_idx]
                    if (cur_input_ids == patch_token_id).sum() != query_token_number:
                        raise ValueError("The number of patch tokens should be the same as the number of patches.")
                    masked_indices = torch.where(cur_input_ids == patch_token_id)[0]
                    mask_index_start = masked_indices[0]
                    if (masked_indices != torch.arange(mask_index_start, mask_index_start+query_token_number, 
                                                      device=masked_indices.device, dtype=masked_indices.dtype)).any():
                        raise ValueError("The patch tokens should be consecutive.")
                    cur_input_embeds = torch.cat((
                        cur_input_embeds[:mask_index_start], 
                        cur_features, 
                        cur_input_embeds[mask_index_start+query_token_number:]
                    ), dim=0)
            
            new_input_embeds.append(cur_input_embeds)
            cur_idx += 1
        inputs_embeds = torch.stack(new_input_embeds, dim=0)

        # 计算损失
        targets = samples['labels']
        attention_mask = samples['attention_masks']
        with self.maybe_autocast():
            outputs = self.llama_model(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                return_dict=True,
                labels=targets
            )
        loss = outputs.loss
        return {"loss": loss}

    @classmethod
    def from_config(cls, cfg):
        """
        从配置文件创建模型实例（与父类逻辑完全一致）
        """
        # 这几个参数必须设置
        visual_encoder_name   = cfg.get("visual_encoder", "xxx")
        acoustic_encoder_name = cfg.get("acoustic_encoder", "xxx")
        llama_model_name      = cfg.get("llama_model", "xxx")
        multi_fusion_type = cfg.get("multi_fusion_type", "attention")
        video_fusion_type = cfg.get("video_fusion_type", "qformer")
        audio_fusion_type = cfg.get("audio_fusion_type", "qformer")
        image_fusion_type = cfg.get("image_fusion_type", "token")

        # Audio/Video Q-Former
        frozen_video_Qformer    = cfg.get("frozen_video_Qformer", False)
        frozen_video_proj = cfg.get("frozen_video_proj", False)
        frozen_audio_Qformer    = cfg.get("frozen_audio_Qformer", False)
        frozen_audio_proj = cfg.get("frozen_audio_proj", False)
        frozen_multi_Qformer    = cfg.get("frozen_multi_Qformer", False)
        frozen_multi_llama_proj = cfg.get("frozen_multi_llama_proj", False)
        frozen_llm = cfg.get("frozen_llm", False)
        lora_r = cfg.get("lora_r", 16)

        # 这几个参数是默认的
        num_audio_query_token = cfg.get("num_audio_query_token", 'xxx')
        num_video_query_token = cfg.get("num_video_query_token", 'xxx')
        num_multi_query_token = cfg.get("num_multi_query_token", 'xxx')
        num_image_query_token = cfg.get("num_image_query_token", 'xxx')

        model = cls(
            visual_encoder_name=visual_encoder_name,
            acoustic_encoder_name=acoustic_encoder_name,
            llama_model_name=llama_model_name,
            frozen_video_proj=frozen_video_proj,
            frozen_audio_proj=frozen_audio_proj,
            frozen_multi_llama_proj=frozen_multi_llama_proj,
            frozen_video_Qformer=frozen_video_Qformer,
            frozen_audio_Qformer=frozen_audio_Qformer,
            frozen_multi_Qformer=frozen_multi_Qformer,
            frozen_llm=frozen_llm,
            lora_r=lora_r,
            num_video_query_token=num_video_query_token,
            num_audio_query_token=num_audio_query_token,
            num_multi_query_token=num_multi_query_token,
            num_image_query_token=num_image_query_token,
            multi_fusion_type=multi_fusion_type,
            video_fusion_type=video_fusion_type,
            audio_fusion_type=audio_fusion_type,
            image_fusion_type=image_fusion_type,
        )

        # priority: ckpt < ckpt_2 < ckpt_3 
        # => 后面的预训练权重会覆盖前面的预训练权重
        ckpt_path = cfg.get("ckpt", "")
        if ckpt_path:
            print("Load first Checkpoint: {}".format(ckpt_path))
            ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
            model.load_state_dict(ckpt['model'], strict=False)
            
        ckpt_path_2 = cfg.get("ckpt_2", "")
        if ckpt_path_2:
            print("Load second Checkpoint: {}".format(ckpt_path_2))
            ckpt = torch.load(ckpt_path_2, map_location="cpu", weights_only=True)
            model.load_state_dict(ckpt['model'], strict=False)

        ckpt_path_3 = cfg.get("ckpt_3", "")
        if ckpt_path_3:
            print("Load third Checkpoint: {}".format(ckpt_path_3))
            ckpt = torch.load(ckpt_path_3, map_location="cpu", weights_only=True)
            
            # 【修复】处理 Gemma 3 模型结构变化导致的 key 不匹配问题
            # 旧版本: llama_model.base_model.model.model.layers...
            # 新版本: llama_model.base_model.model.language_model.model.layers...
            ckpt_state_dict = ckpt['model']
            new_state_dict = {}
            key_mapping_count = 0
            
            for key, value in ckpt_state_dict.items():
                new_key = key
                # 检查是否需要映射 key（旧格式 -> 新格式）
                if 'llama_model.base_model.model.model.' in key:
                    new_key = key.replace(
                        'llama_model.base_model.model.model.',
                        'llama_model.base_model.model.language_model.model.'
                    )
                    key_mapping_count += 1
                new_state_dict[new_key] = value
            
            if key_mapping_count > 0:
                print(f"[INFO] 已映射 {key_mapping_count} 个 key (旧结构 -> 新结构)")
            
            # 【调试】检查 checkpoint 中的 LoRA 相关 key
            ckpt_keys = list(new_state_dict.keys())
            lora_keys = [k for k in ckpt_keys if 'lora' in k.lower()]
            print(f"[DEBUG] Checkpoint 中共有 {len(ckpt_keys)} 个 key, 其中 LoRA 相关 {len(lora_keys)} 个")
            if lora_keys:
                print(f"[DEBUG] LoRA key 示例: {lora_keys[:2]}")
            
            # 加载权重
            missing_keys, unexpected_keys = model.load_state_dict(new_state_dict, strict=False)
            
            # 【调试】打印加载结果
            lora_missing = [k for k in missing_keys if 'lora' in k.lower()]
            lora_unexpected = [k for k in unexpected_keys if 'lora' in k.lower()]
            print(f"[DEBUG] Missing keys: {len(missing_keys)} (LoRA: {len(lora_missing)})")
            print(f"[DEBUG] Unexpected keys: {len(unexpected_keys)} (LoRA: {len(lora_unexpected)})")
            if lora_missing:
                print(f"[WARNING] Missing LoRA keys 示例: {lora_missing[:3]}")
            if lora_unexpected:
                print(f"[WARNING] Unexpected LoRA keys 示例: {lora_unexpected[:3]}")
            
            # 验证 LoRA 权重是否成功加载
            if len(lora_missing) == 0 and len(lora_unexpected) == 0:
                print(f"[SUCCESS] 所有 LoRA 权重已成功加载！")
            else:
                print(f"[ERROR] LoRA 权重加载可能存在问题，请检查！")
        
        return model