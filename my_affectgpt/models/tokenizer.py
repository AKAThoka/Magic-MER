"""
Adapted from salesforce@LAVIS. Below is the original copyright:
 Copyright (c) 2023, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE_Lavis file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""
import contextlib
import logging
import os
import time
import datetime

import torch
import torch.nn as nn
import torch.distributed as dist
import torch.nn.functional as F

import my_affectgpt.common.dist_utils as dist_utils
from my_affectgpt.common.dist_utils import download_cached_file
from my_affectgpt.common.utils import is_url
from my_affectgpt.common.logger import MetricLogger
from my_affectgpt.models.base_model import BaseModel
from my_affectgpt.models.Qformer import BertConfig, BertLMHeadModel
from my_affectgpt.models.eva_vit import create_eva_vit_g
from transformers import AutoTokenizer
import config


## 在 AffectGPT 中，每个 LLM 都需要自己的 'eos', 'pad', 'bos'；否则模型会报错
def load_tokenizer_from_LLM(model_name):
    if model_name in ['Baichuan2']:
        tokenizer = AutoTokenizer.from_pretrained(config.PATH_TO_LLM[model_name], use_fast=False, trust_remote_code=True)
    else:
        tokenizer = AutoTokenizer.from_pretrained(config.PATH_TO_LLM[model_name], use_fast=False)
    
    # 特定模型的 BOS token 设置
    if model_name in ['Qwen2', 'Qwen25']: 
        tokenizer.bos_token='<|im_start|>'
    
    # Gemma 3 特殊处理：确保使用官方 BOS token
    if model_name in ['Gemma3']:
        # Gemma 3 使用 <bos> 作为 BOS token，并且需要在生成时自动添加
        tokenizer.add_bos_token = True  # 确保自动添加 BOS token
        # Gemma 的 pad_token 应该单独设置，不要和 eos_token 混淆
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
    else:
        tokenizer.pad_token = tokenizer.eos_token  # 其他模型：vicuna, llama2, llama3
    
    # 添加多模态占位符（所有模型通用）
    tokenizer.add_tokens([config.DEFAULT_IMAGE_PATCH_TOKEN], special_tokens=True)
    tokenizer.add_tokens([config.DEFAULT_AUDIO_PATCH_TOKEN], special_tokens=True)
    tokenizer.add_tokens([config.DEFAULT_FRAME_PATCH_TOKEN], special_tokens=True)
    tokenizer.add_tokens([config.DEFAULT_FACE_PATCH_TOKEN],  special_tokens=True)
    tokenizer.add_tokens([config.DEFAULT_MULTI_PATCH_TOKEN], special_tokens=True)
    
    return tokenizer
