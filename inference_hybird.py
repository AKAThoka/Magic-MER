import os
import time
import glob
import argparse
import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path

import torch
import torch.backends.cudnn as cudnn
from torch.utils.data import Dataset, DataLoader

import decord
decord.bridge.set_bridge('torch')
# 【推理优化】允许 decord 使用多线程解码，减少 CPU 瓶颈
os.environ['DECORD_EOF_RETRY'] = '1'

from my_affectgpt.tasks import *
from my_affectgpt.models import *
from my_affectgpt.runners import *
from my_affectgpt.processors import *
from my_affectgpt.datasets.builders import *
from my_affectgpt.common.config import Config
from my_affectgpt.common.dist_utils import get_rank
from my_affectgpt.common.registry import registry
from my_affectgpt.conversation.conversation_video import Chat
from my_affectgpt.datasets.builders.image_text_pair_builder import * # 加载所有dataset cls

import config
from toolkit.utils.read_files import *


# 采用的是这个文件下存储数量最多的 root
def search_for_ckpt_root(root_candidates):
    if len(root_candidates) == 0:
        return ''
    
    # 找到 files 最多的 root
    maxcount = 0
    targetroot = ''
    for root in root_candidates:
        count = len([path for path in os.listdir(root) if path.startswith('checkpoint_')])
        print (root, '==>', count)
        if count > maxcount:
            maxcount = count
            targetroot = root
    print ('================================================')
    print (f'Targetroot: epoch range: 0-{maxcount-1}')
    
    # 打印最后一个文件的创建时间 for targetroot
    last_file = sorted(glob.glob(targetroot + '/checkpoint*'))[-1]
    file_stat = Path(last_file).stat()
    creation_time = file_stat.st_ctime
    print("Targetroot: Last ckpt creation time:", datetime.fromtimestamp(creation_time))
    print ('================================================')
    return targetroot


# case1: 默认 => last epoch
# case2: 指定 inference_cfg.test_epoch == a; 那就只跑这个 epoch 下的结果
# case3: 指定 inference_cfg.test_epochs == a-b; 跑最后一个
def get_ckpt3_candidates(ckpt3_root, inference_cfg):
    
    if inference_cfg.test_epoch != 'xxx':
        cur_epoch = inference_cfg.test_epoch
        ckpts = glob.glob("%s/*%06d*.pth" %(ckpt3_root, int(cur_epoch)))
        assert len(ckpts) == 1, 'Error: (ckpt, epoch) combination is not exists or contain multiple candidates!'
        return [ckpts[0]]
    
    elif inference_cfg.test_epochs == 'xxx-xxx':
        last_ckpt = sorted(glob.glob("%s/*.pth" %(ckpt3_root)))[-1]
        last_epoch=  int(last_ckpt.split('_')[-3])
        assert last_epoch > 10, f'Error: too less training time to conduct automatic inference!'
        return [last_ckpt]
    
    else:
        start_epoch, end_epoch = inference_cfg.test_epochs.split('-')
        skip_epoch = int(inference_cfg.skip_epoch) 
        whole_ckpts = []
        for cur_epoch in range(int(start_epoch), int(end_epoch)+1):
            if cur_epoch % skip_epoch == 0:
                ckpts = glob.glob("%s/*%06d*.pth" %(ckpt3_root, int(cur_epoch)))
                assert len(ckpts) == 1, 'Error: (ckpt, epoch) combination is not exists or contain multiple candidates!'
                whole_ckpts.append(ckpts[0])
        return whole_ckpts


# 因为我们目前只处理 merbench，这些是 video 的，需要和原始训练数据中的 video 数据对应的 face_or_frame 一致
def get_face_or_frame(datasets_cfg, outside_face_or_frame):
    if outside_face_or_frame is not None:
        return outside_face_or_frame
    
    face_or_frame_candidates = []
    if 'mercaptionplus' in datasets_cfg:
        face_or_frame_candidates.append(datasets_cfg['mercaptionplus'].face_or_frame)
    if 'ovmerd' in datasets_cfg:
        face_or_frame_candidates.append(datasets_cfg['ovmerd'].face_or_frame)
    assert len(set(face_or_frame_candidates)) == 1, f'must has the unified face_or_frame type'
    face_or_frame = list(set(face_or_frame_candidates))[0]
    return face_or_frame


# 【修复】添加 model_cfg 参数传递，确保数据集类能正确识别模型类型（如 Gemma3）并生成对应 Prompt 格式
def get_name2cls(dataset, model_cfg=None):
    if dataset == 'MER2023':          return MER2023_Dataset(model_cfg=model_cfg)
    if dataset == 'MER2024':          return MER2024_Dataset(model_cfg=model_cfg)
    if dataset == 'MELD':             return MELD_Dataset(model_cfg=model_cfg)
    if dataset == 'IEMOCAPFour':      return IEMOCAPFour_Dataset(model_cfg=model_cfg)
    if dataset == 'CMUMOSI':          return CMUMOSI_Dataset(model_cfg=model_cfg)
    if dataset == 'CMUMOSEI':         return CMUMOSEI_Dataset(model_cfg=model_cfg)
    if dataset == 'SIMS':             return SIMS_Dataset(model_cfg=model_cfg)
    if dataset == 'SIMSv2':           return SIMSv2_Dataset(model_cfg=model_cfg)
    if dataset == 'MER2025OV':        return MER2025OV_Dataset(model_cfg=model_cfg)
    if dataset == 'OVMERDPlus':       return OVMERDPlus_Dataset(model_cfg=model_cfg)
    print ('dataset cls not provided!')
    return None


# 优先级：outside_user_message > zeroshot > dataset-specific default
def get_user_message(dataset_cls, zeroshot, outside_user_message):
    if outside_user_message is not None:
        user_message = outside_user_message
    elif zeroshot: # predict ov labels
        user_message = dataset_cls.func_get_qa_ovlabel(sample=None, question_only=True)
    else:
        # 【修复】默认使用 ovlabel 问题，避免 user_message 未定义
        # 如果需要其他默认问题，可以根据数据集类型调整
        user_message = dataset_cls.func_get_qa_ovlabel(sample=None, question_only=True)
    return user_message


# 【新增】推理时固定随机种子，确保可复现性
def set_inference_seed(seed=42):
    """固定推理时的随机种子，确保结果可复现"""
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f'【可复现性】已固定随机种子: {seed}')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AffectGPT Inference Process")
    parser.add_argument("--cfg-path", default='xxx', help="path to configuration file.")
    parser.add_argument("--options",  nargs="+", help="override some settings in the used config, format: --option xx=xx yy=yy zz=zz")
    parser.add_argument("--dataset", default='merbench', help="evaluate dataset")
    parser.add_argument('--zeroshot', action='store_true', default=False, help='whether testing on zeroshot performance?')
    parser.add_argument('--outside_user_message',  default=None, help="we use the outside user message, rather than dataset dependent.")
    parser.add_argument('--outside_face_or_frame', default=None, help="we use the outside face_or_frame, rather than dataset dependent.")
    # 【推理优化】控制并行解码进程数（注意：Windows 下建议设为 0，Linux 下可以设为 4-8）
    parser.add_argument('--num_workers', type=int, default=32, help="number of workers for parallel data loading (0=single process)")
    # 【批量推理】控制批量生成的 batch size（建议 2-4，避免 OOM）
    parser.add_argument('--inference_batch_size', type=int, default=64, help="batch size for LLM generation (2-4 recommended)")
    # 【可复现性】推理随机种子
    parser.add_argument('--seed', type=int, default=42, help="random seed for reproducibility")
    args = parser.parse_args()
    
    # 【新增】设置推理随机种子
    set_inference_seed(args.seed)
    
    cfg = Config(args)
    model_cfg = cfg.model_cfg
    datasets_cfg = cfg.datasets_cfg
    inference_cfg = cfg.inference_cfg
    device = 'cuda:{}'.format(inference_cfg.gpu)
    inference_datasets = ['MER2023', 'MER2024', 'MELD', 'IEMOCAPFour', 'CMUMOSI', 'CMUMOSEI', 'SIMS', 'SIMSv2', 'OVMERDPlus']
    

    print ('======== Step1: cfg pre-analysis ========')
    # 支持 ckpt_root / ckpt_name 两种类型输入 => (ckpt3_root)
    # 默认情况是依据 os.path.basename(args.cfg_path) 找到 => (ckpt3_root)
    # 【修复】添加对 None 的检查
    ckpt_root = inference_cfg.get('ckpt_root', 'xxx')
    ckpt_name = inference_cfg.get('ckpt_name', 'xxx')
    
    if ckpt_root not in [None, '', 'xxx']:
        ckpt3_root = ckpt_root
    elif ckpt_name not in [None, '', 'xxx']:
        cfg_name = os.path.basename(args.cfg_path)[:-len('.yaml')]
        ckpt3_root = os.path.join('output', cfg_name, ckpt_name)
        assert ckpt_name.startswith(cfg_name) # 这块和 train 部分是相互配合下的结果
    else:
        print ('strat searching for suitable ckpt_root')
        cfg_name = os.path.basename(args.cfg_path)[:-len('.yaml')]
        root_candidates = glob.glob(os.path.join('output', cfg_name, cfg_name+'*'))
        ckpt3_root = search_for_ckpt_root(root_candidates)
    print ('processed ckpt3 root:')
    print (ckpt3_root)

    # (ckpt3_root) => processed epochs
    print ('processed ckpt3 epochs:')
    whole_ckpt3s = get_ckpt3_candidates(ckpt3_root, inference_cfg)
    for item in whole_ckpt3s: print (os.path.basename(item))

    # => (face_or_frame) (这个需要与训练数据采用的 face_or_frame 相同)
    face_or_frame = get_face_or_frame(datasets_cfg, args.outside_face_or_frame)
    print (f'Read data type: {face_or_frame}')
    print ('=======================================')


    ## main process for each ckpt3 candidates
    for ii, ckpt_3 in enumerate(whole_ckpt3s):

        ##############################################################
        print (f'======== Step2: initial model; using ckpt_3: {os.path.basename(ckpt_3)} ========')
        model_cfg.ckpt_3 = ckpt_3 # ckpt_3 has the highest priority
        if ii == 0: # first-round: initialize models
            model_cls = registry.get_model_class(model_cfg.arch) # affectgpt
            model = model_cls.from_config(model_cfg)
        if ii > 0:  # second-round: update trainable params (用新的 ckpt_3 参数覆盖)
            ckpt = torch.load(model_cfg.ckpt_3, map_location="cpu", weights_only=True)
            model.load_state_dict(ckpt['model'], strict=False)
        model = model.to(device).eval() # !! reduce randomness during the inference
        chat = Chat(model, model_cfg, device=device)
        ##############################################################


        print ('======== Step3: Inferece ========')
        if args.dataset == 'inferenceData':
            process_datasets = inference_datasets
        else:
            names = args.dataset.split(',')
            process_datasets = names
        print ('process datasets: ', process_datasets)

        ## for each dataset
        for dataset in process_datasets:
            print (f'current dataset: {dataset}')
            ## 【修复】传递 model_cfg 给数据集类，确保推理时 Prompt 格式与训练一致（对 Gemma3 至关重要）
            dataset_cls = get_name2cls(dataset, model_cfg=model_cfg)
            dataset_cls.needed_data = dataset_cls.get_needed_data(face_or_frame)
            dataset_cls.vis_processor = BaseProcessor()
            dataset_cls.img_processor = BaseProcessor()
            vis_processor_cfg = inference_cfg.get("vis_processor") # read vis processor
            img_processor_cfg = inference_cfg.get("img_processor") # read img processor
            if vis_processor_cfg is not None:
                dataset_cls.vis_processor = registry.get_processor_class(vis_processor_cfg.train.name).from_config(vis_processor_cfg.train)
            if img_processor_cfg is not None:
                dataset_cls.img_processor = registry.get_processor_class(img_processor_cfg.train.name).from_config(img_processor_cfg.train)
            dataset_cls.n_frms = model_cfg.vis_processor.train.n_frms


            ## 读取每个数据集的内容
            test_names = dataset_cls.read_test_names()
            name2subtitle = dataset_cls.name2subtitle

            ## 定义结果存储位置，如果存在相应路径直接跳过
            save_root = os.path.join(inference_cfg.base_root + f'-{dataset.lower()}', # output/results-{dataset}/ckpt3_name
                                    os.path.basename(ckpt3_root)) 
            if not os.path.exists(save_root): os.makedirs(save_root)
            epoch = os.path.basename(cfg.model_cfg.ckpt_3)[:-4]
            save_path = '%s/%s.npz' %(save_root, epoch) # output/result-{dataset}/ckpt3_name/epochname
            if os.path.exists(save_path): continue

            ## 【批量推理优化】定义轻量级的推理 Dataset 类，用于多进程并行数据加载
            class InferenceDataset(Dataset):
                def __init__(self, names, dataset_cls):
                    self.names = names
                    self.dataset_cls = dataset_cls
                
                def __len__(self):
                    return len(self.names)
                
                def __getitem__(self, idx):
                    name = self.names[idx]
                    sample = {'name': name}
                    
                    # 获取各模态数据的路径
                    video_path, image_path, audio_path, face_npy = None, None, None, None
                    if hasattr(self.dataset_cls, '_get_video_path'): video_path = self.dataset_cls._get_video_path(sample)
                    if hasattr(self.dataset_cls, '_get_audio_path'): audio_path = self.dataset_cls._get_audio_path(sample)
                    if hasattr(self.dataset_cls, '_get_face_path'):  face_npy   = self.dataset_cls._get_face_path(sample)
                    if hasattr(self.dataset_cls, '_get_image_path'): image_path = self.dataset_cls._get_image_path(sample)
                    
                    # 【关键耗时操作】视频/音频解码将在多进程中并行执行
                    sample_data = self.dataset_cls.read_frame_face_audio_text(video_path, face_npy, audio_path, image_path)
                    return name, sample_data

            # 【修复】自定义 collate_fn 以处理多模态数据中的 None 值
            def custom_collate_fn(batch):
                """处理包含 None 的多模态数据，支持动态 batch_size"""
                names = [item[0] for item in batch]
                sample_datas = [item[1] for item in batch]
                return names, sample_datas

            # 【新增】DataLoader worker 初始化函数，确保多进程下的随机性一致
            def worker_init_fn(worker_id):
                worker_seed = args.seed + worker_id
                np.random.seed(worker_seed)
                import random
                random.seed(worker_seed)

            ## 创建 DataLoader 实现数据预加载
            inf_dataset = InferenceDataset(test_names, dataset_cls)
            inf_loader = DataLoader(
                inf_dataset, 
                batch_size=args.inference_batch_size,  # 【批量推理】真正的 LLM 批量生成
                num_workers=args.num_workers,  # Windows 建议设为 0
                shuffle=False, 
                pin_memory=True if args.num_workers == 0 else False,  # 单进程时启用
                prefetch_factor=2 if args.num_workers > 0 else None,
                collate_fn=custom_collate_fn,
                worker_init_fn=worker_init_fn if args.num_workers > 0 else None  # 【新增】多进程种子固定
            )

            ## 主要处理函数
            name2reason = {}
            print(f'\n开始推理，共 {len(test_names)} 个样本...')
            print(f'配置: num_workers={args.num_workers}, inference_batch_size={args.inference_batch_size}, device={device}')
            print('=' * 60)
            
            # 用于 ETA 计算
            total_inference_time = 0
            processed_samples = 0
            
            with torch.inference_mode():
                for batch_idx, (name_batch, sample_data_batch) in enumerate(inf_loader):
                    current_batch_size = len(name_batch)
                    
                    # 计算 ETA
                    avg_time_per_sample = total_inference_time / processed_samples if processed_samples > 0 else 0
                    remaining_samples = len(test_names) - processed_samples
                    eta_seconds = avg_time_per_sample * remaining_samples
                    eta_minutes = eta_seconds / 60
                    
                    print(f'\n[Batch {batch_idx+1}] 处理 {current_batch_size} 个样本 (总进度: {processed_samples}/{len(test_names)})')
                    if processed_samples > 0:
                        print(f'  ⏱️  平均耗时: {avg_time_per_sample:.2f}s/样本 | 剩余时间: {eta_minutes:.1f}分钟 ({eta_seconds:.0f}秒)')
                    
                    # 批量提取多模态特征
                    batch_prompts = []
                    batch_img_lists = []
                    
                    for i in range(current_batch_size):
                        name = name_batch[i]
                        sample_data = sample_data_batch[i]
                        subtitle = name2subtitle[name]
                        
                        print(f'  [{i+1}] {name}: {subtitle}')
                        
                        # 提取多模态特征
                        audio_llms, frame_llms, face_llms, image_llms, multi_llms = None, None, None, None, None
                        audio_hiddens, audio_llms = chat.postprocess_audio(sample_data)  
                        frame_hiddens, frame_llms = chat.postprocess_frame(sample_data)
                        face_hiddens,  face_llms  = chat.postprocess_face(sample_data)
                        _,             image_llms = chat.postprocess_image(sample_data)
                        if face_or_frame.startswith('multiface'):
                            _, multi_llms = chat.postprocess_multi(face_hiddens, audio_hiddens)
                        elif face_or_frame.startswith('multiframe'):
                            _, multi_llms = chat.postprocess_multi(frame_hiddens, audio_hiddens)

                        img_list = {
                            'audio': audio_llms,
                            'frame': frame_llms,
                            'face':  face_llms,
                            'image': image_llms,
                            'multi': multi_llms
                        }

                        # 构造 prompt
                        user_message = get_user_message(dataset_cls, args.zeroshot, args.outside_user_message)
                        prompt = dataset_cls.get_prompt_for_multimodal(face_or_frame, subtitle, user_message)
                        
                        # 【调试】打印第一个样本的完整 prompt
                        if i == 0 and batch_idx == 0:
                            print(f'\n[DEBUG] user_message = {user_message}')
                            print(f'[DEBUG] prompt (前500字符) = {prompt[:500]}...')
                        
                        batch_prompts.append(prompt)
                        batch_img_lists.append(img_list)
                    
                    # 【批量生成】一次性推理整个 batch
                    start_time = time.time()
                    responses = chat.answer_batch(
                        prompts=batch_prompts, 
                        img_lists=batch_img_lists,
                        num_beams=1, temperature=0.0, do_sample=False, top_p=1.0, 
                        max_new_tokens=1200, max_length=2000
                    )
                    inference_time = time.time() - start_time
                    
                    # 记录结果
                    for i in range(current_batch_size):
                        name2reason[name_batch[i]] = responses[i]
                    
                    total_inference_time += inference_time
                    processed_samples += current_batch_size
                    
                    avg_time_per_sample_batch = inference_time / current_batch_size
                    print(f'  ✅ Batch 生成时间: {inference_time:.2f}s (平均 {avg_time_per_sample_batch:.2f}s/样本)')
                    for i in range(current_batch_size):
                        print(f'     [{i+1}] {responses[i]}')

                    # if batch_idx == 2: break  # for debug

            print ('\nsave results')
            np.savez_compressed(save_path, name2reason=name2reason)
