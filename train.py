import os
import sys
import time
import random
import argparse
import numpy as np

import torch
from datetime import datetime
import torch.backends.cudnn as cudnn

import my_affectgpt.tasks as tasks
from my_affectgpt.common.config import Config
from my_affectgpt.common.dist_utils import get_rank, init_distributed_mode
from my_affectgpt.common.logger import setup_logger
from my_affectgpt.common.registry import registry
from my_affectgpt.common.optims import LinearWarmupCosineLRScheduler, LinearWarmupStepLRScheduler
from my_affectgpt.tasks import *
from my_affectgpt.models import *
from my_affectgpt.runners import *
from my_affectgpt.processors import *
from my_affectgpt.datasets.builders import *

def setup_seeds(config): 
    seed = config.run_cfg.seed + get_rank()
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    # 【新增】固定 CUDA 随机种子，确保 GPU 计算可复现
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # 多 GPU 情况
    cudnn.benchmark = False
    cudnn.deterministic = True

def parse_args():
    parser = argparse.ArgumentParser(description="Training")
    parser.add_argument("--cfg-path", required=True, help="path to configuration file.")
    parser.add_argument("--options",  nargs="+", help="overwrite params in xxx.config (only for run and model). Example: --options 'ckpt=aaa' 'ckpt_2=bbb'")
    args = parser.parse_args()
    return args

def get_runner_class(cfg):
    """
    Get runner class from config. Default to epoch-based runner.
    """
    runner_cls = registry.get_runner_class(cfg.run_cfg.get("runner", "runner_base")) # 'affectgpt.runners.runner_base.RunnerBase'
    return runner_cls

# ==================== 【新增】日志重定向类 ====================
class TeeLogger:
    """同时输出到控制台和文件的日志类"""
    def __init__(self, log_file, mode='a'):
        self.terminal = sys.stdout
        self.log = open(log_file, mode, encoding='utf-8', buffering=1)  # 行缓冲
    
    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()  # 立即写入文件
    
    def flush(self):
        self.terminal.flush()
        self.log.flush()
    
    def close(self):
        self.log.close()

def setup_log_file(job_name, job_id):
    """
    创建日志文件路径并重定向输出
    
    Args:
        job_name: 配置文件名（不含 .yaml）
        job_id: 训练任务 ID
    
    Returns:
        log_file_path: 日志文件完整路径
    """
    # 创建输出目录（与 checkpoint 同级）
    output_dir = os.path.join('output', job_name, job_id)
    os.makedirs(output_dir, exist_ok=True)
    
    # 日志文件路径
    log_file = os.path.join(output_dir, 'training.log')
    
    # 重定向 stdout 和 stderr
    sys.stdout = TeeLogger(log_file, mode='w')  # 覆盖模式
    sys.stderr = TeeLogger(log_file, mode='a')  # 追加模式
    
    # 打印日志文件路径（会同时输出到控制台和文件）
    print(f"{'='*80}")
    print(f"日志文件: {os.path.abspath(log_file)}")
    print(f"训练开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*80}\n")
    
    return log_file

def main():

    args = parse_args()
    cfg = Config(args)

    # # set before init_distributed_mode() to ensure the same job_id shared across all ranks.
    # # max_epoch = cfg.run_cfg['max_epoch'] 
    # # job_id = f"{args.job_name}_affectgpt_epoch_{max_epoch}_{time.time()}"
    # # job_id = f"{args.job_name}_affectgpt_{time.time()}"
    # job_name = os.path.basename(args.cfg_path)[:-len('.yaml')]
    # job_id = f"{job_name}_{str(int(time.time()))}" # 减少小数点后的存储，便于复制
    # # job_id = job_name # debug
    # # job_id = f"affectgpt_{time.time()}"

    # 用于分布式训练：防止 nccl barrier 卡死，job_id 统一确保进程能找到对应的进程组
    os.environ["TORCH_NCCL_BLOCKING_WAIT"] = "1"
    job_name = os.path.basename(args.cfg_path)[:-len('.yaml')]
    job_id = f"{job_name}_{datetime.now().strftime('%Y%m%d%H%M')[:-1]}" # zhuofan

    print (job_id)

    # ==================== 【新增】设置日志文件 ====================
    log_file = setup_log_file(job_name, job_id)

    # print logging files
    init_distributed_mode(cfg.run_cfg)
    setup_seeds(cfg)
    setup_logger() 
    cfg.pretty_print()

    # load task and start training
    task = tasks.setup_task(cfg) # video_text_pretrain
    datasets = task.build_datasets(cfg)
    model = task.build_model(cfg)
    runner = get_runner_class(cfg)(
        cfg=cfg,
        job_id=job_id, 
        task=task, 
        model=model, 
        datasets=datasets
    )

# ==================== 【新增】记录训练开始 ====================
    print(f"\n{'='*80}")
    print(f"开始训练: {job_name}")
    print(f"任务 ID: {job_id}")
    print(f"输出目录: output/{job_name}/{job_id}/")
    print(f"{'='*80}\n")
    
    try:
        runner.train()
        
        # ==================== 【新增】记录训练完成 ====================
        print(f"\n{'='*80}")
        print(f"✅ 训练完成！")
        print(f"结束时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"日志文件: {os.path.abspath(log_file)}")
        print(f"权重目录: output/{job_name}/{job_id}/")
        print(f"{'='*80}\n")
    
    except Exception as e:
        # ==================== 【新增】记录训练异常 ====================
        import traceback
        print(f"\n{'='*80}")
        print(f"❌ 训练出现异常！")
        print(f"错误时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"{'='*80}")
        print(traceback.format_exc())
        print(f"{'='*80}\n")
        raise

if __name__ == "__main__":
    main()
