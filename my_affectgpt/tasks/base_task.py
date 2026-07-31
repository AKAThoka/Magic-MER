"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE_Lavis file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

import os
import logging

import torch
import torch.distributed as dist
from my_affectgpt.common.dist_utils import get_rank, get_world_size, is_main_process, is_dist_avail_and_initialized
from my_affectgpt.common.logger import MetricLogger, SmoothedValue
from my_affectgpt.common.registry import registry
from my_affectgpt.datasets.data_utils import prepare_sample


def _mem_snap(tag: str) -> None:
    """打印当前 GPU 显存快照（仅在第一个 iter 调用）"""
    if not torch.cuda.is_available():
        return
    alloc  = torch.cuda.memory_allocated()  / 1024**3
    reserv = torch.cuda.memory_reserved()   / 1024**3
    peak   = torch.cuda.max_memory_allocated() / 1024**3
    print(f"  [显存快照] {tag}")
    print(f"    allocated={alloc:.3f} GB  reserved={reserv:.3f} GB  peak={peak:.3f} GB")


# main process: model, dataset, training, evaluation, ...
class BaseTask:
    def __init__(self, **kwargs):
        super().__init__()
        self.inst_id_key = "instance_id"

    @classmethod
    def setup_task(cls, **kwargs):
        return cls() # 'affectgpt.tasks.video_text_pretrain.VideoTextPretrainTask'

    def build_model(self, cfg):
        model_config = cfg.model_cfg
        model_cls = registry.get_model_class(model_config.arch)
        return model_cls.from_config(model_config)

    def build_datasets(self, cfg):
        """
        Build a dictionary of datasets, keyed by split 'train', 'valid', 'test'.

        Args:
            cfg (common.config.Config): _description_

        Returns:
            dict: Dictionary of torch.utils.data.Dataset objects by split.
        """
        
        datasets = dict()
        datasets_cfg = cfg.datasets_cfg
        model_cfg = cfg.model_cfg
        assert len(datasets_cfg) > 0, "At least one dataset has to be specified."

        for name in datasets_cfg:
            dataset_cfg = datasets_cfg[name]
            ############################ dataset_config Post-processing ############################
            assert dataset_cfg is not None
            if dataset_cfg.face_or_frame.startswith('multi'):
                assert model_cfg.multi_fusion_type in ['attention', 'qformer']
            builder = registry.get_builder_class(name)(dataset_cfg, model_cfg) # 找到这个dataset对应的builder
            ########################################################################################
            dataset = builder.build_datasets() # 每个builder有自己的 build_datasets 函数
            dataset['train'].name = name
            if 'sample_ratio' in dataset_cfg:
                dataset['train'].sample_ratio = dataset_cfg.sample_ratio
            datasets[name] = dataset
        return datasets

    # training: one iter
    def train_step(self, model, samples):
        loss = model(samples)["loss"]
        return loss

    def valid_step(self, model, samples):
        raise NotImplementedError

    def before_evaluation(self, model, dataset, **kwargs):
        model.before_evaluation(dataset=dataset, task_type=type(self))

    def after_evaluation(self, **kwargs):
        pass

    def inference_step(self):
        raise NotImplementedError

    def evaluation(self, model, data_loader, cuda_enabled=True):
        metric_logger = MetricLogger(delimiter="  ")
        header = "Evaluation"
        # TODO make it configurable
        print_freq = 10

        results = []

        for samples in metric_logger.log_every(data_loader, print_freq, header):
            samples = prepare_sample(samples, cuda_enabled=cuda_enabled)

            eval_output = self.valid_step(model=model, samples=samples)
            results.extend(eval_output)

        if is_dist_avail_and_initialized():
            dist.barrier()

        return results

    # one epoch contains iters_per_epoch iters (see trains.config)
    def train_epoch(
        self,
        epoch,
        model,
        data_loader,
        optimizer,
        lr_scheduler,
        scaler=None,
        cuda_enabled=False,
        log_freq=50,
        accum_grad_iters=1,
    ):
        inner_epoch = epoch 
        iters_per_epoch = lr_scheduler.iters_per_epoch
        use_amp = scaler is not None

        if not hasattr(data_loader, "__next__"):
            # convert to iterator if not already
            data_loader = iter(data_loader)

        metric_logger = MetricLogger(delimiter="  ")
        metric_logger.add_meter("lr",   SmoothedValue(window_size=1, fmt="{value:.8f}"))
        metric_logger.add_meter("loss", SmoothedValue(window_size=1, fmt="{value:.8f}"))

        # if iter-based runner, schedule lr based on inner epoch.
        logging.info(
            "Start training epoch {}, {} iters per inner epoch.".format(
                epoch, iters_per_epoch
            )
        )
        header = "Train: data epoch: [{}]".format(epoch) # 'Train: data epoch: [0]'
        
        for i in metric_logger.log_every(range(iters_per_epoch), log_freq, header):
            # if using iter-based runner, we stop after iters_per_epoch iterations.
            if i >= iters_per_epoch:
                break
            
            samples = next(data_loader)
            samples = prepare_sample(samples, cuda_enabled=cuda_enabled) # move all samples-tensor into cuda
            samples.update( # add new key-value into map
                {
                    "epoch": inner_epoch,
                    "num_iters_per_epoch": iters_per_epoch,
                    "iters": i,
                    "max_grad_norm": 1.0,  # 梯度裁剪阈值
                }
            )

            lr_scheduler.step(cur_epoch=inner_epoch, cur_step=i)

            # ── 仅在整个训练的第一个 iter 打印显存分解快照 ──────────────────
            _mem_profile = (i == 0 and not getattr(self, '_mem_snap_done', False))

            # 修改此处逻辑以支持 PyTorch 2.x 所有版本    
            from packaging import version        
            current_v = version.parse(torch.__version__)
            
            if current_v >= version.parse("2.4.0"):
                with torch.amp.autocast('cuda', enabled=use_amp):
                    loss = self.train_step(model=model, samples=samples)
            else:
                with torch.cuda.amp.autocast(enabled=use_amp):
                    loss = self.train_step(model=model, samples=samples)

            if _mem_profile:
                _mem_snap("前向传播后 (含激活值)")

            if use_amp:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            if _mem_profile:
                _mem_snap("反向传播后 (含梯度)")

            # update gradients every accum_grad_iters iterations
            if (i + 1) % accum_grad_iters == 0:
                # 梯度裁剪：防止 Gemma 3 训练初期梯度爆炸
                max_grad_norm = getattr(samples, 'max_grad_norm', 1.0)  # 从配置获取，默认 1.0
                if use_amp:
                    scaler.unscale_(optimizer)  # 先 unscale 才能裁剪
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                    scaler.step(optimizer)
                    scaler.update()                     
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                    optimizer.step()
                optimizer.zero_grad()

            if _mem_profile:
                _mem_snap("optimizer.step + zero_grad 后 (含optimizer states)")
                self._mem_snap_done = True   # 整个训练只打印一次

            metric_logger.update(loss=loss.item())
            metric_logger.update(lr=optimizer.param_groups[0]["lr"])

        # gather the stats from all processes
        metric_logger.synchronize_between_processes()
        logging.info("Averaged stats: " + str(metric_logger.global_avg()))
        return {
            k: "{:.3f}".format(meter.global_avg)
            for k, meter in metric_logger.meters.items()
        }

    @staticmethod
    def save_result(result, result_dir, filename, remove_duplicate=""):
        import json

        result_file = os.path.join(
            result_dir, "%s_rank%d.json" % (filename, get_rank())
        )
        final_result_file = os.path.join(result_dir, "%s.json" % filename)

        json.dump(result, open(result_file, "w"))

        if is_dist_avail_and_initialized():
            dist.barrier()

        if is_main_process():
            logging.warning("rank %d starts merging results." % get_rank())
            # combine results from all processes
            result = []

            for rank in range(get_world_size()):
                result_file = os.path.join(
                    result_dir, "%s_rank%d.json" % (filename, rank)
                )
                res = json.load(open(result_file, "r"))
                result += res

            if remove_duplicate:
                result_new = []
                id_list = []
                for res in result:
                    if res[remove_duplicate] not in id_list:
                        id_list.append(res[remove_duplicate])
                        result_new.append(res)
                result = result_new

            json.dump(result, open(final_result_file, "w"))
            print("result file saved to %s" % final_result_file)

        return final_result_file
