from detectron2.data import DatasetMapper, build, build_detection_train_loader, build_detection_test_loader
from detectron2.data.catalog import MetadataCatalog
from detectron2.engine.hooks import HookBase
from detectron2.engine import hooks, DefaultTrainer
from detectron2.evaluation import COCOEvaluator
from detectron2.utils.logger import log_every_n_seconds
from torch.utils.data.sampler import Sampler
import detectron2.utils.comm as comm
from detectron2.utils.logger import setup_logger

from typing import Optional, List
import torch
import time
import datetime
import numpy as np
from pathlib import Path
import pickle
import os
import itertools
import logging

# Local logger setup 
path = Path(__file__)
logger = setup_logger(output=None, name=path.parent.name + "." + path.stem)
class TrainerWithValidation(DefaultTrainer):
  def __init__(self, cfg, val_cfg):
      self.validation_cfg = val_cfg
      super().__init__(cfg)
  @classmethod
  def build_evaluator(cls, cfg, dataset_name, output_folder=None):
    if output_folder is None:
      output_folder = os.path.join(cfg.OUTPUT_DIR, "validation")
    logger.info(f"Validation on {dataset_name}")
    return COCOEvaluator(dataset_name, ("bbox", ), False, output_folder)
  @classmethod
  def build_train_loader(cls, cfg):
    if cfg.SOLVER.IMS_PER_BATCH > 1:
        data_loader = build_detection_train_loader(
            cfg, sampler=DataDistillationTrainingSampler(cfg.SOLVER.IMS_PER_BATCH,
                [MetadataCatalog.get("manual_annotation").size, MetadataCatalog.get("auto_annotation").size],
                [0.6, 0.4]))
    else:
        data_loader = build_detection_train_loader(cfg)
    return data_loader

  def build_hooks(self):
    _hooks = super().build_hooks()
    _hooks.insert(-1, ValLossHook(
        self.validation_cfg.TEST.EVAL_PERIOD,
        self.model,
        build_detection_test_loader(
            self.validation_cfg,
            self.validation_cfg.DATASETS.TEST[0],
            DatasetMapper(self.validation_cfg, True)
        )
    ))
    def test_and_save_results():
        logger.info(f"Doing inference on {self.validation_cfg.MODEL.DEVICE}")
        self.model.to(torch.device(self.validation_cfg.MODEL.DEVICE))
        self._last_eval_results = self.test(self.validation_cfg, self.model)
        self.model.to(torch.device(self.cfg.MODEL.DEVICE))
        return self._last_eval_results
    _hooks.append(hooks.EvalHook(self.validation_cfg.TEST.EVAL_PERIOD, test_and_save_results))
    return _hooks



class ValLossHook(HookBase):
    def __init__(self, eval_period, model, data_loader):
        self._model = model
        self._period = eval_period
        self._data_loader = data_loader
        self.grads = []
    def _do_loss_eval(self):
        # Copying inference_on_dataset from evaluator.py
        total = len(self._data_loader)
        num_warmup = min(5, total - 1)
        start_time = time.perf_counter()
        total_compute_time = 0
        losses = []
        for idx, inputs in enumerate(self._data_loader):
            if idx == num_warmup:
                start_time = time.perf_counter()
                total_compute_time = 0
            start_compute_time = time.perf_counter()
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            total_compute_time += time.perf_counter() - start_compute_time
            iters_after_start = idx + 1 - num_warmup * int(idx >= num_warmup)
            seconds_per_img = total_compute_time / iters_after_start
            if idx >= num_warmup * 2 or seconds_per_img > 5:
                total_seconds_per_img = (time.perf_counter() - start_time) / iters_after_start
                eta = datetime.timedelta(seconds=int(total_seconds_per_img * (total - idx - 1)))
                log_every_n_seconds(
                    logging.INFO,
                    "Loss on Validation  done {}/{}. {:.4f} s / img. ETA={}".format(
                        idx + 1, total, seconds_per_img, str(eta)
                    ),
                    n=5,
                )
            loss_batch = self._get_loss(inputs)
            losses.append(loss_batch)
        self.acc_grad_flow()
        mean_loss = np.mean(losses)
        self.trainer.storage.put_scalar('validation_loss', mean_loss)
        self._model.latest_val_loss = mean_loss
        comm.synchronize()

        return losses

    def _get_loss(self, data):
        # How loss is calculated on train_loop
        metrics_dict = self._model(data)
        metrics_dict = {
            k: v.detach().cpu().item() if isinstance(v, torch.Tensor) else float(v)
            for k, v in metrics_dict.items()
        }
        total_losses_reduced = sum(loss for loss in metrics_dict.values())
        return total_losses_reduced

    def after_step(self):
        next_iter = self.trainer.iter + 1
        is_final = next_iter == self.trainer.max_iter
        if is_final or (self._period > 0 and next_iter % self._period == 0):
            self._do_loss_eval()
        self.trainer.storage.put_scalars(timetest=12)

    def acc_grad_flow(self):
        ave_grads = []
        layers = []
        for n, p in self._model.named_parameters():
            if(p.requires_grad) and ("bias" not in n):
                layers.append(n)
                ave_grads.append(p.grad.abs().mean())
        self.grads.append(ave_grads)
        
    def after_train(self):
        with open('grad_acc.pickle', 'wb') as f:
            pickle.dump(self.grads, f)

class DataDistillationTrainingSampler(Sampler):
    def __init__(self, mini_batch: int, sizes: List[int], subset_ratios: List[float], 
        seeds: Optional[List[int]] = None):
        """
        Args:
            sizes List(int): the total number of data of each underlying dataset to sample from
            subset_ratios List(float): the sampling subset ratios of each underlying dataset to sample from
            seed (int): the initial seeds of the shuffle. Must be the same
                across all workers. If None, will use a random seed shared
                among workers (require synchronization among all workers).
        """
        assert len(sizes) == len(subset_ratios)
        if seeds is not None:
            assert len(sizes) == len(seeds)
        logger.info("Using DataDistillationTrainingSampler......")
        if seeds is None:
            seeds = []
            for i in range(len(sizes)):
                seeds.append(int(comm.shared_random_seed()))
        self._seeds = seeds
        self._sizes = sizes
        self._nums_subsamples = []
        self._rank = comm.get_rank()
        self._world_size = comm.get_world_size()
        for i in range(len(sizes)):
            num_subsamples = round(mini_batch*subset_ratios[i])
            self._nums_subsamples.append(num_subsamples)
            logger.info(f"Randomly sample {num_subsamples} data from the original {sizes[i]} data")

    def __iter__(self):
        start = self._rank
        yield from itertools.islice(self._infinite_indices(), start, None, self._world_size)
    
    def _infinite_indices(self):
        generators = []
        for i in range(len(self._sizes)):
            g = torch.Generator()
            g.manual_seed(self._seeds[i])  # self._seed equals seed_shuffle from __init__()
            generators.append(g)
        while True:
            to_yield = []
            base = 0
            for i in range(len(self._sizes)):
                size = self._sizes[i]
                indexes_randperm = torch.randperm(size, generator=generators[i])
                indexes_randperm_subset = indexes_randperm[:self._nums_subsamples[i]].tolist()
                indexes_randperm_subset = [x + base for x in indexes_randperm_subset]
                to_yield += indexes_randperm_subset
                base += size
            yield from to_yield

