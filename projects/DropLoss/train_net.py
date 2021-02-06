# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Detection Training Script.

This scripts reads a given config file and runs the training or evaluation.
It is an entry point that is made to train standard models in detectron2.

In order to let one script support training of many models,
this script contains logic that are specific to these built-in models and therefore
may not be suitable for your own project.
For example, your research project perhaps only needs a single "evaluator".

Therefore, we recommend you to use detectron2 as an library and take
this file as an example of how to use the library.
You may want to write your own script with your datasets and other customizations.
"""

import logging
import os
from collections import OrderedDict
import torch

import detectron2.utils.comm as comm
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog
from detectron2.engine import DefaultTrainer, default_argument_parser, default_setup, hooks, launch
from detectron2.evaluation import (
    CityscapesEvaluator,
    COCOEvaluator,
    COCOPanopticEvaluator,
    DatasetEvaluators,
    LVISEvaluator,
    PascalVOCDetectionEvaluator,
    SemSegEvaluator,
    verify_results,
)
from detectron2.modeling import GeneralizedRCNNWithTTA
from droploss import add_config

import numpy as np
import matplotlib.pyplot as plt
from multiprocessing import Pool
import multiprocessing as mp
import time
import os.path as osp
import pickle
from detectron2.engine import HookBase
from typing import Any, Dict, Iterable, List, NamedTuple, Optional, Tuple


class PeriodicRecorder(HookBase):
    def __init__(
        self,
        period: int,
        output_dir: str, 
        max_iter: Optional[int] = None,
        max_to_keep: Optional[int] = None,
    ) -> None:
        """
        Args:
            checkpointer (Any): the checkpointer object used to save
            checkpoints.
            period (int): the period to save checkpoint.
            max_iter (int): maximum number of iterations. When it is reached,
                a checkpoint named "model_final" will be saved.
            max_to_keep (int): maximum number of most current checkpoints to keep,
                previous checkpoints will be deleted
        """
        self.period = int(period)
        self.max_iter = max_iter
        if max_to_keep is not None:
            assert max_to_keep > 0
        self.max_to_keep = max_to_keep
        self.record_loss = torch.zeros((1231,1230)).cuda()
        self.record_time = torch.zeros(1231).cuda()
        self._output_dir = output_dir

    def before_train(self):
        self.max_iter = self.trainer.max_iter

    def after_step(self):
        # No way to use **kwargs
        self.step(self.trainer.iter)
    
    def step(self, iteration: int, **kwargs: Any) -> None:
        iteration = int(iteration)
        additional_state = {"iteration": iteration}
        additional_state.update(kwargs)

        try:
            record_loss = self.trainer.model.module.roi_heads.record_loss
            record_cls  = self.trainer.model.module.roi_heads.record_cls
        except:
            record_loss = self.trainer.model.roi_heads.record_loss
            record_cls  = self.trainer.model.roi_heads.record_cls


        
        useful_cls = torch.unique(record_cls)
        for i in useful_cls:
            index = (record_cls == i )
            self.record_loss[i] +=torch.sum(record_loss[index, :], 0)
            self.record_time[i] += torch.sum(index)

        if (iteration + 1) % self.period == 0:
            file_path = os.path.join(self._output_dir, "loss_record_{}.pkl".format(iteration))
            self.record = {}
            self.record['loss'] = self.record_loss
            self.record['time'] = self.record_time

            with open(file_path,'wb') as f:
                pickle.dump(self.record, f)





class Trainer(DefaultTrainer):
    """
    We use the "DefaultTrainer" which contains pre-defined default logic for
    standard training workflow. They may not work for you, especially if you
    are working on a new research project. In that case you can use the cleaner
    "SimpleTrainer", or write your own training loop. You can use
    "tools/plain_train_net.py" as an example.
    """

    # def build_hooks(self):
    #     ret = super().build_hooks()

    #     cfg = self.cfg.clone()
    #     cfg.defrost()

    #     ret.insert(0, PeriodicRecorder(cfg.SOLVER.CHECKPOINT_PERIOD, cfg.OUTPUT_DIR))

    #     return ret

    
    def run_step(self):
        """
        Implement the standard training logic described above.
        """
        assert self.model.training, "[SimpleTrainer] model was changed to eval mode!"
        start = time.perf_counter()
        """
        If you want to do something with the data, you can wrap the dataloader.
        """
        data = next(self._data_loader_iter)
        data_time = time.perf_counter() - start

        """
        If you want to do something with the losses, you can wrap the model.
        """
        loss_dict = self.model(data)
        losses = sum(loss_dict.values())

        self._detect_anomaly(losses, loss_dict)

        metrics_dict = loss_dict
        metrics_dict["data_time"] = data_time
        self._write_metrics(metrics_dict)

        """
        If you need to accumulate gradients or something similar, you can
        wrap the optimizer with your custom `zero_grad()` method.
        """
        self.optimizer.zero_grad()
        losses.backward()
        
        # try:
        #     self.record += torch.mean(self.model.roi_heads.tmp_loss, 0)
        # except:
        #     self.record += torch.mean(self.model.module.roi_heads.tmp_loss, 0)

        # if self.iter in self.record_time:
        #     print(self.iter)
        #     path = osp.join('./output', 'iter_{}.pickle'.format(self.iter))
        #     with open(path,'wb') as f:
        #         pickle.dump(self.record, f)



        # torch.sum(self.model.roi_heads.box_predictor.cls_score.weight.grad, 1)
        # self.model.roi_heads.box_predictor.cls_score.weight.grad
        """
        If you need gradient clipping/scaling or other processing, you can
        wrap the optimizer with your custom `step()` method.
        """
        self.optimizer.step()

    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        """
        Create evaluator(s) for a given dataset.
        This uses the special metadata "evaluator_type" associated with each builtin dataset.
        For your own dataset, you can simply create an evaluator manually in your
        script and do not have to worry about the hacky if-else logic here.
        """
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
        evaluator_list = []
        evaluator_type = MetadataCatalog.get(dataset_name).evaluator_type
        if evaluator_type in ["sem_seg", "coco_panoptic_seg"]:
            evaluator_list.append(
                SemSegEvaluator(
                    dataset_name,
                    distributed=True,
                    num_classes=cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES,
                    ignore_label=cfg.MODEL.SEM_SEG_HEAD.IGNORE_VALUE,
                    output_dir=output_folder,
                )
            )
        if evaluator_type in ["coco", "coco_panoptic_seg"]:
            evaluator_list.append(COCOEvaluator(dataset_name, cfg, True, output_folder))
        if evaluator_type == "coco_panoptic_seg":
            evaluator_list.append(COCOPanopticEvaluator(dataset_name, output_folder))
        elif evaluator_type == "cityscapes":
            assert (
                torch.cuda.device_count() >= comm.get_rank()
            ), "CityscapesEvaluator currently do not work with multiple machines."
            return CityscapesEvaluator(dataset_name)
        elif evaluator_type == "pascal_voc":
            return PascalVOCDetectionEvaluator(dataset_name)
        elif evaluator_type == "lvis":
            return LVISEvaluator(dataset_name, cfg, True, output_folder)
        if len(evaluator_list) == 0:
            raise NotImplementedError(
                "no Evaluator for the dataset {} with the type {}".format(
                    dataset_name, evaluator_type
                )
            )
        elif len(evaluator_list) == 1:
            return evaluator_list[0]
        return DatasetEvaluators(evaluator_list)

    @classmethod
    def test_with_TTA(cls, cfg, model):
        logger = logging.getLogger("detectron2.trainer")
        # In the end of training, run an evaluation with TTA
        # Only support some R-CNN models.
        logger.info("Running inference with test-time augmentation ...")
        model = GeneralizedRCNNWithTTA(cfg, model)
        evaluators = [
            cls.build_evaluator(
                cfg, name, output_folder=os.path.join(cfg.OUTPUT_DIR, "inference_TTA")
            )
            for name in cfg.DATASETS.TEST
        ]
        res = cls.test(cfg, model, evaluators)
        res = OrderedDict({k + "_TTA": v for k, v in res.items()})
        return res

def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    add_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    default_setup(cfg, args)
    return cfg


def main(args):
    cfg = setup(args)

    if args.eval_only:
        model = Trainer.build_model(cfg)
        DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
            cfg.MODEL.WEIGHTS, resume=args.resume
        )
        res = Trainer.test(cfg, model)
        if comm.is_main_process():
            verify_results(cfg, res)
        if cfg.TEST.AUG.ENABLED:
            res.update(Trainer.test_with_TTA(cfg, model))
        return res

    """
    If you'd like to do anything fancier than the standard training logic,
    consider writing your own training loop or subclassing the trainer.
    """
    trainer = Trainer(cfg)
    trainer.resume_or_load(resume=args.resume)
    if cfg.TEST.AUG.ENABLED:
        trainer.register_hooks(
            [hooks.EvalHook(0, lambda: trainer.test_with_TTA(cfg, trainer.model))]
        )
    return trainer.train()


if __name__ == "__main__":
    args = default_argument_parser().parse_args()
    print("Command Line Args:", args)
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )
