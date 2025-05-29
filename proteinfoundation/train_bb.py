# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

import os
import sys

root = os.path.abspath(".")
sys.path.append(root)  # Adds project's root directory

import argparse
import json
import pickle
from pathlib import Path

import hydra
import lightning as L
import loralib as lora
import torch
import wandb
from dotenv import load_dotenv
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.utilities import rank_zero_only
from loguru import logger
from omegaconf import OmegaConf

from proteinfoundation.proteinflow.proteinb import Proteinb
from proteinfoundation.utils.ema_utils.ema_callback import EMA, EmaModelCheckpoint
from proteinfoundation.utils.fetch_last_ckpt import fetch_last_ckpt
from proteinfoundation.utils.lora_utils import replace_lora_layers
from proteinfoundation.utils.metric_utils import (
    transform_global_percentage_to_mask_dropout,
)
from proteinfoundation.utils.seed_callback import SeedCallback
from proteinfoundation.utils.training_analysis_utils import (
    GradAndWeightAnalysisCallback,
    LogEpochTimeCallback,
    LogSetpTimeCallback,
    SkipNanGradCallback,
)


# Things that should only be done by a single process
@rank_zero_only
def log_info(msg):
    logger.info(msg)


@rank_zero_only
def store_configs(path_configs):
    for cfg, path in path_configs:
        with open(path, "w") as f:
            cfg_aux = OmegaConf.to_container(cfg, resolve=True)
            json.dump(cfg_aux, f, indent=4, sort_keys=True)


@rank_zero_only
def log_configs(path_configs, wandb_logger):
    if wandb_logger is None:
        return
    artifact = wandb.Artifact(f"config_files_{run_name}", type="config")
    for _, path in path_configs:
        artifact.add_file(path)
    wandb_logger.experiment.log_artifact(artifact)


@rank_zero_only
def create_dir(checkpoint_path_store, parents=True, exist_ok=True):
    Path(checkpoint_path_store).mkdir(parents=parents, exist_ok=exist_ok)


if __name__ == "__main__":

    load_dotenv()

    parser = argparse.ArgumentParser(description="Job info")
    parser.add_argument(
        "--config_name",
        type=str,
        default="training_ca",
        help="Name of the config yaml file.",
    )
    parser.add_argument(
        "--nolog",
        action="store_true",
        help="Avoids checkpoints and wandb logging, mostly for debugging.",
    )
    parser.add_argument(
        "--single", action="store_true", help="Sets single node and single GPU, ignoring config file."
    )
    parser.add_argument(
        "--show_prog_bar",
        action="store_true",
        help="Shows progress bar as training progresses.",
    )
    args = parser.parse_args()

    logger.add(
        sys.stdout,
        format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {file}:{line} | {message}",
    )  # Send to stdout
    log_info(f"Avoid wandb and checkpointing: {args.nolog}")
    callbacks = [SeedCallback()]  # Different devices will be assigend different seeds

    # Load experiment config
    config_path = "../configs/experiment_config"
    with hydra.initialize(config_path, version_base=hydra.__version__):
        cfg_exp = hydra.compose(config_name=args.config_name)
        if args.single:
            # Rewrite number of GPUs and nodes for local runs or if single flag is used
            cfg_exp.hardware.ngpus_per_node_ = 1
            cfg_exp.hardware.nnodes_ = 1
            cfg_exp.run_name_ = cfg_exp.run_name_ + "_single"
        log_info(f"Exp config {cfg_exp}")

    # Set training precision
    precision = "32"
    if not cfg_exp.force_precision_f32:
        log_info("Using mixed precision")
        torch.set_float32_matmul_precision("medium")
        # precision = "16"
        precision = "bf16-mixed"

    # Set training fold labels dropout rate based on global percentage
    if cfg_exp.training.get("fold_label_sample_ratio") is not None:
        log_info("Setting fold label dropout rate based on fold_label_sample_ratio")
        (
            cfg_exp.training.mask_T_prob,
            cfg_exp.training.mask_A_prob,
            cfg_exp.training.mask_C_prob,
        ) = transform_global_percentage_to_mask_dropout(
            cfg_exp.training.fold_label_sample_ratio
        )
        log_info(
            "Set mask_T_prob: %.3f, mask_A_prob: %.3f, mask_C_prob: %.3f"
            % (
                cfg_exp.training.mask_T_prob,
                cfg_exp.training.mask_A_prob,
                cfg_exp.training.mask_C_prob,
            )
        )

    # Set run name and root directory for the run, used to store things
    run_name = cfg_exp.run_name_
    log_info(f"Job name: {run_name}")
    root_run = os.path.join(
        ".", "store", run_name
    )  # Everything stored in ./store/<run_id>
    log_info(f"Root run: {root_run}")

    # Set checkpoint directory
    checkpoint_path_store = os.path.join(
        root_run, "checkpoints"
    )  # Checkpoints in ./store/run_id/checkpoints/<ckpt-file>
    log_info(f"Checkpoints directory: {checkpoint_path_store}")

    # Check if last checkpoint exists (this is useful if interrupted, it starts from last checkpoint)
    last_ckpt_name = fetch_last_ckpt(checkpoint_path_store)
    last_ckpt_path = (
        os.path.join(checkpoint_path_store, last_ckpt_name)
        if last_ckpt_name is not None
        else None
    )
    log_info(f"Last checkpoint: {last_ckpt_path}")

    # Extract number of cpus from config file
    num_cpus = cfg_exp.hardware.ncpus_per_task_train_
    log_info(
        f"Number of CPUs per task used (will be used for number dataloader number of workers): {num_cpus}"
    )

    # If no checkpoint set seed for correct initialization
    if last_ckpt_path is None:
        log_info(f"Seeding everything to seed {cfg_exp.seed}")
        L.seed_everything(cfg_exp.seed)

    # Load data config
    dataset_config_subdir = cfg_exp.get("dataset_config_subdir", None)
    if dataset_config_subdir is not None:
        # if args.dataset_config_subdir:
        config_path = f"../configs/datasets_config/{dataset_config_subdir}"
    else:
        config_path = "../configs/datasets_config/"
    with hydra.initialize(config_path, version_base=hydra.__version__):
        cfg_data = hydra.compose(config_name=cfg_exp["dataset"])
        cfg_data.datamodule.num_workers = num_cpus  # Overwrite number of cpus
        if cfg_data.get("exclude_id_pkl_path") is not None:
            with open(cfg_data.exclude_id_pkl_path, "rb") as fin:
                exclude_ids = pickle.load(fin)
            if cfg_data.datamodule.dataselector.exclude_ids is not None:
                cfg_data.datamodule.dataselector.exclude_ids += exclude_ids
            else:
                cfg_data.datamodule.dataselector.exclude_ids = exclude_ids
        log_info(f"Data config {cfg_data}")

    # create datamodule containing default train and val dataloader
    datamodule = hydra.utils.instantiate(cfg_data.datamodule)

    # Set logger
    wandb_logger = None
    if cfg_exp.log.log_wandb and not args.nolog:
        wandb_logger = WandbLogger(project=cfg_exp.log.wandb_project, name=run_name)
        # callbacks.append(LogEpochTimeCallback())
        callbacks.append(LogSetpTimeCallback())

    log_info(f"Using EMA with decay {cfg_exp.ema.decay}")
    callbacks.append(EMA(**cfg_exp.ema))

    # Set checkpointing
    if cfg_exp.log.checkpoint and not args.nolog:
        args_ckpt_last = {
            "dirpath": checkpoint_path_store,
            "save_weights_only": False,
            "filename": "ignore",
            "every_n_train_steps": cfg_exp.log.last_ckpt_every_n_steps,
            "save_last": True,
        }
        args_ckpt = {
            "dirpath": checkpoint_path_store,
            "save_last": False,
            "save_weights_only": False,
            "filename": "chk_{epoch:08d}_{step:012d}",
            "every_n_train_steps": cfg_exp.log.checkpoint_every_n_steps,
            "monitor": "train/trans_loss",
            "save_top_k": 10000,
            "mode": "min",
        }
        checkpoint_callback = EmaModelCheckpoint(**args_ckpt)
        checkpoint_callback_last = EmaModelCheckpoint(**args_ckpt_last)

        create_dir(checkpoint_path_store, parents=True, exist_ok=True)
        callbacks.append(checkpoint_callback)
        callbacks.append(checkpoint_callback_last)

        # Save and log config files
        path_configs = [
            (
                cfg_data,
                os.path.join(checkpoint_path_store, f"data_config_{run_name}.json"),
            ),
            (
                cfg_exp,
                os.path.join(checkpoint_path_store, f"exp_config_{run_name}.json"),
            ),
        ]
        store_configs(path_configs)
        log_configs(path_configs, wandb_logger)

    # Gradient and weight stats thoughout training, possibly skip updates with nan in grad
    if cfg_exp.opt.grad_and_weight_analysis:
        callbacks.append(GradAndWeightAnalysisCallback())
    if cfg_exp.opt.skip_nan_grad:
        callbacks.append(SkipNanGradCallback())

    # Define model
    model = Proteinb(cfg_exp, store_dir=root_run)

    # If LoRA is tunred on, replace Linear with LoRA layers
    if cfg_exp.get("lora") and cfg_exp.lora.get("r"):
        replace_lora_layers(
            model, cfg_exp.lora.r, cfg_exp.lora.lora_alpha, cfg_exp.lora.lora_dropout
        )
        lora.mark_only_lora_as_trainable(model, bias=cfg_exp.lora.train_bias)

    # If this is the first run for fine-tuning, load pre-trained checkpoint and don't load optimizer states
    pretrain_ckpt_path = cfg_exp.get("pretrain_ckpt_path", None)
    if last_ckpt_path is None and pretrain_ckpt_path is not None:
        log_info(f"Loading from pre-trained checkpoint path {pretrain_ckpt_path}")
        ckpt = torch.load(pretrain_ckpt_path, map_location="cpu")
        model.load_state_dict(ckpt["state_dict"], strict=False)

    # Train
    plugins = []
    show_prog_bar = args.show_prog_bar
    trainer = L.Trainer(
        max_epochs=cfg_exp.opt.max_epochs,
        accelerator=cfg_exp.hardware.accelerator,
        devices=cfg_exp.hardware.ngpus_per_node_,  # This is number of gpus per node, not total
        num_nodes=cfg_exp.hardware.nnodes_,
        callbacks=callbacks,
        logger=wandb_logger,
        log_every_n_steps=cfg_exp.opt.log_every_n_steps,
        default_root_dir=root_run,
        check_val_every_n_epoch=None,  # Leave like this
        val_check_interval=cfg_exp.opt.val_check_interval,
        strategy=cfg_exp.opt.dist_strategy,
        enable_progress_bar=show_prog_bar,
        plugins=plugins,
        accumulate_grad_batches=cfg_exp.opt.accumulate_grad_batches,
        num_sanity_val_steps=1,
        precision=precision,
        gradient_clip_algorithm="norm",
        gradient_clip_val=1.0,
    )
    trainer.fit(
        model, datamodule, ckpt_path=last_ckpt_path
    )  # If None then it starts from scratch
