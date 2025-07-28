# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# import debugpy 

# debugpy.listen(("0.0.0.0", 10093))  # 监听端口 
# print("Waiting for debugger to attach...")
# debugpy.wait_for_client()  # 等待 VS Code 附加

# a = 1


import os
import subprocess
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Literal
import json

import torch
import tyro
from transformers import TrainingArguments

from gr00t.data.dataset import LeRobotMixtureDataset, LeRobotSingleDataset
from gr00t.data.schema import EmbodimentTag
from gr00t.experiment.data_config import DATA_CONFIG_MAP
from gr00t.experiment.runner import TrainRunner
from gr00t.model.gr00t_n1 import GR00T_N1_5
from gr00t.model.transforms import EMBODIMENT_TAG_MAPPING
from gr00t.utils.peft import get_lora_model
from torch.serialization import add_safe_globals



@dataclass
class ArgsConfig:
    """Configuration for GR00T model fine-tuning."""

    # Dataset parameters
    dataset_path: List[str]
    """Path to the dataset directory or directories"""

    output_dir: str = "./output/gr00t_finetune"
    """Directory to save model checkpoints."""

    data_config: str = "gemanip"
    """Data configuration name from DATA_CONFIG_MAP."""

    # Training parameters
    batch_size: int = 32 # 60G memory
    """Batch size per GPU for training."""

    max_steps: int = 10000
    """Maximum number of training steps."""

    num_gpus: int = 1
    """Number of GPUs to use for training."""

    save_steps: int = 5000
    """Number of steps between saving checkpoints."""

    # Model parameters
    base_model_path: str = "/mnt/petrelfs/zhangjinyu/model_zoo/nvidia/GR00T-N1.5-3B"
    """Path or HuggingFace model ID for the base model."""

    tune_llm: bool = False
    """Whether to fine-tune the language model backbone."""

    tune_visual: bool = False
    """Whether to fine-tune the vision tower."""

    tune_projector: bool = True
    """Whether to fine-tune the projector."""

    tune_diffusion_model: bool = True
    """Whether to fine-tune the diffusion model."""

    resume: bool = False
    """Whether to resume from a checkpoint."""

    # Advanced training parameters
    learning_rate: float = 1e-4
    """Learning rate for training."""

    weight_decay: float = 1e-5
    """Weight decay for AdamW optimizer."""

    warmup_ratio: float = 0.05
    """Ratio of total training steps used for warmup."""

    lora_rank: int = 0
    """Rank for the LORA model. If 0, no LORA will be used."""

    lora_alpha: int = 64
    """Alpha value for the LORA model."""

    lora_dropout: float = 0.1
    """Dropout rate for the LORA model."""

    lora_full_model: bool = False
    """Whether to use the full model for LORA. If False, only the action head will be trained."""

    dataloader_num_workers: int = 8 # default 8
    """Number of workers for data loading."""

    report_to: Literal["wandb", "tensorboard"] = "wandb"
    """Where to report training metrics (e.g., 'wandb', 'tensorboard')."""

    # Data loading parameters
    embodiment_tag: Literal[tuple(EMBODIMENT_TAG_MAPPING.keys())] = "new_embodiment"
    """Embodiment tag to use for training. e.g. 'new_embodiment', 'gr1'"""

    video_backend: Literal["decord", "torchvision_av"] = "decord"
    """Video backend to use for training. [decord, torchvision_av]"""

    # Mixture dataset parameters
    balance_dataset_weights: bool = True
    """Used in LeRobotMixtureDataset. If True, we will balance the dataset weights, by multiplying the total trajectory to each dataset"""

    # Mixture dataset parameters
    balance_trajectory_weights: bool = True
    """Used in LeRobotMixtureDataset. If True, sample trajectories within a dataset weighted by their length; otherwise, equal weighting."""

    # JINYU: these are added for multi-node training according to gpt
    num_nodes: int = int(os.getenv("SLURM_NNODES", 1))
    """Total number of nodes involved in training."""

    node_rank: int = int(os.getenv("SLURM_NODEID", 0))
    """Rank of this node (0–num_nodes-1)."""

    master_addr: str = os.getenv("MASTER_ADDR", "127.0.0.1")
    """Rendezvous master IP / hostname."""

    master_port: int = int(os.getenv("MASTER_PORT", 29500))
    """Rendezvous master port."""
    
    deepspeed_config: str = ""
    """Deepspeed config file."""

#####################################################################################
# main training function
#####################################################################################


def main(config: ArgsConfig):
    """Main training function."""
    # 保存 config 到 output_dir
    os.makedirs(config.output_dir, exist_ok=True)
    config_path = os.path.join(config.output_dir, "config.json")
    with open(config_path, "w") as f:
        json.dump(asdict(config), f, indent=2)
    print(f"Config saved to {config_path}")

    # ------------ step 1: load dataset ------------
    embodiment_tag = EmbodimentTag(config.embodiment_tag)

    # 1.1 modality configs and transforms
    print(f"Loading dataset with data_config: {config.data_config}")
    print(f"DATA_CONFIG_MAP: {DATA_CONFIG_MAP.keys()}")
    data_config_cls = DATA_CONFIG_MAP[config.data_config]
    modality_configs = data_config_cls.modality_config()
    transforms = data_config_cls.transform()

    # 1.2 data loader: we will use either single dataset or mixture dataset
    if len(config.dataset_path) == 1:
        train_dataset = LeRobotSingleDataset(
            dataset_path=config.dataset_path[0],
            modality_configs=modality_configs,
            transforms=transforms,
            embodiment_tag=embodiment_tag,  # This will override the dataset's embodiment tag to "new_embodiment"
            video_backend=config.video_backend,
        )
    else:
        single_datasets = []
        for p in config.dataset_path:
            assert os.path.exists(p), f"Dataset path {p} does not exist"
            ## We use the same transforms, modality configs, and embodiment tag for all datasets here,
            ## in reality, you can use dataset from different modalities and embodiment tags
            dataset = LeRobotSingleDataset(
                dataset_path=p,
                modality_configs=modality_configs,
                transforms=transforms,
                embodiment_tag=embodiment_tag,
                video_backend=config.video_backend,
            )
            single_datasets.append(dataset)

        train_dataset = LeRobotMixtureDataset(
            data_mixture=[
                (dataset, 1.0)  # we will use equal weights for all datasets
                for dataset in single_datasets
            ],
            mode="train",
            balance_dataset_weights=config.balance_dataset_weights,
            balance_trajectory_weights=config.balance_trajectory_weights,
            seed=42,
            metadata_config={
                "percentile_mixing_method": "weighted_average",
            },
        )
        print(f"Loaded {len(single_datasets)} datasets, with {config.dataset_path} ")

    # ------------ step 2: load model ------------
    model = GR00T_N1_5.from_pretrained(
        pretrained_model_name_or_path=config.base_model_path,
        tune_llm=config.tune_llm,  # backbone's LLM
        tune_visual=config.tune_visual,  # backbone's vision tower
        tune_projector=config.tune_projector,  # action head's projector
        tune_diffusion_model=config.tune_diffusion_model,  # action head's DiT
    )

    # Set the model's compute_dtype to bfloat16
    model.compute_dtype = "bfloat16"
    model.config.compute_dtype = "bfloat16"

    if config.lora_rank > 0:
        model = get_lora_model(
            model,
            rank=config.lora_rank,
            lora_alpha=config.lora_alpha,
            lora_dropout=config.lora_dropout,
            action_head_only=not config.lora_full_model,
        )

    # 2.1 modify training args
    training_args = TrainingArguments(
        output_dir=config.output_dir,
        run_name=None,
        remove_unused_columns=False,
        deepspeed=config.deepspeed_config,
        gradient_checkpointing=False,
        bf16=True,
        tf32=True,
        per_device_train_batch_size=config.batch_size,
        gradient_accumulation_steps=1,
        dataloader_num_workers=config.dataloader_num_workers,
        dataloader_pin_memory=False,
        dataloader_persistent_workers=config.dataloader_num_workers > 0,
        optim="adamw_torch",
        adam_beta1=0.95,
        adam_beta2=0.999,
        adam_epsilon=1e-8,
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
        warmup_ratio=config.warmup_ratio,
        lr_scheduler_type="cosine",
        logging_steps=10.0,
        num_train_epochs=300,
        max_steps=config.max_steps,
        save_strategy="steps",
        save_steps=config.save_steps,
        # evaluation_strategy="no",
        save_total_limit=8,
        report_to=config.report_to,
        seed=42,
        do_eval=False,
        ddp_find_unused_parameters=False,
        ddp_bucket_cap_mb=100,
        torch_compile_mode=None,
    )

    # 2.2 run experiment
    experiment = TrainRunner(
        train_dataset=train_dataset,
        model=model,
        training_args=training_args,
        resume_from_checkpoint=config.resume,
    )

    # 2.3 run experiment
    experiment.train()

import os, sys, subprocess, time
from pathlib import Path
import torch
import tyro
from dataclasses import asdict

def already_distributed() -> bool:
    """判断当前进程是否已由 torchrun / torch.distributed.elastic 启动"""
    return int(os.getenv("WORLD_SIZE", "1")) > 1 or os.getenv("TORCHELASTIC_RUN_ID") is not None

def launch_single_node_torchrun(config):
    script_path = Path(__file__).absolute()
    cmd = [
        "torchrun",
        f"--nnodes=1",
        f"--nproc-per-node={config.num_gpus}",
        "--rdzv_backend=c10d",
        f"--rdzv_endpoint=127.0.0.1:{config.master_port}",
        "--rdzv_id", f"local-{int(time.time())}",
        str(script_path),
    ]
    # 将 ArgsConfig 重新展开为 CLI
    for k, v in asdict(config).items():
        cli_key = f"--{k.replace('_', '-')}"
        if isinstance(v, bool):
            cmd.append(cli_key) if v else cmd.append(f"--no-{cli_key.lstrip('--')}")
        elif isinstance(v, list):
            cmd.append(cli_key)
            cmd += map(str, v)
        else:
            cmd += [cli_key, str(v)]
    print("Launching (single-node) torchrun:", " ".join(cmd))
    os.execvp(cmd[0], cmd)      # 用 exec 替换当前进程，避免多余子进程

if __name__ == "__main__":
    # 1️⃣ 解析 CLI
    config = tyro.cli(ArgsConfig)

    # 2️⃣ 设备/参数检查
    config.num_gpus = int(config.num_gpus)           # 防止字符串
    available = torch.cuda.device_count()
    assert 0 < config.num_gpus <= available, f"num_gpus={config.num_gpus}, but only {available} visible"

    # 3️⃣ 决定启动路径
    if already_distributed():
        # 已由 torchrun/srun 启动 → 直接进入主程序
        print("[INFO] Detected distributed environment.")
        main(config)

    elif config.num_gpus == 1:
        print("[INFO] Single-GPU mode (no torchrun).")
        main(config)

    else:
        # 单机多卡 → 本脚本生成 torchrun（1 节点）
        launch_single_node_torchrun(config)
