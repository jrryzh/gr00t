#! /bin/bash

source ~/.bashrc
conda activate gr00t            # 或 module load xxx
cd /mnt/petrelfs/zhangjinyu/code_repo/gr00t-main

export WANDB_PROJECT=gr00t_finetune_select_drink      # 项目名
export WANDB_NAME=select_drink_1node_8gpu    # run 名（也就是你想设置的 task name）
export WANDB_RUN_GROUP=lerobot_h264_select_drink  # 可选：分组名，用于归类这次实验
export WANDB_JOB_TYPE=train                  # 可选：任务类型

python scripts/gr00t_finetune.py \
    --num_gpus 8 \
    --dataset-path data/demonstrations_lerobot/select_drink_backup/select_drink_backup_split_1_h264 \
    data/demonstrations_lerobot/select_drink_backup/select_drink_backup_split_2_h264 \
    data/demonstrations_lerobot/select_drink_backup/select_drink_backup_split_3_h264 \
    data/demonstrations_lerobot/select_drink_backup/select_drink_backup_split_4_h264 \
    data/demonstrations_lerobot/select_drink_backup/select_drink_backup_split_5_h264 \
    data/demonstrations_lerobot/select_drink_backup/select_drink_backup_split_6_h264 \
    data/demonstrations_lerobot/select_drink_backup/select_drink_backup_split_7_h264 \
    data/demonstrations_lerobot/select_drink_backup/select_drink_backup_split_8_h264 \
    data/demonstrations_lerobot/select_drink_backup/select_drink_backup_split_9_h264 \
    data/demonstrations_lerobot/select_drink_backup/select_drink_backup_split_10_h264 \
    --output-dir ./logs/debug \
    --batch_size 16 \
    --save-steps 50000 \
    --max-steps 1000000