#! /bin/bash

source ~/.bashrc
conda activate gr00t            # 或 module load xxx
cd /mnt/petrelfs/zhangjinyu/code_repo/Isaac-GR00T

export WANDB_PROJECT=3L_realdemonstrations      # 项目名
export WANDB_NAME=3L_realdemonstrations_debug    # run 名（也就是你想设置的 task name）

python scripts/gr00t_finetune.py \
    --num_gpus 8 \
    --dataset-path data/demonstrations_lerobot/3L_real_demonstrations \
    --output-dir ./logs/debug \
    --data_config franka_ee_robotiq_v2 \
    --batch_size 48 \
    --save-steps 5000 \
    --max-steps 30000