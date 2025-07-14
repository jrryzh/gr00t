#! /bin/bash

source ~/.bashrc
conda activate gr00t            # 或 module load xxx
cd /mnt/petrelfs/zhangjinyu/code_repo/gr00t-main

export WANDB_PROJECT=gr00t_finetune_ball_longrange      # 项目名
export WANDB_NAME=ball_longrange_1node_8gpu    # run 名（也就是你想设置的 task name）
export WANDB_RUN_GROUP=lerobot_h264_ball_longrange  # 可选：分组名，用于归类这次实验
export WANDB_JOB_TYPE=train                  # 可选：任务类型

python scripts/gr00t_finetune.py \
    --dataset-path data/demonstrations_lerobot/bench_v6_ball_longrange/bench_v6_all_longrange_split0_h264 \
    data/demonstrations_lerobot/bench_v6_ball_longrange/bench_v6_all_longrange_split1_h264 \
    data/demonstrations_lerobot/bench_v6_ball_longrange/bench_v6_all_longrange_split2_h264 \
    data/demonstrations_lerobot/bench_v6_ball_longrange/bench_v6_all_longrange_split3_h264 \
    data/demonstrations_lerobot/bench_v6_ball_longrange/bench_v6_all_longrange_split4_h264 \
    --num-gpus 8 \
    --output-dir ./logs/bench_v6_ball_longrange_lerobot_h264 \
    --save-steps 10000 \
    --max-steps 200000 